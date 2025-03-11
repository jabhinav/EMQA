"""Baseline Details
- For all images sampled from the video, we use the CLIP model to get the image embeddings.
- For each question, we get the text embeddings.
- We calculate the similarity between the image and text embeddings.
- We use the similarity score to rank the images.
- We return the top-k images as the answer. (or Pass them to the LLM model)
"""
import math

from transformers import __version__ as transformers_version
if transformers_version != "4.47.0":
	raise ValueError(f"Please install transformers==4.47.0. Current version: {transformers_version}")

import argparse
import json
import os
import re
import sys
import uuid

from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# sys.path.append('baselines/LongVU/')

import torch

from videoqa_sys.longvu.builder import load_pretrained_model
from videoqa_sys.longvu.constants import (
	DEFAULT_IM_END_TOKEN,
	DEFAULT_IM_START_TOKEN,
	DEFAULT_IMAGE_TOKEN,
	IMAGE_TOKEN_INDEX,
)
from videoqa_sys.longvu.conversation import conv_templates, SeparatorStyle
from videoqa_sys.longvu.mm_datautils import (
	process_images,
	tokenizer_image_token,
)
from videoqa_sys.longvu.mm_utils import KeywordsStoppingCriteria

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord or pip install eva-decord==0.6.1


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
	"""
	This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
	model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
	"""
	square_tensor = torch.pow(tensor, 2)
	sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
	normed_tensor = torch.pow(sum_tensor, 0.5)
	return normed_tensor


def load_retrieval_model(retrieval_model_path: str = "openai/clip-vit-large-patch14"):
	"""
	The retrieval model must embed both images and text into the same embedding space
	:param retrieval_model_path:
	:return:
	"""
	model = CLIPModel.from_pretrained(retrieval_model_path)
	processor = CLIPProcessor.from_pretrained(retrieval_model_path)
	
	# Get number of parameters in the model
	num_params = sum(p.numel() for p in model.parameters())
	# Print the number of parameters in terms of millions
	num_params = num_params / 1e6
	print(f"Number of parameters for the model {retrieval_model_path}: {num_params}M")
	
	return model, processor


def get_image_embeddings(model, processor, image, gpu_idx=0):
	image_processed = processor(images=image, return_tensors="pt")
	image_processed = {k: v.cuda(gpu_idx) for k, v in image_processed.items()}
	with torch.no_grad():
		image_embeds = model.get_image_features(**image_processed)
		# normalized features
		image_embeds = image_embeds / _get_vector_norm(image_embeds)
	return image_embeds


def get_text_embeddings(model, processor, text, gpu_idx=0):
	text_processed = processor(text=text, return_tensors="pt", padding=True)
	text_processed = {k: v.cuda(gpu_idx) for k, v in text_processed.items()}
	with torch.no_grad():
		text_embeds = model.get_text_features(**text_processed)
		# normalized features
		text_embeds = text_embeds / _get_vector_norm(text_embeds)
	return text_embeds


def predict(args, annotations: dict) -> None:
	# dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
	version = args.version

	model_name = args.model_name
	model_path = args.model_path
	# torch.distributed.barrier()
	tokenizer, model, image_processor, context_len = load_pretrained_model(
		model_path,  # pyre-fixme
		None,
		model_name,
		device_map=None,
	)
	model.get_model().config.drop_threshold = 0.8
	model.config.use_cache = True
	# Either manually set the device or specify the device arg in load_pretrained_model.
	# Its default is 'cuda'. For custom, set it to 'cuda:args.gpu_idx_qa'
	model.cuda(args.gpu_idx_qa)
	for vision_tower_aux in model.get_vision_tower_aux_list():
		vision_tower_aux.cuda(args.gpu_idx_qa)
	print("[INFO] Model Loaded Successfully")
	
	# Load the embedding/retrieval model
	ret_model, ret_processor = load_retrieval_model(args.retrieval_model_path)
	if args.gpu_idx_ret is not None:
		ret_model.cuda(args.gpu_idx_ret).eval()  # Set the model to evaluation mode
	else:
		ret_model.eval()
	
	results = {}
	for video_id in annotations:
		
		# For debugging
		# if not video_id == 'c4333895-ed19-42fe-9323-271a41bdfe4c':
		# 	continue
		
		results[video_id] = []
		
		video_path = os.path.join(args.video_dir, video_id)
		video_path += args.video_ext
		if not os.path.exists(video_path):
			raise FileNotFoundError(f"Video file not found at {video_path}")
		
		vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)  # vr is a VideoReader object
		print(f"[INFO] Video {video_id} Loaded Successfully")
		
		# Get frames. Sampling at 2xfps.
		fps = round(vr.get_avg_fps())
		sampling_rate = round(fps / 0.5)
		frame_idxs = [i for i in range(0, len(vr), sampling_rate)]
		
		# Limit to max_frames frames.
		# [Comment it! We want to use cosine similarity to select the frames instead of thresholding the max frames]
		selected_frames = frame_idxs
		if len(selected_frames) > args.max_frame_limit:
			selected_frames = [
				selected_frames[i]
				for i in range(0, len(selected_frames), math.ceil(len(selected_frames) / args.max_frame_limit))
			]
		print(f"[INFO] #{len(selected_frames)} Frames Extracted Successfully")
		
		frame_embeddings = []
		for frame_idx in tqdm(selected_frames, desc="Extracting Frame Embeddings", total=len(selected_frames)):
			frame = vr.get_batch([frame_idx]).asnumpy()  # Expensive step
			frame_embed = get_image_embeddings(ret_model, ret_processor, frame, gpu_idx=args.gpu_idx_ret)
			frame_embeddings.append(frame_embed)

		for qa_sample in annotations[video_id]:
			_id, ques, ans = qa_sample["sample_id"], qa_sample["question"], qa_sample["answer"]
			print(f"Processing: {_id}")

			# Get text embeddings
			ques_embed = get_text_embeddings(ret_model, ret_processor, ques, gpu_idx=args.gpu_idx_ret)

			# Calculate similarity between the frame and text embeddings
			similarity_scores = []
			for frame_idx, frame_embed in tqdm(zip(selected_frames, frame_embeddings), desc="Calculating Similarity Scores)", total=len(selected_frames)):
				similarity_score = torch.matmul(ques_embed, frame_embed.t())
				similarity_scores.append(similarity_score.detach().cpu().item())

			# Sort the frames based on similarity scores and get top-k frame indices
			sorted_frames_scores = sorted(zip(selected_frames, similarity_scores), key=lambda x: x[1], reverse=True)
			top_k_frames = [frame_idx for frame_idx, _ in sorted_frames_scores[:args.top_k]]
			top_k_frames = sorted(top_k_frames)

			# Use the top-k frames to get the video
			video = vr.get_batch(top_k_frames).asnumpy()  # Expensive step
			image_sizes = [video[0].shape[:2]]
			video = process_images(video, image_processor, model.config, args.gpu_idx_qa)
			video = [item.unsqueeze(0) for item in video]
			print("[INFO] Frames Processed Successfully")

			qs = f"Question: {ques}\nAnswer: "
			if getattr(model.config, "mm_use_im_start_end", False):
				qs = (
						DEFAULT_IM_START_TOKEN
						+ DEFAULT_IMAGE_TOKEN
						+ DEFAULT_IM_END_TOKEN
						+ "\n"
						+ qs
				)
			else:
				qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

			conv = conv_templates[version].copy()
			conv.append_message(conv.roles[0], qs)
			conv.append_message(conv.roles[1], None)
			prompt = conv.get_prompt()

			input_ids = (
				tokenizer_image_token(
					prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
				)
				.unsqueeze(0)
				.cuda(args.gpu_idx_qa)
			)

			if "llama3" in version:
				input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos

			stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
			keywords = [stop_str]
			stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

			with torch.inference_mode():
				output_ids = model.generate(
					input_ids,
					images=video,
					image_sizes=image_sizes,
					do_sample=False,
					temperature=0.0,
					max_new_tokens=100,  # Set to 5 for Closed AQ and 100 for Open AQ
					use_cache=True,
					stopping_criteria=[stopping_criteria],
				)
			if isinstance(output_ids, tuple):
				output_ids = output_ids[0]
			pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
			if pred.endswith(stop_str):
				pred = pred[: -len(stop_str)]
				pred = pred.strip()
			pred = pred.replace("Answer", "")

			if args.qa_type == "closed":
				letters = ["A", "B", "C", "D", "E"]
				pred_answer = re.findall("[\(\ ]*[A-E][\)\ ]*", pred)
				pred_answer = pred_answer[0].strip()
				pred_answer = pred_answer.strip("()")
				if pred_answer in letters:
					pred_idx = letters.index(pred_answer)
					pred = letters[pred_idx]
				else:
					print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
					pred_idx = 2
					pred = letters[pred_idx]
				ans_id = uuid.uuid4()

			results[video_id].append(
				{
					"sample_id": _id,
					"question-string": qs,
					"prompt": prompt,
					"pred": pred,
					"model_id": model_name,
					"metadata": {
						"ret_model_id": args.retrieval_model_path,
						"top_{}_frames".format(args.top_k): sorted_frames_scores[:args.top_k],
					},
				}
			)

	print(json.dumps(results, indent=4))
	save_at = f"results_{model_name}_baseline_top{args.top_k}_frame_retrieval.json"
	with open(save_at, "w") as f:
		json.dump(results, f, indent=4)
			
			
if __name__ == "__main__":
	
	# # For Open QA
	video_dir = '../ext_storage/Ego4D/hierarchical-emv/v2/full_scale'
	annotations_path = '../ext_storage/Ego4D/my_hierarchical-emv_qa.json'
	with open(annotations_path) as f:
		annotations = json.load(f)
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--model_path',
						default="./checkpoints/LongVU_Llama3_2_3B")  # LongVU_Qwen2_7B, LongVU_Llama3_2_3B
	parser.add_argument('--model_name', default="cambrian_llama3")  # cambrian_qwen, cambrian_llama3
	parser.add_argument('--retrieval_model_path', default="openai/clip-vit-large-patch14")
	parser.add_argument('--version', default="llama3_2")  # qwen, llama3_2
	parser.add_argument('--local-rank', default=0)
	parser.add_argument('--qa_type', default="open", choices=["open", "closed"])
	parser.add_argument('--video_dir', default=video_dir)
	parser.add_argument('--video_ext', default=".mp4")
	parser.add_argument('--max_frame_limit', type=int, default=500)
	parser.add_argument('--top_k', type=int, default=10)
	
	parser.add_argument('--gpu_idx_qa', type=int, help='GPU ID', default=1)
	parser.add_argument('--gpu_idx_ret', type=int, help='GPU ID', default=2)
	
	args = parser.parse_args()
	
	if "llama3" in args.version:
		args.model_name = "cambrian_llama3"
	
	predict(args, annotations)

