"""Baseline Details
- For all segments sampled from the video, we use the CLIP model to get the segment embeddings.
- For each question, we get the text embeddings.
- We calculate the similarity between the segment and text embeddings.
- We use the similarity score to rank the segments.
- We return the top-k segments as the answer. (or Pass them to the LLM model)
"""
from datetime import datetime
from typing import List

import cv2
from transformers import __version__ as transformers_version

if transformers_version != "4.47.0":
	raise ValueError(f"Please install transformers==4.47.0. Current version: {transformers_version}")

import argparse
import math
import json
import os
import re
import sys
import uuid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from mm_encoder.videoclip_xl_utils.text_encoder import text_encoder

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

from decord import cpu, VideoReader, gpu  # @manual=fbsource//third-party/pypi/decord:decord or pip install eva-decord==0.6.1


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
	"""
	This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
	model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
	"""
	square_tensor = torch.pow(tensor, 2)
	sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
	normed_tensor = torch.pow(sum_tensor, 0.5)
	return normed_tensor


def plot_seg_sim_scores(sample_id, ques, similarity_scores, plot_dir):
	"""Line plot of similarity scores"""
	plt.figure(figsize=(10, 5))
	sns.lineplot(x=[i for i in range(len(similarity_scores))], y=similarity_scores)
	plt.title(f"Similarity Scores for Sample ID: {sample_id}\nQuestion: {ques}")
	plt.xlabel("Segment Index")
	plt.ylabel("Similarity Score")
	plt.grid()
	plt.savefig(os.path.join(plot_dir, f"{sample_id}.png"))
	plt.close()


def load_videoclip_model(retrieval_model_path: str):
	"""
	Load the VideoCLIP model
	:param retrieval_model_path:
	:return:
	"""
	from mm_encoder.videoclip_xl_modeling import VideoCLIP_XL
	model = VideoCLIP_XL()
	state_dict = torch.load(retrieval_model_path)
	model.load_state_dict(state_dict)
	
	# Get number of parameters in the model
	num_params = sum(p.numel() for p in model.parameters())
	num_params = num_params / 1e6
	print(f"Number of parameters for the model {retrieval_model_path}: {num_params}M")
	return model


def normalize(data):
	v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
	v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
	return (data / 255.0 - v_mean) / v_std


def get_video_embeddings(videoclip_xl, batch_segments, fnum=8, gpu_idx=0) -> List[torch.Tensor]:
	def preprocess(segment):
		frames = [segment[i] for i in range(len(segment))]
		step = len(frames) // fnum
		frames = frames[::step][:fnum]
		
		vid_tube = []
		for fr in frames:
			fr = fr[:, :, ::-1]  # BGR to RGB
			fr = cv2.resize(fr, (224, 224))  # Resize to 224x224
			fr = np.expand_dims(normalize(fr), axis=(0, 1))
			vid_tube.append(fr)
			
		vid_tube = np.concatenate(vid_tube, axis=1)
		vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
		vid_tube = torch.from_numpy(vid_tube)
		return vid_tube.float()
	
	video_inputs = [preprocess(segment) for segment in batch_segments]
	video_inputs = torch.cat(video_inputs, dim=0).float()
	video_inputs = video_inputs.cuda(gpu_idx)
	with torch.no_grad():
		video_embeds = videoclip_xl.vision_model.get_vid_features(video_inputs).float()
		video_embeds = [video_embeds[i]/_get_vector_norm(video_embeds[i]) for i in range(len(video_embeds))]
	return video_embeds


def get_text_embeddings(videoclip_xl, text_encoder, text, gpu_idx=0):
	with torch.no_grad():
		text_inputs = text_encoder.tokenize(text, truncate=True)
		text_inputs = text_inputs.cuda(gpu_idx)
		text_embeds = videoclip_xl.text_model.encode_text(text_inputs)
		text_embeds /= _get_vector_norm(text_embeds)
	return text_embeds


def predict(args, annotations: dict) -> None:
	# dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
	version = args.version
	
	model_name = args.model_name
	model_path = args.model_path
	# torch.distributed.barrier()
	tokenizer, model, image_processor, context_len = load_pretrained_model(
		model_path=model_path,  # pyre-fixme
		model_base=None,
		model_name=model_name,
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
	ret_model = load_videoclip_model(args.retrieval_model_path)
	if args.gpu_idx_ret is not None:
		ret_model.cuda(args.gpu_idx_ret).eval()  # Set the model to evaluation mode
	else:
		ret_model.eval()
	print("[INFO] Retrieval Model Loaded Successfully")
	
	results = {}
	for video_id in annotations:
		
		# # For debugging
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
		frame_idxs = [i for i in range(0, len(vr), sampling_rate)]  # Between 2 frames, 2 seconds elapsed
		
		# ######################################## Extract segments ######################################## #
		try:
			assert len(frame_idxs) > args.m
		except AssertionError:
			print("[Error] Segment length is greater than the number of frames in the video")
			sys.exit(1)
		
		segments = []
		for i in range(0, len(frame_idxs), args.n):
			start = i
			end = min(i + args.m, len(frame_idxs))
			segment = frame_idxs[start:end]
			
			try:
				assert len(segment) == args.m
			except AssertionError:
				print(f"[INFO] At idx={i} Segment length (={len(segment)}) is not equal to {args.m}")
				# Manually create a segment by considering last m frames
				segment = frame_idxs[-args.m:]
			segments.append(segment)
		segment_idxs = [i for i in range(len(segments))]
		print(f"[INFO] {len(segment_idxs)} Segments Extracted Successfully")

		# # Compute segment embeddings
		# segment_embeddings = []
		# for segment in tqdm(segments, desc="Extracting Segment Embeddings", total=len(segments)):
		# 	segment = vr.get_batch(segment).asnumpy()  # Expensive step
		# 	seg_embed = get_video_embeddings(ret_model, segment, fnum=args.n_frames, gpu_idx=args.gpu_idx)
		# 	segment_embeddings.append(seg_embed)
		# [Output] -> So far we have segments, segment_idxs, and embeddings
		
		# Optimised code to consider a set of segments at once
		segment_embeddings = []
		for i in tqdm(range(0, len(segments), args.fwd_batch_size_retrieval), desc="Extracting Segment Embeddings", total=math.ceil(len(segments)/args.fwd_batch_size_retrieval)):
			batch_segments = segments[i:i + args.fwd_batch_size_retrieval]
			batch_segments = [vr.get_batch(seg).asnumpy() for seg in batch_segments]
			batch_embeddings = get_video_embeddings(ret_model, batch_segments, fnum=args.n_frames, gpu_idx=args.gpu_idx_ret)
			segment_embeddings.extend(batch_embeddings)
		
		for qa_sample in annotations[video_id]:
			_id, ques, ans = qa_sample["sample_id"], qa_sample["question"], qa_sample["answer"]
			print(f"Processing: {_id}")
			
			# Get text embeddings
			ques_embed = get_text_embeddings(ret_model, text_encoder, ques, gpu_idx=args.gpu_idx_ret)
			
			# Calculate similarity between the segment and text embeddings
			similarity_scores = []
			for seg_embed in tqdm(segment_embeddings, desc="Calculating Similarity Scores", total=len(segment_embeddings)):
				similarity_score = torch.matmul(ques_embed, seg_embed.t())
				similarity_scores.append(similarity_score.detach().cpu().item())
				
			# Let's plot the distribution of similarity scores
			plot_seg_sim_scores(_id, ques, similarity_scores, args.plot_dir)
			
			# Sort the segment based on similarity scores and get top-k segment indices
			sorted_segment_scores = sorted(zip(segment_idxs, segments, similarity_scores),
										   key=lambda x: x[-1], reverse=True)
			top_k_segments = [seg_idx for seg_idx, _, _ in sorted_segment_scores[:args.top_k]]
			top_k_segments_frames = []
			for seg_idx in top_k_segments:
				top_k_segments_frames += segments[seg_idx]
			
			# Use the top-k segment's frames to get the video
			video = vr.get_batch(top_k_segments_frames).asnumpy()  # Expensive step
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
			
			# # [Debug] Print the cuda device of the input_ids and video
			# print(f"[DEBUG] input_ids.device: {input_ids.device}")
			# print(f"[DEBUG] video.device: {[v.device for v in video]}")
			
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
						"top_{}_segments".format(args.top_k): sorted_segment_scores[:args.top_k],
					},
				}
			)
	
	print(json.dumps(results, indent=4))
	save_at = f"{args.logging_dir}/results_{model_name}_baseline_top{args.top_k}_segment_VideoCLIP-XL_{args.m}m_{args.n}n_retrieval.json"
	with open(save_at, "w") as f:
		json.dump(results, f, indent=4)


if __name__ == "__main__":
	# Current time stamp
	current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
	logging_dir = f"logging/{current_time}"
	if not os.path.exists(logging_dir):
		os.makedirs(logging_dir)
	
	# # For Open QA
	video_dir = '../ext_storage/Ego4D/hierarchical-emv/v2/full_scale'
	annotations_path = '../ext_storage/Ego4D/my_hierarchical-emv_qa.json'
	with open(annotations_path) as f:
		annotations = json.load(f)
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--model_path',
						default="./checkpoints/LongVU_Llama3_2_3B")  # LongVU_Qwen2_7B, LongVU_Llama3_2_3B
	parser.add_argument('--model_name', default="cambrian_llama3")  # cambrian_qwen, cambrian_llama3
	parser.add_argument('--retrieval_model_path', default="./checkpoints/VideoCLIP-XL.bin")  # VideoCLIP-XL, VideoCLIP-XL-v2
	parser.add_argument('--version', default="llama3_2")  # qwen, llama3_2
	parser.add_argument('--local-rank', default=0)
	parser.add_argument('--qa_type', default="open", choices=["open", "closed"])
	parser.add_argument('--video_dir', default=video_dir)
	parser.add_argument('--video_ext', default=".mp4")
	parser.add_argument('--max_frame_limit', type=int, default=500, help="[Not Used]")
	parser.add_argument('--top_k', type=int, default=1, help='prefer 1 for segment-level')
	
	parser.add_argument('--m', type=int, default=10, help='segment length in terms of frames')
	parser.add_argument('--n', type=int, default=10, help='sample segment every n frames')
	parser.add_argument('--n_frames', type=int, default=8, help='Number of frames the video embedding model can handle. VideoCLIP-XL can handle 8 frames (not more, not less)')
	
	parser.add_argument('--fwd_batch_size_retrieval', type=int, default=4, help='Number of segments to process at once')
	parser.add_argument('--gpu_idx_qa', type=int, help='GPU ID', default=1)
	parser.add_argument('--gpu_idx_ret', type=int, help='GPU ID', default=2)
	parser.add_argument('--logging_dir', default=logging_dir)
	parser.add_argument('--plot_dir', default=os.path.join(logging_dir, "plots"))
	args = parser.parse_args()
	
	if not os.path.exists(args.plot_dir):
		os.makedirs(args.plot_dir)
	
	if "llama3" in args.version:
		args.model_name = "cambrian_llama3"
	
	predict(args, annotations)

