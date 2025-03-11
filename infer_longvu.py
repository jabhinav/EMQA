import json
import os
import re
import uuid
import argparse

import sys

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

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord
# from torch import distributed as dist


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
	model.cuda()
	print("[INFO] Model Loaded Successfully")
	
	results = {}
	for video_id in annotations:
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
		frame_idx = [i for i in range(0, len(vr), sampling_rate)]
		# Limit to args.max_frame_limit frames.
		if len(frame_idx) > args.max_frame_limit:
			frame_idx = [
				frame_idx[i]
				for i in range(0, len(frame_idx), len(frame_idx) // args.max_frame_limit)
			]
		video = vr.get_batch(frame_idx).asnumpy()  # Expensive step
		image_sizes = [video[0].shape[:2]]
		print(f"[INFO] Frames #{len(frame_idx)} Extracted Successfully")
	
		# world_size = torch.distributed.get_world_size()
		# world_rank = torch.distributed.get_rank()
		# torch.distributed.barrier()
		video = process_images(video, image_processor, model.config)
		video = [item.unsqueeze(0) for item in video]
		print("[INFO] Frames Processed Successfully")
		
		for qa_sample in annotations[video_id]:
			_id, ques, ans = qa_sample["sample_id"], qa_sample["question"], qa_sample["answer"]
			print(f"Processing: {_id}")
			
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
				.cuda()
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
					"metadata": {},
				}
			)
	
	# global_rank = dist.get_rank()
	# if global_rank == 0:
	print(json.dumps(results, indent=4))
	save_at = f"results_{model_name}.json"
	with open(save_at, "w") as f:
		json.dump(results, f, indent=4)


if __name__ == "__main__":
	
	# # For Closed QA
	# question = 'What key adjustments did c make to the lawn mower, and how can those adjustments be concisely described as a whole?'
	# a0 = "C adjusted the lawn mower's engine, pipe, and bottom."
	# a1 = "C adjusted the lawn mower's wheels."
	# a2 = "Carefully, c adjusted the lawn mower's sharp blades to achieve optimal cutting height."
	# a3 = "Carefully, c adjusted the height of the lawn mower's handle for more comfortable use."
	# a4 = "Carefully, c adjusted the lawn mower's comfortable seat for a better fit."
	# prompt = f"Question: {question}\nOptions:\n(A) {a0}\n(B) {a1}\n(C) {a2}\n(D) {a3}\n(E) {a4}\nRespond with only the letter (A, B, C, D or E) of the correct option."
	
	# # For Open QA
	video_dir = '../ext_storage/Ego4D/hierarchical-emv/v2/full_scale'
	annotations_path = '../ext_storage/Ego4D/my_hierarchical-emv_qa.json'
	with open(annotations_path) as f:
		annotations = json.load(f)
	parser = argparse.ArgumentParser()
	
	parser.add_argument('--model_path', default="./checkpoints/LongVU_Llama3_2_3B")  # LongVU_Qwen2_7B, LongVU_Llama3_2_3B
	parser.add_argument('--model_name', default="cambrian_llama3")  # cambrian_qwen, cambrian_llama3
	parser.add_argument('--version', default="llama3_2")  # qwen, llama3_2
	parser.add_argument('--local-rank', default=0)
	parser.add_argument('--qa_type', default="open", choices=["open", "closed"])
	parser.add_argument('--video_dir', default=video_dir)
	parser.add_argument('--video_ext', default=".mp4")
	parser.add_argument('--max_frame_limit', type=int, default=500)
	
	args = parser.parse_args()
	
	if "llama3" in args.version:
		args.model_name = "cambrian_llama3"
	
	predict(args, annotations)
