# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.

# Check the version of transformers and make sure it is 4.47.0
from transformers import __version__ as transformers_version
if transformers_version != "4.47.0":
	raise ValueError(f"Please install transformers==4.47.0. Current version: {transformers_version}")

import json
import os
import re
import uuid
import argparse
import math

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

from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord  or pip install eva-decord==0.6.1
from typing import List

# from torch import distributed as dist


def select_window_aroung_GT(
		gt_frames: List[int],
		frames: List[int],
		window_size,
		debug=False
) -> List[int]:
	if len(gt_frames) == 0 or gt_frames is None:
		raise ValueError("No G.T. frames provided")
	
	selected_frames = []
	for gt_frame in gt_frames:
		
		# Locate and then select +- window_size frames around gt_frame
		for i in range(len(frames) - 1):
			if frames[i] <= gt_frame <= frames[i + 1]:
				break
		
		# Edge-case
		if frames[-1] < gt_frame:
			raise ValueError(f"G.T. frame {gt_frame} out of bounds (Last frame = {frames[i]})")
		
		start = max(0, i - window_size + 1)  # To enforce selection:  i - window_size + 1: i + 1
		end = min(len(frames) - 1, i + window_size + 1)  # To enforce selection: i + 1: i + window_size + 1
		if frames[start] != gt_frame and frames[end] != gt_frame:
			selected_frames.extend(frames[start: i + 1])
			selected_frames.append(gt_frame)
			selected_frames.extend(frames[i + 1: end])
		else:
			selected_frames.extend(frames[start:end])
	
	selected_frames = set(selected_frames)
	selected_frames = sorted(list(selected_frames))
	if debug:
		print()
		print(f"[INFO] Selected {selected_frames} frames for the G.T. frames {gt_frames}")
		print()
	return selected_frames
	


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
		
		# # For debugging
		# if not video_id == '6b0f2b6d-74aa-466e-a8b8-762223c4d581':
		# 	continue
		
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
		
		for qa_sample in annotations[video_id]:
			_id, ques, ans = qa_sample["sample_id"], qa_sample["question"], qa_sample["answer"]
			print(f"Processing: {_id}")
			
			support_frames = qa_sample["support"]
			if support_frames is None or len(support_frames) == 0:
				selected_frames = frame_idxs
			else:
				selected_frames = select_window_aroung_GT(support_frames, frame_idxs, args.gt_window_size)
			
			# Limit to max_frames frames.
			if len(selected_frames) > args.max_frame_limit:
				selected_frames = [
					selected_frames[i]
					for i in range(0, len(selected_frames), math.ceil(len(selected_frames) / args.max_frame_limit))
				]
				
			print(f"[INFO] #{len(selected_frames)} Frames Extracted Successfully")
			video = vr.get_batch(selected_frames).asnumpy()  # Expensive step
			image_sizes = [video[0].shape[:2]]
			video = process_images(video, image_processor, model.config)
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
	
	print(json.dumps(results, indent=4))
	save_at = f"results_{model_name}_gt_w{args.gt_window_size}.json"
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
						default="./checkpoints/LongVU_Qwen2_7B")  # LongVU_Qwen2_7B, LongVU_Llama3_2_3B
	parser.add_argument('--model_name', default="cambrian_qwen")  # cambrian_qwen, cambrian_llama3
	parser.add_argument('--version', default="qwen")  # qwen, llama3_2
	parser.add_argument('--local-rank', default=0)
	parser.add_argument('--qa_type', default="open", choices=["open", "closed"])
	parser.add_argument('--video_dir', default=video_dir)
	parser.add_argument('--video_ext', default=".mp4")
	parser.add_argument('--max_frame_limit', type=int, default=500)
	parser.add_argument('--gt_window_size', type=int, default=10)
	
	args = parser.parse_args()
	
	if "llama3" in args.version:
		args.model_name = "cambrian_llama3"
	
	predict(args, annotations)
