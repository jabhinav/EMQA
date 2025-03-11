import os

from PIL import Image, ImageSequence
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from typing import Dict, Sequence, Any

from utils.img_misc import expand2square

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


def get_padding_offset(cur_size, original_size):
	cur_w, cur_h = cur_size
	original_w, original_h = original_size
	
	original_aspect_ratio = original_w / original_h
	current_aspect_ratio = cur_w / cur_h
	
	if original_aspect_ratio > current_aspect_ratio:
		scale_factor = cur_w / original_w
		new_height = int(original_h * scale_factor)
		padding = (cur_h - new_height) // 2
		return 0, 0, padding, padding
	else:
		scale_factor = cur_h / original_h
		new_width = int(original_w * scale_factor)
		padding = (cur_w - new_width) // 2
		return padding, padding, 0, 0


def prepare_image_info(image_size, image_token_len, newline=False):
	num_tokens_per_side = int(image_token_len ** 0.5)
	if newline:
		# for the newline embedding
		attention_mask = torch.ones(
			num_tokens_per_side, num_tokens_per_side + 1, dtype=torch.bool
		)
	else:
		attention_mask = torch.ones(
			num_tokens_per_side, num_tokens_per_side, dtype=torch.bool
		)
	left_offset, right_offset, top_offset, bottom_offset = get_padding_offset(
		(num_tokens_per_side, num_tokens_per_side), image_size
	)
	if newline:
		if left_offset > 0:
			attention_mask[:, :left_offset] = 0
		if right_offset > 0:
			attention_mask[:, -right_offset - 1: -1] = 0
		if top_offset > 0:
			attention_mask[:top_offset, :] = 0
		if bottom_offset > 0:
			attention_mask[-bottom_offset:, :] = 0
	else:
		if left_offset > 0:
			attention_mask[:, :left_offset] = 0
		if right_offset > 0:
			attention_mask[:, -right_offset:] = 0
		if top_offset > 0:
			attention_mask[:top_offset, :] = 0
		if bottom_offset > 0:
			attention_mask[-bottom_offset:, :] = 0
	attention_mask = attention_mask.flatten()
	position_ids = attention_mask.cumsum(0) - 1
	return attention_mask, position_ids


def prepare_multimodal_data(
		input_ids,
		labels,
		attention_mask,
		image_sizes,
		image_token_len,
		image_aux_token_len_list,
		max_length,
):
	input_ids_im_replaced = []
	labels_im_replaced = []
	attention_mask_im_replaced = []
	position_ids_im_replaced = []
	im_aux_attention_masks_list = [[] for _ in range(len(image_aux_token_len_list))]
	base_image_token_len_per_side = int(image_token_len ** 0.5)
	image_aux_token_len_per_side_list = [
		int(image_aux_token_len_per_side ** 0.5)
		for image_aux_token_len_per_side in image_aux_token_len_list
	]
	# insert the padding tokens to the places of image so we can embed them together
	for batch_idx, cur_input_ids in enumerate(input_ids):
		num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
		assert num_images == 1, num_images
		image_size = image_sizes[batch_idx]
		
		image_token_indices = (
				[-1]
				+ torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
				+ [cur_input_ids.shape[0]]
		)
		
		cur_input_ids_im_replaced = []
		cur_labels_im_replaced = []
		cur_attention_mask_im_replaced = []
		cur_position_ids_im_replaced = []
		
		cur_labels = labels[batch_idx]
		cur_attention_mask = attention_mask[batch_idx]
		index = 0
		for i in range(len(image_token_indices) - 1):
			# still keep the first image token in input_ids for further use
			cur_input_ids_im_replaced.append(
				cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1] + 1]
			)
			cur_labels_im_replaced.append(
				cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]]
			)
			cur_attention_mask_im_replaced.append(
				cur_attention_mask[image_token_indices[i] + 1: image_token_indices[i + 1]]
			)
			cur_position_ids_im_replaced.append(
				torch.arange(
					index,
					index + image_token_indices[i + 1] - (image_token_indices[i] + 1),
					dtype=torch.long,
					device=cur_input_ids.device,
				)
			)
			index += image_token_indices[i + 1] - (image_token_indices[i] + 1)
			
			if i < len(image_token_indices) - 2:
				num_tokens_per_side = int(image_token_len ** 0.5)
				image_token_len_with_newline = image_token_len + num_tokens_per_side
				cur_input_ids_im_replaced.append(
					torch.full(
						(image_token_len_with_newline - 1,),
						0,
						device=cur_input_ids.device,
						dtype=cur_input_ids.dtype,
					)
				)
				cur_labels_im_replaced.append(
					torch.full(
						(image_token_len_with_newline,),
						IGNORE_INDEX,
						device=cur_labels.device,
						dtype=cur_labels.dtype,
					)
				)
				
				cur_im_attention_mask, cur_im_position_ids = prepare_image_info(
					image_size, image_token_len, newline=True
				)
				
				for aux_i, image_aux_token_len_per_side in enumerate(
						image_aux_token_len_per_side_list
				):
					assert image_aux_token_len_per_side >= base_image_token_len_per_side
					num_base_crops_per_aux_side = (
							image_aux_token_len_per_side // base_image_token_len_per_side
					)
					
					cur_im_aux_attention_mask, _ = prepare_image_info(image_size, image_aux_token_len_per_side ** 2)
					cur_im_aux_attention_mask = cur_im_aux_attention_mask.view(
						base_image_token_len_per_side,
						num_base_crops_per_aux_side,
						base_image_token_len_per_side,
						num_base_crops_per_aux_side,
					)
					cur_im_aux_attention_mask = (
						cur_im_aux_attention_mask.permute(0, 2, 1, 3)
						.contiguous()
						.flatten(0, 1)
						.flatten(1, 2)
					)
					cur_im_aux_attention_mask[
						cur_im_aux_attention_mask.sum(dim=1) == 0
						] = True
					im_aux_attention_masks_list[aux_i].append(cur_im_aux_attention_mask)
				cur_im_position_ids += index
				
				if cur_attention_mask[image_token_indices[i + 1]]:
					cur_attention_mask_im_replaced.append(cur_im_attention_mask)
					cur_position_ids_im_replaced.append(cur_im_position_ids.to(torch.long))
					index = cur_im_position_ids.max() + 1
				else:
					num_tokens_per_side = int(image_token_len ** 0.5)
					image_token_len_with_newline = image_token_len + num_tokens_per_side
					cur_attention_mask_im_replaced.append(
						torch.full(
							(image_token_len_with_newline,),
							0,
							device=cur_attention_mask.device,
							dtype=cur_attention_mask.dtype,
						)
					)
					cur_position_ids_im_replaced.append(
						torch.full(
							(image_token_len_with_newline,),
							0,
							device=cur_input_ids.device,
							dtype=torch.long,
						)
					)
		
		input_ids_im_replaced.append(torch.cat(cur_input_ids_im_replaced))
		labels_im_replaced.append(torch.cat(cur_labels_im_replaced))
		attention_mask_im_replaced.append(torch.cat(cur_attention_mask_im_replaced))
		position_ids_im_replaced.append(torch.cat(cur_position_ids_im_replaced))
	
	# Truncate sequences to max length as image embeddings can make the sequence longer
	new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
	new_labels = [x[0:max_length] for x in labels_im_replaced]
	new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
	new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
	
	# Stack the tensors
	new_input_ids = torch.stack(new_input_ids)
	new_labels = torch.stack(new_labels)
	new_attention_mask = torch.stack(new_attention_mask)
	new_position_ids = torch.stack(new_position_ids)
	im_aux_attention_masks_list = [
		torch.stack(im_aux_attention_masks)
		for im_aux_attention_masks in im_aux_attention_masks_list
	]
	
	return (
		new_input_ids,
		new_labels,
		new_attention_mask,
		new_position_ids,
		im_aux_attention_masks_list,
	)


class VideoDataset(Dataset):
	def __init__(self,
				 image_processors,
				 tokenizer: Any,
				 max_prompt_length: int = 512,
				 max_length: int = 1024,
				 mode='train',
				 data_dir=None,
				 debug=False
				 ):
		self.image_processors = image_processors
		
		self.tokenizer = tokenizer
		self.max_prompt_length = max_prompt_length
		self.max_length = max_length
		
		self.mode = mode
		if debug is True:
			self.qa_data, self.paths = self.get_sample_data()
		else:
			self.qa_data, self.paths = self.read_data()
		
		self.video_fps: float = 0.25
		self.uniform_sample: bool = False
		self.sampler_num_samples: int = 1000
		
	def __len__(self):
		return len(self.paths)
		
	def get_sample_data(self):
		video_paths = [
			'./baselines/LongVU/examples/video1.mp4',
			'./baselines/LongVU/examples/video2.mp4',
			'./baselines/LongVU/examples/video3.mp4',
		]
		
		qa_pairs = [
			("What is the color of the car?", "The car is red."),
			("What is the color of the car?", "The car is blue."),
			("What is the color of the car?", "The car is green."),
		]
		
		return qa_pairs, video_paths
	
	def read_data(self):
		raise NotImplementedError("Not implemented")
	
	def read_video(self, video_path):
		if video_path.endswith(".npy"):
			"""If the provided file is a numpy file"""
			video = np.load(video_path)
			image_size = video[0].shape[:2]
		
		elif video_path.endswith(".gif"):
			"""If the provided file is a gif"""
			image = Image.open(video_path)
			video = []
			for frame in ImageSequence.Iterator(image):
				frame_copy = frame.copy()
				video.append(frame_copy.convert("RGB"))
			image_size = video[0].size
		
		elif os.path.isdir(video_path):
			"""If the provided file is a directory of images"""
			files = [f for f in sorted(os.listdir(video_path))]
			video = []
			for file in files:
				video.append(
					Image.open(os.path.join(video_path, file)).convert(
						"RGB"
					)
				)
			image_size = video[0].size
		
		else:
			"""If the provided file is a video file like .mp4"""
			vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)  # vr is a VideoReader object
			
			# # Get frames. Sampling at 2xfps.
			fps = round(vr.get_avg_fps() / self.video_fps)
			frame_idxs = [i for i in range(0, len(vr), fps)]
			
			# # Limit to 1000 frames.
			if len(frame_idxs) > 1000:
				frame_idxs = [frame_idxs[i] for i in range(0, len(frame_idxs), len(frame_idxs) // 1000)]
			video = vr.get_batch(frame_idxs).asnumpy()
			image_size = video[0].shape[:2]
		
		if self.uniform_sample:
			if len(video) > self.sampler_num_samples:
				interval = len(video) / float(self.sampler_num_samples)
				indices = [int(interval * i) for i in range(self.sampler_num_samples)]
				video = [video[idx] for idx in indices]
				
		return image_size, video
	
	def process_text(self, q_str, a_str):
		"""Process the text data specific to models and datasets"""
		return q_str, a_str
	
	def process_qa(self, idx: int):
		
		q_str, a_str = self.qa_data[idx]
		q_str, a_str = self.process_text(q_str, a_str)
		
		src_input_ids, trg_label_ids = [], []
		question_token_ids = self.tokenizer.encode(q_str, verbose=False)
		question_token_ids = question_token_ids[-self.max_prompt_length:]  # Truncate the prompt from left
		src_input_ids.extend(question_token_ids)
		
		if self.mode == 'test':
			
			# Pad from left: Allows batched generation
			if len(src_input_ids) < self.max_prompt_length:
				new_input_ids = [self.tokenizer.pad_token_id] * self.max_prompt_length
				new_input_ids[-len(src_input_ids):] = src_input_ids
				src_input_ids = new_input_ids
			
			src_input_ids = torch.LongTensor(src_input_ids)
			mask = src_input_ids.ne(self.tokenizer.pad_token_id)
			return {
				"input_ids": src_input_ids,
				"attention_mask": mask
			}
		
		answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
		answer_token_ids.append(self.tokenizer.eos_token_id)
		src_input_ids.extend(answer_token_ids)
		
		trg_label_ids.extend([IGNORE_INDEX] * len(question_token_ids))
		trg_label_ids.extend(answer_token_ids)
		
		# Cut off the excess
		if len(src_input_ids) >= self.max_length:
			# Truncate i/p and label from right (this will auto. truncate only the response)
			src_input_ids = src_input_ids[:self.max_length]
			trg_label_ids = trg_label_ids[:self.max_length]
			
		else:
			# Pad input [prompt+response] with pad token
			new_input_ids = [self.tokenizer.pad_token_id] * self.max_length
			new_label_ids = [IGNORE_INDEX] * self.max_length
			
			if self.tokenizer.padding_side == "right":
				new_input_ids[:len(src_input_ids)] = src_input_ids
				new_label_ids[:len(trg_label_ids)] = trg_label_ids
			else:
				new_input_ids[-len(src_input_ids):] = src_input_ids
				new_label_ids[-len(trg_label_ids):] = trg_label_ids
			
			src_input_ids = new_input_ids
			trg_label_ids = new_label_ids
		
		# # Print the shapes
		# print(f"[Debug] src_input_ids: {len(src_input_ids)}")
		# print(f"[Debug] trg_label_ids: {len(trg_label_ids)}")
		
		# Convert to tensors
		src_input_ids = torch.LongTensor(src_input_ids)
		trg_label_ids = torch.LongTensor(trg_label_ids)
		
		# mask out padding
		src_mask = src_input_ids.ne(self.tokenizer.pad_token_id)
		
		return {
			"input_ids": src_input_ids,
			"attention_mask": src_mask,
			"labels": trg_label_ids,
		}
		
	def process_vid(self, video):
		
		processor = self.image_processors
		new_imgs = []
		if isinstance(processor, list):
			
			for image in video:
				if not isinstance(image, Image.Image):
					image = Image.fromarray(image)
				
				processed_img_list = []
				for _processor in processor:
					
					image_copy = image
					if hasattr(_processor, "image_mean"):
						
						try:
							target_resolution = _processor.crop_size["height"]
						except:
							target_resolution = _processor.size["height"]
						
						# First square it and then resize to trg resolution
						image_copy = expand2square(
							image_copy, tuple(int(x * 255) for x in _processor.image_mean)
						).resize((target_resolution, target_resolution))
					
					# Pre-process and convert to tensor
					image_copy = _processor.preprocess(image_copy, return_tensors="pt")["pixel_values"][0]
					processed_img_list.append(image_copy)
				
				new_imgs.append(processed_img_list)
			
			new_imgs = [list(_batch_image) for _batch_image in zip(*new_imgs)]
			new_imgs = [torch.stack(img_list) for img_list in new_imgs]
		
		else:
			new_imgs = []
			for image in video:
				image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
				image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
				new_imgs.append(image)
			
			if all(x.shape == new_imgs[0].shape for x in new_imgs):
				new_imgs = torch.stack(new_imgs, dim=0)
				
		return new_imgs
	
	def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
		
		video_path = self.paths[i]
		image_size, video = self.read_video(video_path)
		vid_data = self.process_vid(video)
		qa_data = self.process_qa(i)
		
		data = {
			"image_size": image_size,
			"vid_data": vid_data  # Is a list if self.image_processors is a list else simply a tensor
		}
		
		# Combine the video data with the qa data
		data.update(qa_data)
		
		return data
	
	def extra(
			self,
			batch,
			max_length,
			image_position=91,
			image_token_len=144,
			image_aux_token_len_list=[576, 576],
	):
		input_ids = batch['input_ids']
		labels = batch['labels']
		attention_mask = batch['attention_mask']
		image_sizes = batch['image_sizes']
		
		# insert dummy image
		for i in range(len(input_ids)):
			if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
				cur_input_ids_tmp = input_ids[i].clone()
				cur_input_ids_tmp[image_position + 1:] = input_ids[
														 i, image_position:-1
														 ]
				cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
				input_ids[i] = cur_input_ids_tmp
				
				cur_labels_tmp = labels[i].clone()
				cur_labels_tmp[image_position + 1:] = labels[i, image_position:-1]
				cur_labels_tmp[image_position] = IGNORE_INDEX
				labels[i] = cur_labels_tmp
				
				cur_attention_mask_tmp = attention_mask[i].clone()
				cur_attention_mask_tmp[image_position + 1:] = attention_mask[
															  i, image_position:-1
															  ]
				cur_attention_mask_tmp[image_position] = False
				attention_mask[i] = cur_attention_mask_tmp
		(
			new_input_ids,
			new_labels,
			new_attention_mask,
			new_position_ids,
			im_aux_attention_masks_list,
		) = prepare_multimodal_data(
			input_ids,
			labels,
			attention_mask,
			image_sizes,
			image_token_len,
			image_aux_token_len_list,
			max_length,
		)
		
		batch['input_ids'] = new_input_ids
		batch['labels'] = new_labels
		batch['attention_mask'] = new_attention_mask
		batch['position_ids'] = new_position_ids
		batch['im_aux_attention_masks_list'] = im_aux_attention_masks_list

	
	def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
		batch = dict()
	
		# Collate the other fields
		for key in instances[0].keys():
			if key == "vid_data":
				# -> List (size=batch_size) of list (size=num_img_encoders) of tensors
				image_list = [instance["vid_data"] for instance in instances]
				image_list = [
					list(batch_images) for batch_images in zip(*image_list)
				]
				if all(
						x is not None and x.shape == image_list[0][0].shape
						for x in image_list[0]
				):
					batch["images"] = [
						torch.stack(batch_images) for batch_images in image_list
					]
				else:
					batch["images"] = image_list
			
			elif key == "image_size":
				batch['image_sizes'] = [instance[key] for instance in instances]
			
			else:
				batch[key] = torch.stack([instance[key] for instance in instances])
		
		self.extra(batch, self.max_length)
		
		return batch  # ['images'] := Has List (size=num_img_encoders) of list (size=batch_size) of tensors
		
		
# @dataclass
# class DataCollator(object):
# 	def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

