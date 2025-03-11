import torch
from typing import List


def filter_feature_sim(
	config,
	feature_list,
	image_list,
	split_sizes,
	input_ids,
	image_sizes: List[int],
	window_size=16,
	threshold=0.83,
):
	"""
	Filter frames based on their feature similarity. Correspondingly -
	 > retain the selected frames for each encoder
	 > update the split sizes post frame pruning
	 > Update the feature list provided as input to the function
	
	:param feature_list:
	:param image_list:
	:param split_sizes: List of number of frames representing each sample/video
	:param window_size: Consecutive frames within which redundancy is measured
	:param threshold: Similarity threshold below which the frames will be retained
	:return:
	"""
	# Split the features for every sample based on number of frames representing the sample (video)
	features_batch = torch.split(feature_list, split_sizes, dim=0)
	enc_0_images = torch.split(image_list[0], split_sizes, dim=0)
	enc_1_images = torch.split(image_list[1], split_sizes, dim=0)
	
	new_split_sizes = []
	selected_frames_all_0 = []
	selected_frames_all_1 = []
	selected_frames_feature_all = []
	selected_frame_indices_all = []
	for i_batch, frame_features in enumerate(features_batch):
		
		# Compute max_num_frames
		text_len = torch.where(input_ids[i_batch] == config.padding_token_id)[-1][0]
		token_per_frame = config.image_token_len
		max_num_frames = max(
			1,
			(
					config.model_max_length
					- text_len
					- getattr(config, "inference_max_length", 16)
			)
			// token_per_frame,
		)
		
		# No need to filter if there is enough room for this frame
		if len(frame_features) < max_num_frames:
			
			# [Update]
			selected_frames_all_0.append(enc_0_images[i_batch])
			selected_frames_all_1.append(enc_1_images[i_batch])
			selected_frames_feature_all.append(frame_features)
			new_split_sizes.append(len(frame_features))
			selected_frame_indices_all.append(torch.arange(len(frame_features)))
			continue
		
		num_segments = len(frame_features) // window_size
		if num_segments == 0:
			
			# Measure how similar a frame is to rest of the frames on average
			query_feature = frame_features.flatten(1, 2)  # The number of visual tokens along with their embedding dim is flattened
			query_feature /= torch.norm((query_feature), dim=1, keepdim=True)
			similarities = torch.mean(query_feature @ query_feature.T, dim=1)
			similarities[len(frame_features) // 2] = 0
			indices = torch.where(similarities < threshold)[0]
			
			# [Update]
			selected_frame_indices_all.append(indices)
			selected_frames_all_0.append(enc_0_images[i_batch][indices])
			selected_frames_all_1.append(enc_1_images[i_batch][indices])
			selected_frames_feature_all.append(frame_features[indices])
			new_split_sizes.append(len(indices))
			continue
		
		# Divide the number of frames into segments
		segments_frames_0 = []
		segments_frames_1 = []
		segments_features = []
		for start_idx in range(0, len(frame_features), window_size):
			end_idx = min(start_idx + window_size, len(frame_features))
			segments_frames_0.append(enc_0_images[i_batch][start_idx:end_idx])
			segments_frames_1.append(enc_1_images[i_batch][start_idx:end_idx])
			segments_features.append(frame_features[start_idx:end_idx])
		
		selected_frames_0 = []
		selected_frames_1 = []
		selected_features = []
		selected_frame_indices = []
		for i, segment in enumerate(segments_features):
			# Measure how similar a frame is to rest of the frames on average
			query_feature = segment.flatten(1, 2)
			query_feature /= torch.norm((query_feature), dim=1, keepdim=True)
			similarities = torch.mean(query_feature @ query_feature.T, dim=1)
			similarities[len(segment) // 2] = 0
			indices = torch.where(similarities < threshold)[0]
			
			selected_frames_0.append(segments_frames_0[i][indices])
			selected_frames_1.append(segments_frames_1[i][indices])
			selected_features.append(segment[indices])
			selected_frame_indices.extend(indices + i * window_size)
		selected_frames_0 = torch.cat(selected_frames_0, dim=0)
		selected_frames_1 = torch.cat(selected_frames_1, dim=0)
		selected_features = torch.cat(selected_features, dim=0)
		selected_frame_indices = torch.tensor(selected_frame_indices)
		
		# TODO: Optional - Further prune by selecting a frame at every ith interval
		max_num_frames = 400  # in case of OOM (Src: Original Implementation)
		if len(selected_frame_indices) > max_num_frames:
			interval = len(selected_frame_indices) / float(max_num_frames)
			indices = [int(interval * i) for i in range(max_num_frames)]
			
			# [Update]
			new_split_sizes.append(len(indices))
			selected_frames_all_0.append(selected_frames_0[indices])
			selected_frames_all_1.append(selected_frames_1[indices])
			selected_frames_feature_all.append(selected_features[indices])
			selected_frame_indices = selected_frame_indices[indices]
		
		else:
			# [Update]
			new_split_sizes.append(len(selected_frames_0))
			selected_frames_all_0.append(selected_frames_0)
			selected_frames_all_1.append(selected_frames_1)
			selected_frames_feature_all.append(selected_features)
		selected_frame_indices_all.append(selected_frame_indices)
	
	selected_frames_all_0 = torch.cat(selected_frames_all_0, dim=0)
	selected_frames_all_1 = torch.cat(selected_frames_all_1, dim=0)
	selected_frames_feature_all = torch.cat(selected_frames_feature_all, dim=0)
	return (
		selected_frames_feature_all,
		new_split_sizes,
		[selected_frames_all_0, selected_frames_all_1],
		selected_frame_indices_all,
	)


def split_features_by_sample(features, split_sizes):
	"""
	Splits the features for each sample based on the number of frames representing the sample (video).
	
	Parameters:
	features (torch.Tensor): Features to split.
	split_sizes (List[int]): List of number of frames representing each sample/video.
	
	Returns:
	List[torch.Tensor]: List of features for each sample.
	"""
	
	# Split the features for every sample based on the number of frames representing the sample (video)
	features_batch = torch.split(features, split_sizes, dim=0)
	
	return features_batch


def get_spatio_temporal_features_using_pooling(features, use_padding=True, max_length=100):
	"""
	Credits: Video ChatGPT
	URL: https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/video_chatgpt/model/video_chatgpt.py
	Computes spatio-temporal features from given features.
	
	Parameters:
	features (torch.Tensor): Input features to process.
	use_padding (bool): Whether to pad the features to max_length.
	max_length (int): Maximum length to pad the temporal features.
	
	Returns:
	torch.Tensor: Spatio-temporal features.
	"""
	
	# Extract the dimensions of the features -> num_frames, num_patches, feature_dim
	t, s, c = features.shape
	
	assert t <= max_length, "Number of frames should be less than or equal to max_length set for temporal tokens in pooling."
	
	# Compute temporal tokens as the mean along the time axis
	temporal_tokens = torch.mean(features, dim=1)
	
	# Padding size calculation
	padding_size = max_length - t if use_padding else 0
	
	# Pad temporal tokens if necessary
	if padding_size > 0:
		padding = torch.zeros(padding_size, c, device=features.device)
		temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)
	
	# Compute spatial tokens as the mean along the spatial axis
	spatial_tokens = torch.mean(features, dim=0)
	
	# Concatenate temporal and spatial tokens and cast to half precision
	concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0)
	
	return concat_tokens