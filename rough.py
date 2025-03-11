import cv2
import numpy as np
import torch
from decord import cpu, VideoReader
from mm_encoder.videoclip_xl_utils.text_encoder import text_encoder


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
	"""
	This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
	model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
	"""
	square_tensor = torch.pow(tensor, 2)
	sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
	normed_tensor = torch.pow(sum_tensor, 0.5)
	return normed_tensor


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


def get_video_embeddings(videoclip_xl, segment, fnum=8):
	def preprocess():
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
	
	video_inputs = preprocess()
	with torch.no_grad():
		video_embeds = videoclip_xl.vision_model.get_vid_features(video_inputs).float()
		video_embeds /= _get_vector_norm(video_embeds)
	return video_embeds


def get_text_embeddings(videoclip_xl, text_encoder, text):
	with torch.no_grad():
		text_inputs = text_encoder.tokenize(text, truncate=True)
		text_embeds = videoclip_xl.text_model.encode_text(text_inputs)
		text_embeds /= _get_vector_norm(text_embeds)
	return text_embeds


video_path = './data/sample1.mp4'
fnum = 4

vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
fps = round(vr.get_avg_fps())
sampling_rate = round(fps / 0.5)
frame_idxs = [i for i in range(0, len(vr), sampling_rate)]
segment = vr.get_batch(frame_idxs).asnumpy()

videoclip_xl = load_videoclip_model('./checkpoints/VideoCLIP-XL.bin')
video_embeds = get_video_embeddings(videoclip_xl, segment, fnum)

text_sample = 'Are there cars in the video?'
text_embeds = get_text_embeddings(videoclip_xl, text_encoder, text_sample)

print(video_embeds.shape, text_embeds.shape)
Tmp = 100.
# Compute similarity between video and text embeddings
similarity = torch.matmul(video_embeds, text_embeds.T) * Tmp
print(similarity)

