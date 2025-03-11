import numpy as np
from PIL import Image
import os


def unpad_image(tensor, original_size):
    """
    Un-pads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The un-padded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding: current_width - padding]

    return unpadded_tensor


def expand2square(pil_img, background_color):
	# Credits: https://github.com/Vision-CAIR/LongVU
	width, height = pil_img.size
	if width == height:
		return pil_img
	elif width > height:
		result = Image.new(pil_img.mode, (width, width), background_color)
		result.paste(pil_img, (0, (width - height) // 2))  # Paste s.t. img is positioned at centre height-wise
		return result
	else:
		result = Image.new(pil_img.mode, (height, height), background_color)
		result.paste(pil_img, ((height - width) // 2, 0))  # Paste s.t. img is positioned at centre width-wise
		return result


def extract_frames(video_file_path, out_dir):
	"""
	Extract individual frames from a scan video using opencv and save them as separate images.
	Parameters
	----------
	video_file_path: str
		Path to the video file
	out_dir: str
		Path to the directory where the frames will be saved
	Returns
	-------
	
	"""
	import cv2
	os.makedirs(out_dir, exist_ok=True)
	print('Extracting frames from: ' + video_file_path)
	
	cap = cv2.VideoCapture(video_file_path)
	i = 0
	while (cap.isOpened()):
		ret, frame = cap.read()
		if ret == False:
			break
		if 'depth' not in video_file_path:
			cv2.imwrite(os.path.join(out_dir,  str(i).zfill(6) + '.jpg'), frame)
		else:
			max_depth_val = 4500  # maximum depth in mm ( default 10m )
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame = frame * float(max_depth_val / 255.0)
			frame = frame.astype(np.uint16)
			cv2.imwrite(os.path.join(out_dir, str(i).zfill(6) + '.png'), frame)
		i += 1
	
	cap.release()
	cv2.destroyAllWindows()
	print('Done')
