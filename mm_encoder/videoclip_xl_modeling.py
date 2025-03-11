import torch.nn as nn

from .videoclip_xl_utils import text_encoder
from .videoclip_xl_utils import get_vision_encoder


class VideoCLIP_XL(nn.Module):
    def __init__(self):
        super(VideoCLIP_XL, self).__init__()
        self.text_model = text_encoder.load().float()
        self.vision_model = get_vision_encoder().float()