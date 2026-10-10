"""DINO image conditioning, constructed locally without downloading weights."""
import torch
from torch import nn
from torchvision import transforms
from transformers import Dinov2Config, Dinov2Model


class ImageConditioner(nn.Module):
    def __init__(self, config, image_size):
        super().__init__()
        self.model = Dinov2Model(Dinov2Config.from_dict(config))
        self.transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def forward(self, image):
        image = (image + 1) / 2
        return self.model(self.transform(image)).last_hidden_state
