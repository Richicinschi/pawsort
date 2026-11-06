"""Model loading and inference. Same preprocessing as training/train.py."""
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 160
_prep = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    return _prep(image.convert("RGB"))


def build_head() -> nn.Sequential:
    return nn.Sequential(nn.Linear(576, 128), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(128, 1))


def load_model(head_path: str) -> nn.Module:
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Identity()
    head = build_head()
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    return nn.Sequential(backbone, head).eval()


@torch.no_grad()
def predict_dog_probability(model: nn.Module, tensor: torch.Tensor) -> float:
    return torch.sigmoid(model(tensor.unsqueeze(0))).item()
