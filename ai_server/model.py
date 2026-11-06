"""Second step: cut out the animal and let our model name it."""
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
_prep = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(image: Image.Image) -> torch.Tensor:
    return _prep(image.convert("RGB"))


def crop_to_box(image: Image.Image, box: list[float], margin: float = 0.10) -> Image.Image:
    x, y, w, h = box  # numbers between 0 and 1
    pad = margin * max(w, h)
    width, height = image.size
    left = int(max(0.0, x - pad) * width)
    top = int(max(0.0, y - pad) * height)
    right = max(int(min(1.0, x + w + pad) * width), left + 1)
    bottom = max(int(min(1.0, y + h + pad) * height), top + 1)
    return image.crop((left, top, right, bottom))


def load_model(weights_path: str, labels_path: str) -> tuple[nn.Module, list[str]]:
    with open(labels_path, encoding="utf-8") as f:
        labels = json.load(f)
    net = models.mobilenet_v3_small()
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(labels))
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return net.eval(), labels


@torch.no_grad()
def predict_scores(model: nn.Module, labels: list[str], tensor: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(model(tensor.unsqueeze(0)), dim=1)[0]
    return {label: p.item() for label, p in zip(labels, probs)}
