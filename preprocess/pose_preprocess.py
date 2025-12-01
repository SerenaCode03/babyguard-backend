# preprocess/pose_preprocess.py
import torch
from torchvision import transforms
from .common import decode_base64_image
from PIL import Image

# Same normalization as your training (ImageNet stats)
POSE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # -> [C,H,W], values in [0,1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_pose_image(img: Image.Image, transform=POSE_TRANSFORM, device: str = "cpu") -> torch.Tensor:
    """
    Takes a PIL Image, applies transforms, adds batch dim, and moves to device.
    """
    # Ensure image is RGB (handles PNGs with transparency)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    x = transform(img)             # [3, 224, 224]
    x = x.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    return x
