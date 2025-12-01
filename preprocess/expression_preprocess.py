# preprocess/expression_preprocess.py
import torch
from torchvision import transforms
from .common import decode_base64_image
from PIL import Image

EXPR_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_expression_image(img: Image.Image, transform=EXPR_TRANSFORM, device: str = "cpu") -> torch.Tensor:
    """
    Takes a PIL Image (from app.py), applies normalization, 
    adds batch dimension, and moves to device.
    """
    # Ensure image is 3-channel RGB (prevents errors with PNGs/Alpha channels)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Apply transform (Resize -> ToTensor -> Normalize)
    x = transform(img)
    
    # Add batch dimension [1, 3, 224, 224] and move to GPU/CPU
    x = x.unsqueeze(0).to(device)
    
    return x