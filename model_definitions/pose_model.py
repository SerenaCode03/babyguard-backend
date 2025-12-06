# model_definitions/pose_model.py
import torch
import torch.nn as nn
from functools import lru_cache
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from preprocess.pose_preprocess import POSE_TRANSFORM

POSE_LABELS = ["Abnormal", "Normal"]

# Match your actual saved path
POSE_WEIGHTS_PATH = "weights/efficientnetb0_best.pth"
def build_finetune_efficientnet_b0(num_classes: int = 2) -> nn.Module:
    """
    Rebuilds the SAME architecture you used during training.
    For inference we don't care about requires_grad, only the layer shapes.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # Replace classifier head to match num_classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

@lru_cache(maxsize=1)
def get_pose_model(device: str = "cpu") -> nn.Module:
    model = build_finetune_efficientnet_b0(num_classes=len(POSE_LABELS))
    # Load your trained weights
    state = torch.load(POSE_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_pose_model(device: str = "cpu"):
    """
    Helper for app.py:
      - loads and caches the pose model
      - returns everything Grad-CAM + API need
    """
    model = get_pose_model(device)
    class_names = POSE_LABELS
    target_layer = model.features[-1]  # last conv block in EfficientNet-B0

    return model, POSE_TRANSFORM, class_names, target_layer