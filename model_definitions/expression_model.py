# model_definitions/expression_model.py
import torch
import torch.nn as nn
from functools import lru_cache
from torchvision.models import mobilenet_v3_small
from preprocess.expression_preprocess import EXPR_TRANSFORM 

# Match your class order used in Flutter / risk scoring
EXPR_LABELS = ["Distressed", "Normal"]

# Adjust path to where you’ll put your weights in the backend
EXPR_WEIGHTS_PATH = "weights/mobilenetv3_best_v3.pth"


def build_finetune_mobilenetv3_small(
    num_classes: int = 2,
    dropout: float = 0.5,
) -> nn.Module:
    """
    Rebuilds the SAME architecture you used in training:

    - mobilenet_v3_small backbone
    - classifier = Linear(576→1024) → Hardswish → Dropout → Linear(1024→num_classes)
    """
    # We don't need pretrained weights here, because we will load your fine-tuned
    # state_dict right after. So weights=None is fine.
    model = mobilenet_v3_small(weights=None)

    # Replace classifier head EXACTLY as in your training code
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=dropout),
        nn.Linear(1024, num_classes),
    )

    return model


@lru_cache(maxsize=1)
def get_expression_model(device: str = "cpu") -> nn.Module:
    """
    Lazily load and cache the expression model.
    """
    model = build_finetune_mobilenetv3_small(num_classes=len(EXPR_LABELS))

    # If during training you did:
    #   torch.save(model.state_dict(), "mobilenetv3_best_v3.pth")
    # then this is correct:
    state = torch.load(EXPR_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

def load_expression_model(device: str = "cpu"):
    """
    Helper for app.py:
      - loads and caches the expression model
      - returns everything Grad-CAM + API need
    """
    model = get_expression_model(device)
    class_names = EXPR_LABELS
    target_layer = model.features[-1]  # last conv block of MobileNetV3-Small

    return model, EXPR_TRANSFORM, class_names, target_layer