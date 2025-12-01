# gradcam_utils.py

import io
from typing import Tuple, Optional, Callable, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

# Generic Grad-CAM
def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    target_layer: torch.nn.Module,
) -> np.ndarray:
    """
    Generic Grad-CAM:
      - model: PyTorch model
      - input_tensor: [1, C, H, W] with requires_grad=True
      - target_class: int index of the class to explain
      - target_layer: layer to hook (e.g. model.features[-1], model.layer4)

    Returns:
      cam: 2D numpy array (H_cam, W_cam) normalized to [0, 1]
    """
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    # Register hooks
    fw_handle = target_layer.register_forward_hook(forward_hook)
    bw_handle = target_layer.register_backward_hook(backward_hook)

    # Forward + backward
    model.zero_grad()
    output = model(input_tensor)          # [1, num_classes]
    score = output[0, target_class]
    score.backward()

    # Remove hooks
    fw_handle.remove()
    bw_handle.remove()

    grad = gradients[0]   # [B, C, H, W]
    act = activations[0]  # [B, C, H, W]

    # Global-average-pool the gradients => weights
    weights = grad.mean(dim=(2, 3), keepdim=True)      # [B, C, 1, 1]
    cam = (weights * act).sum(dim=1).squeeze(0)        # [H, W]

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-6)

    return cam.cpu().numpy()  # (H_cam, W_cam) in [0,1]


def overlay_cam_on_image(
    base_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """
    Resize CAM to image size, make heatmap, blend with original.
    base_image: PIL.Image (RGB)
    cam: 2D numpy array in [0,1]
    alpha: blending factor for heatmap

    Returns a new PIL.Image with overlay.
    """
    base_image = base_image.convert("RGB")
    img_np = np.array(base_image)

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(img_np, 1.0 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)

# 2) Pose-specific explanation (text) and wrapper
def explain_pose_prediction(pred_label: str, cam_array: np.ndarray) -> str:
    """
    Turn pose prediction + Grad-CAM focus into a human-readable explanation.
    cam_array: 2D numpy array in [0,1]
    """
    cam_h, cam_w = cam_array.shape

    # Position of maximum attention (roughly)
    horizontal_focus = np.argmax(cam_array.mean(axis=0)) / cam_w
    vertical_focus = np.argmax(cam_array.mean(axis=1)) / cam_h

    # Left / center / right
    horiz_str = (
        "left side of the body" if horizontal_focus < 0.33 else
        "central body region" if horizontal_focus < 0.66 else
        "right side of the body"
    )

    # Top / middle / bottom
    region_str = (
        "upper body or head" if vertical_focus < 0.33 else
        "midsection" if vertical_focus < 0.66 else
        "lower limbs"
    )

    norm_label = pred_label.lower().strip()

    # Use exact match (==) because "abnormal" contains the word "normal"
    if norm_label == "normal":
        explanation = (
            f"The model classified this pose as **normal**, focusing mainly on the {region_str} "
            f"and {horiz_str}. These areas show posture patterns consistent with "
            f"typical infant sleep positioning."
        )
    else:
        # Matches "abnormal"
        explanation = (
            f"The model classified this pose as **abnormal**, with attention on the {region_str} "
            f"and {horiz_str}. This may indicate irregular posture such as twisting, stiff limbs, "
            f"or unsafe orientation."
        )

    return explanation

def run_pose_gradcam(
    model: torch.nn.Module,
    device: torch.device,
    image_pil: Image.Image,
    val_transform,
    class_names: list,
    target_layer: torch.nn.Module,
    detect_human_fn: Optional[Callable[[Image.Image], bool]] = None,
):
    """
    High-level helper used by FastAPI endpoint (or any backend):

    - image_pil    : raw PIL image from client (pose frame)
    - val_transform: same transform used during training (Resize + ToTensor + Normalize)
    - class_names  : e.g. ['Abnormal', 'Normal']
    - target_layer : layer to hook for Grad-CAM (e.g. model.features[-1])
    - detect_human_fn (optional): function(PIL.Image)->bool, to reject images with no infant

    Returns:
      {
        "label": str,
        "confidence": float,
        "explanation": str,
        "overlay_image": PIL.Image
      }

    Raises:
      ValueError on "no human" or other sanity checks.
    """
    # 1) Optional human/infant detection gate
    if detect_human_fn is not None:
        has_human = detect_human_fn(image_pil)
        if not has_human:
            raise ValueError("No human detected in frame.")

    # 2) Preprocess
    model.eval()
    with torch.no_grad():
        orig_image = image_pil.convert("RGB")
        # keep a 224x224 version for overlay (to be consistent with model)
        overlay_base = orig_image.resize((224, 224))

    input_tensor = val_transform(orig_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()  # needed for Grad-CAM

    # 3) Forward pass for prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_conf = float(probs[0, pred_idx].item())
    pred_label = class_names[pred_idx]

    # 4) Grad-CAM (this will re-run forward + backward with gradients)
    cam = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=pred_idx,
        target_layer=target_layer,
    )

    # 5) Overlay heatmap on image
    overlay_img = overlay_cam_on_image(overlay_base, cam, alpha=0.4)

    # 6) Text explanation
    explanation = explain_pose_prediction(pred_label, cam)

    return {
        "label": pred_label,
        "confidence": pred_conf,
        "explanation": explanation,
        "overlay_image": overlay_img,
    }

def explain_expression_prediction(label: str, cam_array: np.ndarray) -> str:
    """
    Turn facial expression + Grad-CAM result into a human-readable explanation.
    cam_array: 2D numpy array in [0,1]
    """
    cam_h, cam_w = cam_array.shape

    # Where Grad-CAM is focusing vertically (top → bottom)
    vertical_focus = np.argmax(cam_array.mean(axis=1)) / cam_h

    region = (
        "forehead and eyes" if vertical_focus < 0.33 else
        "mid-face (nose and cheeks)" if vertical_focus < 0.66 else
        "lower face and mouth"
    )

    label_lower = label.lower()
    if "distressed" in label_lower:
        return (
            f"The model detected **distress**, focusing on the {region}, "
            f"where discomfort cues such as tension, frowning, or crying "
            f"typically appear in infants."
        )
    else:
        return (
            f"The model detected a **normal** expression, focusing on the {region}, "
            f"which is usually more relaxed and stable in calm infants."
        )

def run_expression_gradcam(
    model: torch.nn.Module,
    device: torch.device,
    image_pil: Image.Image,
    val_transform,
    class_names: list,                # e.g. ['Distressed', 'Normal']
    target_layer: torch.nn.Module,    # e.g. model.features[-1]
):
    """
    High-level helper for facial expression Grad-CAM.

    Returns:
      {
        "label": str,
        "confidence": float,
        "explanation": str,
        "overlay_image": PIL.Image
      }
    """
    model.eval()

    # 1) Prepare image
    orig_image = image_pil.convert("RGB")
    overlay_base = orig_image.resize((224, 224))

    # 2) Preprocess using the SAME val_transform as training
    input_tensor = val_transform(orig_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    # 3) Forward pass (prediction)
    with torch.no_grad():
      logits = model(input_tensor)
      probs = torch.softmax(logits, dim=1)

    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_conf = float(probs[0, pred_idx].item())
    pred_label = class_names[pred_idx]

    # 4) Grad-CAM
    cam = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=pred_idx,
        target_layer=target_layer,
    )

    # 5) Overlay heatmap
    overlay_img = overlay_cam_on_image(overlay_base, cam, alpha=0.4)
    # 6) Text explanation
    explanation = explain_expression_prediction(pred_label, cam)

    return {
        "label": pred_label,
        "confidence": pred_conf,
        "explanation": explanation,
        "overlay_image": overlay_img,
    }

CRY_LABELS = ["asphyxia", "hungry", "normal", "pain"]

def explain_cry_prediction(pred_label: str, cam_array: np.ndarray) -> str:
    """Turn cry label + Grad-CAM result into a human-readable explanation."""
    cry_insights = {
        "pain": "Pain cries often show sharp bursts in mid-to-high frequencies, indicating distress.",
        "hungry": "Hungry cries typically have rhythmic, rising low-frequency tones.",
        "asphyxia": "Asphyxia cries may include irregular, high-pitched tones and lower energy.",
        "normal": "Normal cries have balanced acoustic energy and consistent patterns.",
    }
    
    cam_h, cam_w = cam_array.shape
    horizontal_focus = np.argmax(cam_array.mean(axis=0)) / cam_w
    vertical_focus = np.argmax(cam_array.mean(axis=1)) / cam_h

    time_str = (
        "early in the segment" if horizontal_focus < 0.33 else
        "in the middle of the segment" if horizontal_focus < 0.66 else
        "towards the end of the segment"
    )
    freq_str = (
        "low-frequency" if vertical_focus > 0.66 else
        "mid-frequency" if vertical_focus > 0.33 else
        "high-frequency"
    )

    base = (
        f"The model predicted **{pred_label}**. In the most significant audio segment, "
        f"it focused on {freq_str} features {time_str}."
    )
    extra = cry_insights.get(pred_label.lower(), "")
    return f"{base} {extra}".strip()

def run_cry_gradcam_from_image(
    model: torch.nn.Module,
    device: torch.device,
    input_tensor: torch.Tensor,
    class_names: Sequence[str],
    target_layer: torch.nn.Module,
    extra_info: List[Image.Image],
):
    """
    Handles a BATCH of audio segments.
    
    Arguments:
      - input_tensor: [N, 3, 224, 224] batch of N segments.
      - extra_info: List of N PIL Images (the spectrograms for each segment).
    """
    model.eval()

    # 1. Get predictions for ALL segments
    # input_tensor is already [N, 3, 224, 224] from preprocess
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)  # Shape: [N, num_classes]

    # 2. Average probabilities across all segments to get the "Global" prediction
    avg_probs = torch.mean(probs, dim=0)
    final_pred_idx = int(torch.argmax(avg_probs).item())
    final_conf = float(avg_probs[final_pred_idx].item())
    final_label = class_names[final_pred_idx]

    # 3. Select the "Best" Segment for Grad-CAM
    # We want to show the heatmap for the segment that looked *most* like the predicted class.
    # We look at the column for the predicted class (final_pred_idx) across all N segments.
    segment_scores = probs[:, final_pred_idx] # Shape: [N]
    best_segment_idx = int(torch.argmax(segment_scores).item())

    # 4. Extract the specific data for that best segment
    target_segment_tensor = input_tensor[best_segment_idx].unsqueeze(0) # [1, 3, 224, 224]
    target_segment_image = extra_info[best_segment_idx]                 # PIL Image

    # IMPORTANT: Enable gradients for Grad-CAM on this specific tensor
    target_segment_tensor.requires_grad_()

    # 5. Run Grad-CAM on the single best segment
    cam = generate_gradcam(
        model=model,
        input_tensor=target_segment_tensor,
        target_class=final_pred_idx,
        target_layer=target_layer,
    )

    # 6. Overlay
    overlay_base = target_segment_image.resize((224, 224)).convert("RGB")
    overlay_img = overlay_cam_on_image(overlay_base, cam, alpha=0.4)

    # 7. Explanation
    explanation = explain_cry_prediction(final_label, cam)

    return {
        "label": final_label,
        "confidence": final_conf,
        "explanation": explanation,
        "overlay_image": overlay_img,
    }

