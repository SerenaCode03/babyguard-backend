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
    
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fw_handle = target_layer.register_forward_hook(forward_hook)
    bw_handle = target_layer.register_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)          
    score = output[0, target_class]
    score.backward()
    fw_handle.remove()
    bw_handle.remove()

    grad = gradients[0]   
    act = activations[0]  

    weights = grad.mean(dim=(2, 3), keepdim=True)      
    cam = (weights * act).sum(dim=1).squeeze(0)      

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= (cam.max() + 1e-6)

    return cam.cpu().numpy()  


def overlay_cam_on_image(
    base_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
   
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

def explain_pose_prediction(pred_label: str, cam_array: np.ndarray) -> str:
    cam_h, cam_w = cam_array.shape
    horizontal_focus = np.argmax(cam_array.mean(axis=0)) / cam_w
    vertical_focus = np.argmax(cam_array.mean(axis=1)) / cam_h

    horiz_str = (
        "left side of the body" if horizontal_focus < 0.33 else
        "central body region" if horizontal_focus < 0.66 else
        "right side of the body"
    )

    region_str = (
        "upper body or head" if vertical_focus < 0.33 else
        "midsection" if vertical_focus < 0.66 else
        "lower limbs"
    )

    norm_label = pred_label.lower().strip()

    if norm_label == "normal":
        explanation = (
            f"The model classified this pose as **normal**, focusing mainly on the {region_str} "
            f"and {horiz_str}. These areas show posture patterns consistent with "
            f"typical infant sleep positioning."
        )
    else:
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
    if detect_human_fn is not None:
        has_human = detect_human_fn(image_pil)
        if not has_human:
            raise ValueError("No human detected in frame.")

    model.eval()
    with torch.no_grad():
        orig_image = image_pil.convert("RGB")
        overlay_base = orig_image.resize((224, 224))

    input_tensor = val_transform(orig_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()  

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)

    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_conf = float(probs[0, pred_idx].item())
    pred_label = class_names[pred_idx]

    cam = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=pred_idx,
        target_layer=target_layer,
    )

    overlay_img = overlay_cam_on_image(overlay_base, cam, alpha=0.4)

    explanation = explain_pose_prediction(pred_label, cam)

    return {
        "label": pred_label,
        "confidence": pred_conf,
        "explanation": explanation,
        "overlay_image": overlay_img,
    }

def explain_expression_prediction(label: str, cam_array: np.ndarray) -> str:
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
            f"The model detected **distressed**, focusing on the {region}, "
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
   
    model.eval()

    orig_image = image_pil.convert("RGB")
    overlay_base = orig_image.resize((224, 224))

    input_tensor = val_transform(orig_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    with torch.no_grad():
      logits = model(input_tensor)
      probs = torch.softmax(logits, dim=1)

    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_conf = float(probs[0, pred_idx].item())
    pred_label = class_names[pred_idx]

    cam = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        target_class=pred_idx,
        target_layer=target_layer,
    )
    overlay_img = overlay_cam_on_image(overlay_base, cam, alpha=0.4)
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
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)  # Shape: [N, num_classes]
    avg_probs = torch.mean(probs, dim=0)
    final_pred_idx = int(torch.argmax(avg_probs).item())
    final_conf = float(avg_probs[final_pred_idx].item())
    final_label = class_names[final_pred_idx]
    segment_scores = probs[:, final_pred_idx] # Shape: [N]
    best_segment_idx = int(torch.argmax(segment_scores).item())
    target_segment_tensor = input_tensor[best_segment_idx].unsqueeze(0) # [1, 3, 224, 224]
    target_segment_image = extra_info[best_segment_idx]                 # PIL Image
    target_segment_tensor.requires_grad_()

    cam = generate_gradcam(
        model=model,
        input_tensor=target_segment_tensor,
        target_class=final_pred_idx,
        target_layer=target_layer,
    )

    overlay_base = target_segment_image.resize((224, 224)).convert("RGB")
    overlay_img = overlay_cam_on_image(overlay_base, cam, alpha=0.4)
    explanation = explain_cry_prediction(final_label, cam)

    return {
        "label": final_label,
        "confidence": final_conf,
        "explanation": explanation,
        "overlay_image": overlay_img,
    }

