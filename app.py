import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch

from model_definitions.pose_model import load_pose_model
from model_definitions.expression_model import load_expression_model
from model_definitions.cry_model import load_cry_model

from preprocess.pose_preprocess import preprocess_pose_image
from preprocess.expression_preprocess import preprocess_expression_image
from preprocess.cry_preprocess import preprocess_cry_audio

from gradcam_utils import (
    run_pose_gradcam,
    run_expression_gradcam,
    run_cry_gradcam_from_image
)

app = FastAPI(title="BabyGuard Backend API")

# --- Load all models at startup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unpack Pose components
pose_model, pose_val_transform, pose_class_names, pose_target_layer = load_pose_model(device)
expression_model, expr_val_transform, expression_class_names, expr_target_layer = load_expression_model(device)
cry_model, cry_transform, cry_labels, cry_target_layer = load_cry_model(device)


def pil_to_base64(img: Image.Image) -> str:
    """Helper to convert PIL Image to Base64 string for JSON response"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


@app.post("/predict/pose")
async def predict_pose(file: UploadFile = File(...)):
    try:
        # 1. Read bytes and convert to PIL
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Preprocess (Pass the transform loaded from load_pose_model)
        input_tensor = preprocess_pose_image(image, pose_val_transform, device)

        # 3. Run Grad-CAM logic
        # Note: Your run_pose_gradcam likely performs the inference internally. 
        # If it doesn't, you might need: outputs = pose_model(input_tensor)
        result = run_pose_gradcam(
            model=pose_model,
            device=device,
            image_pil=image,
            val_transform=pose_val_transform,
            class_names=pose_class_names,
            target_layer=pose_target_layer,
        )

        # 4. Convert overlay result to base64
        overlay_b64 = pil_to_base64(result["overlay_image"])

        return {
            "label": result["label"],
            "confidence": float(result["confidence"]), # Ensure float for JSON serialization
            "explanation": result["explanation"],
            "overlay_image": overlay_b64,
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # Print error to server logs for easier debugging
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/expression")
async def predict_expression(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess
        input_tensor = preprocess_expression_image(image, expr_val_transform, device)

        result = run_expression_gradcam(
            model=expression_model,
            device=device,
            image_pil=image,
            val_transform=expr_val_transform,
            class_names=expression_class_names,
            target_layer=expr_target_layer,
        )

        overlay_b64 = pil_to_base64(result["overlay_image"])

        return {
            "label": result["label"],
            "confidence": float(result["confidence"]),
            "explanation": result["explanation"],
            "overlay_image": overlay_b64,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict/cry")
async def predict_cry(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # Preprocess
        input_tensor, extra_info = preprocess_cry_audio(audio_bytes, cry_transform, device)

        result = run_cry_gradcam_from_image(
            model=cry_model,
            device=device,
            input_tensor=input_tensor,
            class_names=cry_labels,
            target_layer=cry_target_layer,
            extra_info=extra_info,
        )

        overlay_b64 = pil_to_base64(result["overlay_image"])

        return {
            "label": result["label"],
            "confidence": float(result["confidence"]),
            "explanation": result["explanation"],
            "overlay_image": overlay_b64,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
def root():
    return {"message": "BabyGuard Backend API Running"}