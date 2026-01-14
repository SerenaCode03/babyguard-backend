#app.py
import io
import base64
import os
import smtplib
from email.mime.text import MIMEText
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from dotenv import load_dotenv  

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

load_dotenv()

app = FastAPI(title="BabyGuard Backend API")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pose_model, pose_val_transform, pose_class_names, pose_target_layer = load_pose_model(device)
expression_model, expr_val_transform, expression_class_names, expr_target_layer = load_expression_model(device)
cry_model, cry_transform, cry_labels, cry_target_layer = load_cry_model(device)


def pil_to_base64(img: Image.Image) -> str:
    """Helper to convert PIL Image to Base64 string for JSON response"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

class RiskEmailRequest(BaseModel):
  to_email: str
  risk_level: str
  sleep_label: str
  expr_label: str
  cry_label: str
  summary: str   


def send_risk_email(req: RiskEmailRequest):
  """Send a simple text-only risk alert email to parent."""
  subject = f"BabyGuard Alert: {req.risk_level.upper()} risk detected"

  body = (
    f"Dear Parent,\n\n"
    f"BabyGuard has detected a {req.risk_level.upper()} risk event.\n\n"
    f"Detected anomalies:\n"
    f"- Sleeping posture: {req.sleep_label}\n"
    f"- Facial expression: {req.expr_label}\n"
    f"- Cry pattern: {req.cry_label}\n\n"
    f"Summary:\n{req.summary}\n\n"
    f"This email was sent automatically by the BabyGuard monitoring system."
  )

  msg = MIMEText(body)
  from_addr = os.getenv("EMAIL_FROM") or os.getenv("EMAIL_USER") or "babyguard@example.com"
  msg["Subject"] = subject
  msg["From"] = from_addr
  msg["To"] = req.to_email

  smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
  smtp_port = int(os.getenv("SMTP_PORT", "587"))
  smtp_user = os.getenv("EMAIL_USER")
  smtp_pass = os.getenv("EMAIL_PASS")

  if not smtp_user or not smtp_pass:
    raise RuntimeError("Email credentials not configured. Set EMAIL_USER and EMAIL_PASS in .env")

  with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()
    server.login(smtp_user, smtp_pass)
    server.send_message(msg)


@app.post("/notify/risk_email")
def notify_risk_email(req: RiskEmailRequest):
  try:
    send_risk_email(req)
    return {"status": "ok"}
  except Exception as e:
    print(f"[EMAIL] Error sending risk email: {e}")
    return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
  
@app.post("/predict/pose")
async def predict_pose(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = preprocess_pose_image(image, pose_val_transform, device)

        result = run_pose_gradcam(
            model=pose_model,
            device=device,
            image_pil=image,
            val_transform=pose_val_transform,
            class_names=pose_class_names,
            target_layer=pose_target_layer,
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


@app.post("/predict/expression")
async def predict_expression(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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