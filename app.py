import base64, io
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ----- FastAPI app -----
app = FastAPI(title="BabyGuard Grad-CAM API (MVP)", version="0.1")

# Allow your phone/app to call this API (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for demo; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LABELS = ["asphyxia", "hungry", "normal", "pain"]

class GradcamRequest(BaseModel):
    image: str                   # base64 PNG/JPEG of spectrogram or frame
    mode: Optional[str] = "mel"  # optional: "mel" or "frame"
    target_class: Optional[str] = None

class GradcamResponse(BaseModel):
    probs: List[float]
    top_label: str
    top_index: int
    heatmap_png: str             # base64 PNG (placeholder)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/gradcam", response_model=GradcamResponse)
def gradcam(payload: GradcamRequest):
    # --- decode image just to validate payload (will raise if bad) ---
    raw = base64.b64decode(payload.image)
    _ = Image.open(io.BytesIO(raw)).convert("RGB")  # not used yet; just validation

    # --- DUMMY prediction for now (deterministic placeholder) ---
    # Replace with your real model + Grad-CAM later.
    probs = [0.10, 0.15, 0.65, 0.10]  # asphyxia, hungry, normal, pain
    top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

    # --- DUMMY heatmap: 1x1 transparent PNG (so your app pipeline works) ---
    empty_png_b64 = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01"
        b"\x0b\xe7\x02\x9e\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode("utf-8")

    return GradcamResponse(
        probs=probs,
        top_label=LABELS[top_idx],
        top_index=top_idx,
        heatmap_png=empty_png_b64,
    )

