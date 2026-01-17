import io
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module

def make_png_bytes(w=64, h=64):
    img = Image.new("RGB", (w, h), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):

    def fake_load_pose_model(device):
        return object(), "pose_transform", ["abnormal", "normal"], "pose_layer"

    def fake_load_expression_model(device):
        return object(), "expr_transform", ["normal", "distressed"], "expr_layer"

    def fake_load_cry_model(device):
        return object(), "cry_transform", ["asphyxia", "hungry", "normal", "pain"], "cry_layer"

    monkeypatch.setattr(app_module, "load_pose_model", fake_load_pose_model)
    monkeypatch.setattr(app_module, "load_expression_model", fake_load_expression_model)
    monkeypatch.setattr(app_module, "load_cry_model", fake_load_cry_model)
    monkeypatch.setattr(app_module, "send_risk_email", lambda req: None)
    dummy_overlay = Image.new("RGB", (64, 64), (0, 0, 0))

    monkeypatch.setattr(
        app_module,
        "run_pose_gradcam",
        lambda **kwargs: {
            "label": "normal",
            "confidence": 0.95,
            "explanation": "dummy pose explanation",
            "overlay_image": dummy_overlay,
        },
    )

    monkeypatch.setattr(
        app_module,
        "run_expression_gradcam",
        lambda **kwargs: {
            "label": "normal",
            "confidence": 0.90,
            "explanation": "dummy expression explanation",
            "overlay_image": dummy_overlay,
        },
    )

    monkeypatch.setattr(
        app_module,
        "preprocess_cry_audio",
        lambda audio_bytes, transform, device: ("dummy_tensor", {"info": "dummy"}),
    )

    monkeypatch.setattr(
        app_module,
        "run_cry_gradcam_from_image",
        lambda **kwargs: {
            "label": "normal",
            "confidence": 0.85,
            "explanation": "dummy cry explanation",
            "overlay_image": dummy_overlay,
        },
    )

    # Using context manager ensures lifespan runs (so app.state gets filled)
    with TestClient(app_module.app) as c:
        yield c


def is_base64_png(s: str) -> bool:
    try:
        raw = base64.b64decode(s)
        return raw.startswith(b"\x89PNG")
    except Exception:
        return False


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "BabyGuard Backend API Running"


def test_notify_risk_email_ok(client):
    payload = {
        "to_email": "test@example.com",
        "risk_level": "high",
        "sleep_label": "abnormal",
        "expr_label": "distressed",
        "cry_label": "pain",
        "summary": "test summary",
    }
    r = client.post("/notify/risk_email", json=payload)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_pose_ok(client):
    files = {"file": ("test.png", make_png_bytes(), "image/png")}
    r = client.post("/predict/pose", files=files)
    assert r.status_code == 200

    data = r.json()
    assert data["label"] in ["abnormal", "normal"]
    assert isinstance(data["confidence"], float)
    assert isinstance(data["explanation"], str)
    assert isinstance(data["overlay_image"], str)
    assert is_base64_png(data["overlay_image"])


def test_predict_expression_ok(client):
    files = {"file": ("test.png", make_png_bytes(), "image/png")}
    r = client.post("/predict/expression", files=files)
    assert r.status_code == 200

    data = r.json()
    assert data["label"] in ["normal", "distressed"]
    assert isinstance(data["overlay_image"], str)
    assert is_base64_png(data["overlay_image"])


def test_predict_cry_ok(client):
    # dummy wav bytes (we mocked preprocess_cry_audio anyway)
    files = {"file": ("test.wav", b"RIFFxxxxWAVEfmt ", "audio/wav")}
    r = client.post("/predict/cry", files=files)
    assert r.status_code == 200

    data = r.json()
    assert data["label"] in ["asphyxia", "hungry", "normal", "pain"]
    assert isinstance(data["overlay_image"], str)
    assert is_base64_png(data["overlay_image"])


def test_predict_pose_missing_file_returns_422(client):
    r = client.post("/predict/pose")
    assert r.status_code == 422 
