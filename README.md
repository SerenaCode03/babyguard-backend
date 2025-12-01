# 🍼 BabyGuard Backend (FastAPI)

This is the backend service for **BabyGuard**, providing inference APIs for:

- Infant **pose classification**
- Infant **facial expression classification**
- Infant **cry classification**
- **Grad-CAM explainability** heatmaps for all models

The backend communicates with the Flutter mobile app and supports image/audio uploads.

---

## 🚀 Features

- **FastAPI server** with automatic Swagger UI
- **PyTorch model loading**:
  - EfficientNet-B0 (Pose)
  - MobileNetV3-Small (Expression)
  - ResNet18 (Cry)
- **Robust Preprocessing**:
  - Pose + expression image transforms
  - Cry audio → Mel spectrogram → Model input
- **Grad-CAM utilities** for Explainable AI (XAI)

---

## 📦 Requirements

This project uses **[uv](https://github.com/astral-sh/uv)** for extremely fast environment & dependency management.

Install `uv` (if not already installed):

```
pip install uv
```

## 🔧 Installation

1. **Clone the repository**
   ```
   git clone <your-repository-url>
   cd babyguard-backend
    ```

2. **Install dependencies**
    ```
    uv sync
    ```

## ▶️ Running the Backend
Start the server with hot-reloading (useful for development):
```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at: http://127.0.0.1:8000

## 📌 API Documentation
Once the server is running, you can access the interactive Swagger UI to test endpoints directly in your browser:

👉 Open: http://127.0.0.1:8000/docs