# preprocess/cry_preprocess.py
import io
import os
import tempfile
import uuid
import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

# Set non-interactive backend to prevent GUI errors on servers
matplotlib.use('Agg')

# 1. Transform matching your training code
CRY_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),            # -> [3, H, W], values in [0,1]
])

def split_audio(y: np.ndarray, sr: int, segment_duration: float = 1.0):
    """Split waveform into fixed-length segments (e.g. 1 second)."""
    segment_len = int(sr * segment_duration)
    segments = []
    # If audio is shorter than 1s, pad it? 
    # For now, we assume audio >= 1s. If < 1s, we might need padding logic.
    if len(y) < segment_len:
        # Simple zero padding
        y = np.pad(y, (0, segment_len - len(y)))
        
    for i in range(0, len(y), segment_len):
        seg = y[i:i + segment_len]
        if len(seg) == segment_len:
            segments.append(seg)
    return segments

def segment_to_mel_image(segment: np.ndarray, sr: int = 16000, n_mels: int = 128) -> Image.Image:
    """
    Convert a single 1s waveform segment into a Mel spectrogram image 
    using Matplotlib (to match training pipeline).
    """
    # Peak normalize
    if np.max(np.abs(segment)) > 0:
        segment = segment / np.max(np.abs(segment))

    mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Render as image in memory
    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis("off")
    # inferno cmap matches many standard implementations; ensure it matches your training
    plt.imshow(mel_db, cmap="inferno", aspect="auto") 

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig) # Important: Close figure to free memory
    buf.seek(0)

    return Image.open(buf).convert("L") # Convert to Grayscale (PIL)

def preprocess_cry_audio(audio_bytes: bytes, transform=CRY_TRANSFORM, device: str = "cpu"):
    """
    Main entry point for app.py.
    
    1. Saves bytes to temp file.
    2. Loads via Librosa.
    3. Splits into 1s segments.
    4. Converts to MelImages -> Tensors.
    
    Returns:
        input_tensor: torch.Tensor [N, 3, 224, 224] (Batch of N segments)
        extra_info: List[Image.Image] (The original Mel spectrogram images for GradCAM overlay)
    """
    # 1. Write bytes to a temp file so Librosa can read it
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
    
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        # 2. Load and Resample
        y, sr = librosa.load(temp_path, sr=16000)
        
        # 3. Split
        segments = split_audio(y, sr, segment_duration=1.0)
        
        if not segments:
            raise ValueError("Audio is too short (less than 1 second).")

        tensors_list = []
        mel_images_list = []

        # 4. Process each segment
        for seg in segments:
            # Generate Spectrogram Image
            mel_img = segment_to_mel_image(seg, sr=sr)
            
            # Ensure RGB/3-channel for model input if transform expects it
            # Your transform has Grayscale(3), so 'L' mode is fine as input to transform
            # Apply Transform
            tensor = transform(mel_img) # [3, 224, 224]
            
            tensors_list.append(tensor)
            mel_images_list.append(mel_img)

        # Stack into a batch: [N, 3, 224, 224]
        input_tensor = torch.stack(tensors_list).to(device)
        
        return input_tensor, mel_images_list

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)