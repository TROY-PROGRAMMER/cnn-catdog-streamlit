# webapp/app.py

import streamlit as st
from PIL import Image
import os
import torch

# Add root directory to Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predict import load_model, predict_image

# Page configuration
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾")

st.title("🐶 Cat vs 🐱 Dog Classifier")
st.write("Upload an image and the model will predict whether it's a **cat** or a **dog**.")

# Load model once and cache
@st.cache_resource
def load_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("model/checkpoint.pth", device)
    return model, device

model, device = load_classifier()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file for predict()
    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    # Predict
    label, confidence = predict_image(temp_path, model, device)

    # Show result
    st.markdown(f"### 🧠 Prediction: **{label.upper()}**")
    st.markdown(f"**Confidence:** {confidence:.2%}")

    # Clean up temp file (optional)
    os.remove(temp_path)

# Optional: show some sample image previews
st.markdown("---")
st.markdown("💡 Sample Images:")
sample_dir = "data/processed"
if os.path.isdir(sample_dir):
    samples = sorted(os.listdir(sample_dir))[:6]
    cols = st.columns(len(samples))
    for col, fname in zip(cols, samples):
        path = os.path.join(sample_dir, fname)
        col.image(path, use_column_width=True, caption=fname)
