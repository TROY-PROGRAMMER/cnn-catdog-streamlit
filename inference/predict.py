# inference/predict.py

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import os

from model.cnn import SimpleCNN

# Define preprocessing to match training phase
def get_transform(image_size=128):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

# Load model from file
def load_model(model_path="model/checkpoint.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict single image file
def predict_image(image_path, model, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = "cat" if predicted.item() == 0 else "dog"
    return label, confidence.item()
