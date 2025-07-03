from inference.predict import load_model, predict_image

model = load_model("model/checkpoint.pth")
label, confidence = predict_image("data/processed/cat_001.jpg", model)
print(f"Prediction: {label} ({confidence:.2f})")
