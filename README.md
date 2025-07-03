```markdown
# 🐱 Cat vs 🐶 Dog Image Classifier

A lightweight image classification web app that identifies whether an uploaded image is a **cat** or a **dog**, using a custom-trained **CNN model** and a simple **Streamlit** frontend.

This project is suitable for demonstrating deep learning, image preprocessing, model training, and interactive web deployment — all in one streamlined solution.

---

## 🚀 Live Demo

🖥 Try the app here:  
**[https://cnn-catdog-app-t8wadjpx6ndryacnd3peqm.streamlit.app/](https://cnn-catdog-app-t8wadjpx6ndryacnd3peqm.streamlit.app/)**

---

## 📌 Features

- CNN architecture built with PyTorch
- Trained on 40 images (20 cats, 20 dogs) for quick demo/testing
- Custom preprocessing pipeline
- User-friendly drag-and-drop interface via Streamlit
- Real-time image prediction and confidence score
- Lightweight enough to deploy for free

---

## 🧠 Model Overview

- A simple convolutional neural network (CNN) with 2 conv layers and FC layers
- Optimized with Adam optimizer, cross-entropy loss
- Trained on a small sample from Hugging Face `cats_vs_dogs` dataset
- Model output: class (`cat` or `dog`) + confidence score

---

## 🖼 Sample Screenshots

| Upload Image | Result |
|--------------|--------|
| ![upload](webapp/assets/sample_1.jpg) | ![result](webapp/assets/sample_pred.jpg) |

---

## 📂 Project Structure

```

cnn-catdog-classifier/
│
├── data/                      # Sample images and Hugging Face loader
│   └── download\_from\_huggingface.py
│
├── model/                    # CNN architecture
│   └── cnn.py
│
├── trainer/                  # Training logic
│   └── train.py
│
├── inference/                # Prediction module
│   └── predict.py
│
├── webapp/                   # Streamlit frontend
│   └── app.py
│
├── test\_inference.py         # Quick script to test predictions
├── requirements.txt
└── README.md

````

---

## 💻 Local Setup

You can run the project locally with:

```bash
git clone https://github.com/TROY-PROGRAMMER/cnn-catdog-streamlit.git
cd cnn-catdog-streamlit
pip install -r requirements.txt
streamlit run webapp/app.py
````

---

## 📦 Dependencies

* Python 3.8+
* PyTorch
* torchvision
* Streamlit
* Hugging Face Datasets
* Pillow, tqdm, numpy

Install them via:

```bash
pip install -r requirements.txt
```

---

## 📌 Notes

* The model is trained on a small balanced dataset for demonstration. For production use, expand the dataset size.
* Streamlit Cloud handles the deployment automatically. You can fork this repo and redeploy under your own name.

---

## 📫 Contact

If you'd like to adapt this project to your needs or extend it (e.g., breed detection, larger datasets, mobile UI), feel free to reach out.

```

