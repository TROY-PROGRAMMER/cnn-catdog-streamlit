# trainer/train.py

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from model.cnn import SimpleCNN
from data_loader.data_loader import load_dataset
from utils.helper import load_label_map_from_csv

def train(model, train_loader, val_loader, device, epochs=10, lr=0.001, save_path="model/checkpoint.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        # Validation step
        val_acc = evaluate(model, val_loader, device)
        print(f"\n🧪 Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            print(f"💾 Saving best model: Accuracy improved from {best_val_acc:.2f}% → {val_acc:.2f}%")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100. * correct / total


if __name__ == "__main__":
    # Configuration
    csv_path = "data/sample.csv"
    image_dir = "data/processed"
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and labels
    label_map = load_label_map_from_csv(csv_path)
    train_loader, val_loader = load_dataset(image_dir, label_map, batch_size=batch_size)

    # Initialize model
    model = SimpleCNN(num_classes=2).to(device)

    # Start training
    train(model, train_loader, val_loader, device, epochs=num_epochs, lr=learning_rate)
