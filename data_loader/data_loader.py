# cnn_pet_classifier/data_loader/data_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms


class PetDataset(Dataset):
    def __init__(self, image_dir, label_map, transform=None):
        self.image_dir = image_dir
        self.label_map = label_map  # Dict[str filename] = 0/1
        self.transform = transform
        self.image_files = list(label_map.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[img_name]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(image_size=128):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    return train_transform, val_transform


def load_dataset(image_dir, label_map, batch_size=32, image_size=128, val_split=0.2):
    train_tf, val_tf = get_transforms(image_size)

    full_dataset = PetDataset(image_dir, label_map, transform=train_tf)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    # Replace transform for val_ds
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
