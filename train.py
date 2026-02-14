import os
import shutil
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models
from sklearn.model_selection import train_test_split


# Path to your dataset (update this with your actual path)
base_path = path_dir
train_path = os.path.join(base_path, 'train')
original_test_path = os.path.join(base_path, 'test')




# Get all files from original test folder
all_test_files = []
for class_name in ['def_front', 'ok_front']:
    class_folder = os.path.join(original_test_path, class_name)
    for img_name in os.listdir(class_folder):
        all_test_files.append(os.path.join(class_folder, img_name))

# Simple split: 50% valid, 50% test
valid_files, test_files = train_test_split(all_test_files, test_size=0.5, random_state=42)

print(f"Total test files: {len(all_test_files)}")
print(f"Valid files: {len(valid_files)}")
print(f"Test files: {len(test_files)}")


# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
])

# Only normalization for validation
valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
])


# Custom Dataset class for defect detection
class Defect(Dataset):
    def __init__(self, folder_path=None, file_list=None, transforms=None):
        self.transforms = transforms
        self.files = []
        self.labels = []
        self.class_names = ["def_front", "ok_front"]
        self.cls_to_int = {cls: idx for idx, cls in enumerate(self.class_names)}

        if file_list is not None:
            # Use provided file list
            self.files = file_list
            # Extract labels from file paths
            for filepath in file_list:
                if 'def_front' in filepath:
                    self.labels.append('def_front')
                else:
                    self.labels.append('ok_front')
        else:
            # Load from folder path (original behavior)
            for class_name in self.class_names:
                class_folder_path = os.path.join(folder_path, class_name)
                for img_name in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, img_name)
                    self.files.append(image_path)
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        label = self.cls_to_int[self.labels[idx]]
        return image, label





# Create datasets
train_dataset = Defect(folder_path=train_path, transforms=train_transforms)
valid_dataset = Defect(file_list=valid_files, transforms=valid_transforms)
test_dataset = Defect(file_list=test_files, transforms=valid_transforms)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)


# Model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final layer for binary classification
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Calculate accuracy
def cal_acc(true, pred):
    pred = torch.argmax(pred, dim=1)
    acc = (true == pred).float().mean().item()
    return round(acc, 4)


def run_model(model, criterion, optimizer, device, train_loader, valid_loader, epochs=100, patience=10, output_path="best.pth"):
    history = {
        "train_loss": [],
        "train_acc": [],
        "valid_loss": [],
        "valid_acc": []
    }

    best_loss = np.inf
    counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_acc = 0

        for image, label in tqdm(train_loader):
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            output = model(image)
            loss = criterion(output, label)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += cal_acc(label, output)

        # Calculate average training metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation phase
        model.eval()
        valid_loss = 0
        valid_acc = 0

        with torch.no_grad():
            for image, label in tqdm(valid_loader):
                image = image.to(device)
                label = label.to(device)

                output = model(image)
                loss = criterion(output, label)

                valid_loss += loss.item()
                valid_acc += cal_acc(label, output)

        # Calculate average validation metrics
        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)

        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(f"{epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}")

        # Early stopping check
        if valid_loss <= best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), output_path)
            counter = 0
            print("model is saved")
        else:
            counter += 1
            print("no improvement at all")

        if counter >= patience:
            print("early stopping triggered")
            break

    return history



history = run_model(model, criterion, optimizer, device, train_loader, valid_loader, epochs=100, patience=10, output_path="best.pth")
print("\nTraining complete!")

