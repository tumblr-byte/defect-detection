import os 
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models



base_path = #path of your dataset

train_path = os.path.join(base_path, 'train')
valid_path = os.path.join(base_path, 'test')


# Transforms 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=14),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.41, 0.42, 0.43), std=(0.32, 0.31, 0.33))
])

valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.41, 0.42, 0.43), std=(0.32, 0.31, 0.33))
])


# Dataset 
class Defect(Dataset):
    def __init__(self, folder_path, transforms=None):
        self.folder_path = folder_path
        self.transforms = transforms
        self.files = []
        self.labels = []

        self.class_names = ["def_front", "ok_front"]
        self.cls_to_int = {cls: idx for idx, cls in enumerate(self.class_names)}

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


# DataLoader 
train_dataset = Defect(train_path, train_transforms)
valid_dataset = Defect(valid_path, valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8)


# Model
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

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
        model.train()
        train_loss = 0
        train_acc = 0

        for image, label in tqdm(train_loader):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += cal_acc(label, output)

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

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

        valid_loss /= len(valid_loader)
        valid_acc /= len(valid_loader)

        history["valid_loss"].append(valid_loss)
        history["valid_acc"].append(valid_acc)

        print(f"{epoch + 1}/{epochs}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}")

       
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



run_model(model, criterion, optimizer, device, train_loader, valid_loader, epochs=100, patience=10, output_path="best.pth"
)
