import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
from torchvision import models


valid_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
])

#Dataset 
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


# Update this with your dataset path
base_path = "path/to/your/dataset"
valid_path = os.path.join(base_path, 'test')
valid_dataset = Defect(valid_path, valid_transforms)

#Load Model
model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("best.pth", map_location=device))
model = model.to(device)
model.eval()

def show_predictions(model, dataset, device, num_images=12):
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.flatten()
    
    class_names = ["Defect", "OK"]
    
    # Random samples
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx]
            
            # Predict
            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_label].item() * 100
            
            # Denormalize for display
            img = image.permute(1, 2, 0).cpu().numpy()
            img = img * np.array([0.32, 0.31, 0.33]) + np.array([0.41, 0.42, 0.43])
            img = np.clip(img, 0, 1)
            
            axes[i].imshow(img)
            
            color = 'green' if pred_label == true_label else 'red'
            
            axes[i].set_title(
                f"True: {class_names[true_label]}\n"
                f"Pred: {class_names[pred_label]} ({confidence:.1f}%)",
                color=color,
                fontsize=11,
                fontweight='bold'
            )
            axes[i].axis('off')
    
    plt.suptitle('Defect Detection - Model Predictions on Validation Set')
    plt.tight_layout()
    plt.show()


show_predictions(model, valid_dataset, device, num_images=12)
