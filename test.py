import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision import models
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms



class Defect(Dataset):
    def __init__(self, folder_path=None, file_list=None, transforms=None):
        self.transforms = transforms
        self.files = []
        self.labels = []
        self.class_names = ["def_front", "ok_front"]
        self.cls_to_int = {cls: idx for idx, cls in enumerate(self.class_names)}

        if file_list is not None:
            self.files = file_list
            for filepath in file_list:
                if 'def_front' in filepath:
                    self.labels.append('def_front')
                else:
                    self.labels.append('ok_front')
        else:
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


def evaluate_model(model, data_loader, device):
    """Evaluate model on test set and generate confusion matrix"""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader: 
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX (TEST SET)")
    print("="*60)
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['defective', 'ok'], 
                yticklabels=['defective', 'ok'],
                cbar=True, square=True, linewidths=2)
    plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['defective', 'ok']))
    
    accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
    
    print("="*60)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Missed Defects (FN): {cm[0][1]}")
    print(f"False Alarms (FP): {cm[1][0]}")
    print("="*60)
    
    return cm


def show_predictions(model, dataset, device, num_images=12):
    """Show model predictions on random test samples"""
    fig, axes = plt.subplots(3, 4, figsize=(15, 12))
    axes = axes.flatten()

    class_names = ["Defect", "OK"]
    indices = np.random.choice(len(dataset), num_images, replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx]

            image_input = image.unsqueeze(0).to(device)
            output = model(image_input)
            pred_label = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_label].item() * 100

            img = image.permute(1, 2, 0).cpu().numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
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

    plt.suptitle('Defect Detection - Model Predictions on Test Set', fontsize=14)
    plt.tight_layout()
    plt.savefig('predictions_test.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Path to your dataset
    base_path = "path_dir"  # UPDATE THIS
    original_test_path = os.path.join(base_path, 'test')

    # Get test files
    all_test_files = []
    for class_name in ['def_front', 'ok_front']:
        class_folder = os.path.join(original_test_path, class_name)
        for img_name in os.listdir(class_folder):
            all_test_files.append(os.path.join(class_folder, img_name))

    # Split (same as train.py to get same test set)
    _, test_files = train_test_split(all_test_files, test_size=0.5, random_state=42)

    # Transforms
    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    # Create test dataset
    test_dataset = Defect(file_list=test_files, transforms=valid_transforms)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Load model
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("best.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluate
    cm = evaluate_model(model, test_loader, device)
    show_predictions(model, test_dataset, device, num_images=12)

    print("\nTest evaluation complete!")
