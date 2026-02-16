import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DefectDataset(Dataset):
    """Custom Dataset for defect detection"""
    
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
            # Load from folder path
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


def get_transforms():
    """Get data transforms for training and validation"""
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, valid_transforms
