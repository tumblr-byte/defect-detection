# Industrial Defect Detection using Deep Learning

Binary image classification for detecting manufacturing defects in metal casting using transfer learning with ResNet-18.

## Project Overview

Developed a computer vision model to automatically detect defects in industrial metal casting components, achieving **99.6% accuracy** on validation data.

## Results

- **Accuracy**: 99.72% (713/715 correct predictions)
- **Defect Detection Rate**: 99.56% (451/453 defects caught)
- **False Positive Rate**: 0% (no good parts rejected)
- **Model**: ResNet-18 with transfer learning

### Confusion Matrix
<img width="1600" height="1200" alt="Image" src="https://github.com/user-attachments/assets/95b04a61-6258-43cf-b64d-f1978276a19d" />

### Validation Predictions
<img width="1773" height="663" alt="Image" src="https://github.com/user-attachments/assets/dd9ad5fd-87b0-431a-afc9-ae170220d93b" />

<img width="1480" height="1180" alt="Image" src="https://github.com/user-attachments/assets/c967d6f7-99c4-4e91-aed4-263a7c3c14d5" />

### Training History


<img width="827" height="476" alt="Image" src="https://github.com/user-attachments/assets/88b1e64a-2d69-432f-9c76-ad732bc12960" />
<img width="1255" height="455" alt="Image" src="https://github.com/user-attachments/assets/3d2b667d-140e-4a4b-9f93-495fbde0a505" />

## Tech Stack

- **Framework**: PyTorch
- **Model**: ResNet-18 (pretrained on ImageNet)
- **Techniques**: Transfer learning, data augmentation, early stopping
- **Libraries**: torchvision, matplotlib, scikit-learn

## Dataset
https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product


## Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Testing/Visualization
```bash
python test.py
```

## Model Architecture

- Base: ResNet-18 (pretrained on ImageNet)
- Modified final layer for binary classification
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: CrossEntropyLoss
- Early stopping with patience=10

## Key Features

- **Data Augmentation**: Random rotation, horizontal flip, color jitter
- **Early Stopping**: Prevents overfitting
- **Transfer Learning**: Leverages ImageNet pretrained weights
- **Validation**: Held-out test set for unbiased evaluation

## Industrial Applications

This model can be deployed for:
- Automated quality control in manufacturing
- Real-time defect detection on production lines
- Reducing manual inspection costs

## Future Improvements

-  Deploy as REST API using FastAPI
-  Add Grad-CAM visualization for explainability
-  Optimize for edge deployment (ONNX/TensorRT)
-  Expand to multi-class defect classification
