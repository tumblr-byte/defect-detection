import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import numpy as np
import os
import urllib.request

# Page configuration
st.set_page_config(
    page_title="Defect Detection System",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Industrial Defect Detection")
st.markdown("Upload an image of a casting product to detect defects")

# Model download function
def download_model():
    """Download model from GitHub releases if not present"""
    model_path = "best.pth"
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from GitHub... (this may take a moment)")
        
        # TODO: Update this URL with your actual GitHub release URL
        # After uploading to GitHub releases, replace with:
        # model_url = "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/best.pth"
        model_url = "https://github.com/tumblr-byte/defect-detection/releases/download/v1.0.0/best.pth"
        
        try:
            with st.spinner("Downloading model..."):
                urllib.request.urlretrieve(model_url, model_path)
            st.success(" Model downloaded successfully!")
        except Exception as e:
            st.error(f" Failed to download model: {e}")
            st.info("Please manually download 'best.pth' from GitHub releases and place it in the same directory.")
            st.stop()
    
    return model_path

# Load model
@st.cache_resource
def load_model():
    # Download model if needed
    model_path = download_model()
    
    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model, device

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100
    
    class_names = ["Defective", "OK"]
    return class_names[predicted_class], confidence, probabilities[0].cpu().numpy()

# Load model
try:
    model, device = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload an image of a casting product"
)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    # Preprocess and predict
    image_tensor = preprocess_image(image)
    prediction, confidence, probabilities = predict(model, image_tensor, device)
    
    with col2:
        st.subheader("Prediction Results")
        
        # Show prediction with color
        if prediction == "Defective":
            st.error(f"🔴 **{prediction}**")
        else:
            st.success(f"🟢 **{prediction}**")
        
        # Show confidence
        st.metric("Confidence", f"{confidence:.2f}%")
        
        # Show probabilities
        st.subheader("Class Probabilities")
        st.progress(probabilities[0], text=f"Defective: {probabilities[0]*100:.2f}%")
        st.progress(probabilities[1], text=f"OK: {probabilities[1]*100:.2f}%")
    
    # Additional info
    st.divider()
    st.info("""
    **About the Model:**
    - Architecture: ResNet-18 with Transfer Learning
    - Training Accuracy: 99.64%
    - Test Accuracy: 99.44%
    - False Positive Rate: 0%
    """)

else:
    st.info("👆 Please upload an image to get started")
    
    # Show example
    st.markdown("---")
    st.subheader("Example Use Cases")
    st.markdown("""
    This system can detect defects in:
    - Submersible pump impellers
    - Metal casting products
    - Manufacturing quality control
    """)
