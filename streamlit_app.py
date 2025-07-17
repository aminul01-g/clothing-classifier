import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Fashion-MNIST class names
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Model definition (same as notebook)
class MyNN(nn.Module):
    def __init__(self, num_features, dropout_prob=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.model(x)

# Load model
model = MyNN(784)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

st.title("Clothing Classifier (Fashion-MNIST)")
st.write("Upload any grayscale image. It will be resized to 28x28 and classified into one of the Fashion-MNIST classes.")

# --- Display class names as a grid (no images) ---
st.subheader("Fashion-MNIST Classes")
cols = st.columns(5)
for i, class_name in enumerate(CLASS_NAMES):
    with cols[i % 5]:
        st.markdown(f"**{i}: {class_name}**")

# --- File uploader and prediction ---
st.subheader("Upload and Predict")
uploaded_file = st.file_uploader("Upload a grayscale image (any size, PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and show original image
    orig_img = Image.open(uploaded_file).convert('L')
    st.write("Original Image:")
    st.image(orig_img, width=140, caption="Original Uploaded Image", channels="GRAY")

    # Resize to 28x28
    img_28 = orig_img.resize((28, 28))
    arr = np.array(img_28).flatten()
    arr_norm = arr / 255.0
    st.write("Resized 28x28 Image:")
    st.image(img_28, width=140, caption="Resized 28x28", channels="GRAY")

    # Predict
    x = torch.tensor([arr_norm], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        pred_class = int(predicted.item())
        st.success(f"Predicted class: {pred_class} - {CLASS_NAMES[pred_class]}") 