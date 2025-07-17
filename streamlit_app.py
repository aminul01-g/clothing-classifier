import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

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
st.write("Upload a 28x28 grayscale image as a PNG or CSV file. The model will predict the clothing class.")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (PNG or CSV)", type=["png", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        arr = np.loadtxt(uploaded_file, delimiter=',').flatten()
        st.write("CSV loaded. Shape:", arr.shape)
    else:
        img = Image.open(uploaded_file).convert('L').resize((28,28))
        arr = np.array(img).flatten()
        st.write("PNG loaded. Shape:", arr.shape)
    arr = arr / 255.0
    st.image(arr.reshape(28,28), caption="Input Image", width=140, channels="GRAY")
    x = torch.tensor([arr], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        st.success(f"Predicted class: {int(predicted.item())}") 