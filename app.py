import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import numpy as np

# Model definition (copied from notebook)
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

# Initialize and load model
INPUT_SIZE = 784
model = MyNN(INPUT_SIZE)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'pixels' not in data:
        return jsonify({'error': 'Missing pixels field'}), 400
    pixels = data['pixels']
    if len(pixels) != 784:
        return jsonify({'error': 'Input must be a list of 784 values'}), 400
    # Convert to tensor
    x = torch.tensor([pixels], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        return jsonify({'class': int(predicted.item())})

if __name__ == '__main__':
    app.run(debug=True) 