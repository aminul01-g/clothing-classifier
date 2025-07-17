from flask import Flask, request, jsonify
import torch
import numpy as np
from model import MyNN

app = Flask(__name__)

# Load model
model = MyNN(num_features=784)
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
model.eval()

@app.route("/")
def home():
    return "Fashion MNIST ANN API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = np.array(data["input"]).reshape(1, 784) / 255.0
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
