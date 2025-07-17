# 🧠 Fashion Cloth Classifier

A lightweight Flask API that serves a deep learning model (ANN) trained on the Fashion MNIST dataset using PyTorch. This project demonstrates how to train, save, and deploy a neural network to classify clothing items like shirts, sneakers, trousers, and more via a RESTful API.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-%23EE4C2C)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 🚀 Features
- 📦 Simple Flask REST API for real-time predictions
- 🧠 Artificial Neural Network built with PyTorch
- 👕 Trained on the Fashion MNIST dataset
- 🧪 Predicts 10 categories of clothing from raw pixel input
- 🧰 Easily extendable for frontend or mobile integration

---

## 🗂️ Tech Stack
- Python 3
- PyTorch
- Flask
- NumPy
- Fashion MNIST dataset (CSV format)

---

## 📊 Fashion MNIST Classes

| Label | Class       |
|-------|-------------|
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## 🔧 Installation & Setup

```bash
git clone https://github.com/your-username/FashionClassifierAPI.git
cd FashionClassifierAPI
pip install -r requirements.txt
```
⚠️ Make sure to place your trained model file fashion_ann.pth inside this directory before running.

## ▶️ Run the App
```bash
python app.py
```
By default, the API will be available at:
http://127.0.0.1:5000

## 📄 License
This project is licensed under the MIT License.
