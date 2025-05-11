import string
import torch
import torchvision
import numpy as np
import PIL
import cv2 as cv
import kivy

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import kagglehub
import pandas as pd
import struct
from os.path import join

"""
print(f"torch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"numpy version: {np.__version__}")
print(f"pillow version: {PIL.__version__}")
print(f"opencv-python version: {cv.__version__}")
print(f"kivy version: {kivy.__version__}")
"""

class MNISTDataLoader(object):
    def __init__(self, train_file_path, train_label_path, test_file_path, test_label_path):
        #Just basic init
        self.train_file_path = train_file_path
        self.train_label_path = train_label_path
        self.test_file_path = test_file_path
        self.test_label_path = test_label_path

    def read_images_labels(self, image_file_path, label_file_path):
        labels = []

        with open(label_file_path, 'rb') as file:
            #Read to determine magic as file type and size as how many labels there are
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = np.frombuffer(file.read(), dtype=np.uint8)        

        with open(image_file_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = np.frombuffer(file.read(), dtype=np.uint8)

        images = []

        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_file_path, self.train_label_path)
        x_test, y_test = self.read_images_labels(self.test_file_path, self.test_label_path)
        return (x_train, y_train),(x_test, y_test)

class HiMom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

import random
import torch.nn.functional as F

if __name__ == "__main__":
    model = HiMom().to("cpu")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_file_path = r"C:\Users\ADMIN\.cache\kagglehub\datasets\zalando-research\fashionmnist\versions\4\FashionMNIST\raw\train-images-idx3-ubyte"
    train_label_path = r"C:\Users\ADMIN\.cache\kagglehub\datasets\zalando-research\fashionmnist\versions\4\FashionMNIST\raw\train-labels-idx1-ubyte"
    test_file_path = r"C:\Users\ADMIN\.cache\kagglehub\datasets\zalando-research\fashionmnist\versions\4\FashionMNIST\raw\t10k-images-idx3-ubyte"
    test_label_path = r"C:\Users\ADMIN\.cache\kagglehub\datasets\zalando-research\fashionmnist\versions\4\FashionMNIST\raw\t10k-labels-idx1-ubyte"

    mnist_dataloader = MNISTDataLoader(train_file_path, train_label_path, test_file_path, test_label_path)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # Convert to torch tensors
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(np.array(x_test), dtype=torch.float32) / 255.0
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Add channel dimension (needed for consistency, though your model flattens)
    x_train = x_train.unsqueeze(1)
    x_test = x_test.unsqueeze(1)

    # Create data loaders
    train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(list(zip(x_test, y_test)), batch_size=64)

    # Train loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            pred = model(batch_x)
            predicted_labels = pred.argmax(1)
            correct += (predicted_labels == batch_y).sum().item()
            total += batch_y.size(0)
    print(f"Test accuracy: {correct / total * 100:.2f}%")