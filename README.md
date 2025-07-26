# Real World Image Classification Scenarios

This project demonstrates a real-world image classification scenario using a Convolutional Neural Network (CNN) to classify images of cats and dogs.

---

## Dataset

- The dataset is stored in Google Drive under a folder named `cats_dogs`.
- Folder structure:

---

## Model

- Implemented using **PyTorch**
- Network structure includes:
- Convolutional layers
- ReLU activations
- MaxPooling
- Fully connected linear layers
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

---

## Key Features

- Image augmentation with `transforms.Compose`
- `ImageFolder` based dataset loading
- Train-Validation-Test split with DataLoader
- Visualization of:
- Training and Validation Loss
- Training and Validation Accuracy
- Confusion Matrix
- Ability to predict new unseen images

---

## Full Code with Step-by-Step Explanation

### 1. Mount Google Drive and Import Libraries
```python
from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from PIL import Image
# Root dataset folder path
root = "/content/drive/MyDrive/cats_dogs"

# Image transformations
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
# Load datasets
train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

# Split training data into train and validation
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = train_data.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

