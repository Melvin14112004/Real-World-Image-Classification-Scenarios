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
```
### 2. Dataset Preparation
```python
root = "/content/drive/MyDrive/cats_dogs"

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

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_subset, val_subset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = train_data.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
### 3. CNN Model Definition
```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNNModel().to(device)
```
### 4. Training Loop
```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss, total_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
        
        val_loss, val_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        print(f"Epoch {epoch+1}: Train Acc: {total_correct/len(train_subset):.4f}, Val Acc: {val_correct/len(val_subset):.4f}")

train_model(model, train_loader, val_loader, epochs=10)
```
### 5. Evaluation and Confusion Matrix
```python
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()
```
## Output
### Sample
<img width="485" height="508" alt="Screenshot 2025-07-26 172617" src="https://github.com/user-attachments/assets/6076abdc-6ca4-4e89-b1db-5d8785d785c0" />
### Confusion Matrix
<img width="662" height="595" alt="Screenshot 2025-07-26 172606" src="https://github.com/user-attachments/assets/233b0777-2a1b-4804-aa9e-d3303d7e5596" />


## Result
Model successfully trained and evaluated on the cats vs dogs dataset with visualization and predictions on unseen images.



