import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
import numpy as np
import pandas as pd
import pickle

# Load data function
def load_data(path_to_data):
    with open(path_to_data, 'rb') as f:
        return pickle.load(f)

# Paths to data
test_path = 'data/test_data.pkl'
train_path = 'data/train_data.pkl'

# Load train and test data
test_data = load_data(test_path)
train_data = load_data(train_path)

unique_labels, counts = np.unique(train_data['labels'], return_counts=True)
label_counts = dict(zip(unique_labels, counts))

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0  # Normalize and add channel
        self.labels = torch.tensor(labels).long() if labels is not None else None
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image

# Data Augmentation for training
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Normalize test and validation data
test_transforms = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets
dataset = CustomDataset(train_data['images'], train_data['labels'], transform=None)
test_dataset = CustomDataset(test_data['images'], transform=None)
from sklearn.model_selection import train_test_split
labels = train_data['labels']
# Perform stratified split
train_indices, val_indices = train_test_split(
    range(len(labels)),  # Indices of the dataset
    test_size=0.2,       # Validation set size (20%)
    stratify=labels      # Preserve class distribution
)

# Create stratified datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Adjust for single-channel input (grayscale images)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Adjust final layer for 4 classes
model.fc = nn.Linear(model.fc.in_features, len(unique_labels))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader):.4f}, "
          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {correct/total:.4f}")

# Generate predictions for the test set
model.eval()
predictions = []
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Save predictions to CSV
ids = np.arange(1, len(predictions) + 1)  # IDs start from 1
output_df = pd.DataFrame({'ID': ids, 'Class': predictions})
output_df.to_csv('test_predictions_pretrained.csv', index=False)
print("Predictions saved to 'test_predictions_pretrained.csv'")
