from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import balanced_accuracy_score

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

# Unique labels and counts
unique_labels, counts = np.unique(train_data['labels'], return_counts=True)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0
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

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Dataset preparation
dataset = CustomDataset(train_data['images'], train_data['labels'], transform=None)
test_dataset = CustomDataset(test_data['images'], transform=None)
from sklearn.model_selection import train_test_split

labels = train_data['labels']
train_indices, val_indices = train_test_split(
    range(len(labels)), test_size=0.2, stratify=labels
)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(unique_labels))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paths for saving
checkpoint_path = "checkpoint_resnet18.pth"
best_model_path = "best_model_resnet18.pth"

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {start_epoch}")
        return model, optimizer, start_epoch
    print("No checkpoint found. Starting from scratch.")
    return model, optimizer, 0

# Function to train and validate
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, checkpoint_path, best_model_path,predictions_file='train_val_predictions.csv'):
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    best_val_accuracy = 0.0

    # For saving predictions
    all_train_predictions = []
    all_val_predictions = []

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_labels = []
        train_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(predicted.cpu().numpy())

        train_accuracy = train_correct / train_total
        for label, pred in zip(train_labels, train_preds):
            all_train_predictions.append((label, pred, "train"))

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())

        val_accuracy = val_correct / val_total
        for label, pred in zip(val_labels, val_preds):
            all_val_predictions.append((label, pred, "val"))

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}", end=" ")
        print('Balanced Accuracy:', 'Train:',balanced_accuracy_score(train_labels, train_preds), 'Val:', balanced_accuracy_score(val_labels, val_preds))

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with Val Accuracy: {val_accuracy:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Save train and validation predictions
    predictions_df = pd.DataFrame(
        all_train_predictions + all_val_predictions,
        columns=["True Label", "Predicted Label", "Set"]
    )
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Train and validation predictions saved to '{predictions_file}.csv'")

# Generate metrics and confusion matrices
def generate_metrics(predictions_file):
    predictions_df = pd.read_csv(predictions_file)
    for set_type in ["train", "val", "both"]:
        subset = predictions_df if set_type == "both" else predictions_df[predictions_df["Set"] == set_type]
        true_labels = subset["True Label"]
        pred_labels = subset["Predicted Label"]

        print(f"\nMetrics for {set_type.capitalize()} Set:")
        print(confusion_matrix(true_labels, pred_labels))
        print(classification_report(true_labels, pred_labels, target_names=[f"Class {i}" for i in range(len(unique_labels))]))

# Train and validate the model
predictions_file = "train_val_predictions_resnet18.csv"
train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=10, checkpoint_path=checkpoint_path, best_model_path=best_model_path, predictions_file=predictions_file)

# Generate metrics
generate_metrics(predictions_file)
