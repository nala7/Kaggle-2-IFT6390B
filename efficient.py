from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

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


# Selected model parameter
#selected_model = "resnet18"  # Choose between "resnet18" and "efficientnet_b0"
selected_model = 'efficientnet_b0'

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
        # Preprocess images to PIL format during initialization
        self.images = [transforms.functional.to_pil_image(img.squeeze()) for img in images]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # Pre-converted PIL Image
        if self.transform:
            image = self.transform(image)  # Apply transformations

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image

# Transforms (Adjust for model-specific requirements)
if selected_model == "resnet18":
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_channels = 1  # Grayscale input
elif selected_model == "efficientnet_b0":
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_channels = 1  # Grayscale input

# Dataset preparation
dataset = CustomDataset(train_data['images'], train_data['labels'], transform=train_transforms)
test_dataset = CustomDataset(test_data['images'], transform=train_transforms)

labels = train_data['labels']
train_indices, val_indices = train_test_split(
    range(len(labels)), test_size=0.2, stratify=labels, random_state=42, shuffle=True
)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# DataLoaders
batch_size = 64
class_counts = np.bincount(train_data['labels'])
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
subset_indices = train_dataset.indices
subset_labels = [train_data['labels'][i] for i in subset_indices]
sample_weights = [class_weights[label] for label in subset_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Function to dynamically create the model
def get_model(selected_model, num_classes, input_channels):
    if selected_model == "resnet18":
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif selected_model == "efficientnet_b0":
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
        model.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {selected_model} is not supported.")
    return model

# Model initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(selected_model, num_classes=len(unique_labels), input_channels=input_channels)
model = model.to(device)

# Switch: Train only FC layer or entire model
train_fc_only = False
fc_str = "FC" if train_fc_only else "Entire"
if train_fc_only:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)

# Paths for saving
checkpoint_path = f"checkpoint2_{selected_model}_{fc_str}.pth"
best_model_path = f"best_model2_{selected_model}_{fc_str}.pth"

# TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{selected_model}_{fc_str}_experiment")

# Training and validation function
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, checkpoint_path, best_model_path, writer, device):

    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)

        # load best model and its accuracy
    if os.path.exists(best_model_path):
        val_labels = []
        val_preds = []
        val_correct=[]
        val_total=[]

        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
        best_score = balanced_accuracy_score(val_labels, val_preds)
        print(f"Best model loaded from {best_model_path} with Val Accuracy: {val_accuracy:.4f}, Val Balanced Accuracy: {val_balanced_accuracy:.4f}")
    else:
        best_score = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

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

        train_accuracy = train_correct / train_total
        train_balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Balanced Accuracy/Train", train_balanced_accuracy, epoch)


        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_balanced_accuracy = balanced_accuracy_score(labels.cpu().numpy(), predicted.cpu().numpy())
        writer.add_scalar("Loss/Validation", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        writer.add_scalar("Balanced Accuracy/Validation", val_balanced_accuracy, epoch)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Train Balanced Accuracy: {train_balanced_accuracy:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}"
              f"Val Balanced Accuracy: {val_balanced_accuracy:.4f}")

        scheduler.step(val_balanced_accuracy)

        # Save the best model
        this_score = val_balanced_accuracy
        if this_score > best_score:
            best_score = this_score
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with score: {best_score:.4f}")

# Train the model
train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=20, checkpoint_path=checkpoint_path, best_model_path=best_model_path, writer=writer, device=device)

# Close TensorBoard writer
writer.close()

import pandas as pd

def generate_test_predictions(test_loader, model, device, output_csv_path):
    """
    Generate predictions for the test dataset and save them in a CSV file.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model (torch.nn.Module): Trained model to generate predictions.
        device (str): Device to use ('cuda' or 'cpu').
        output_csv_path (str): Path to save the CSV file with predictions.
    """
    model.eval()  # Set model to evaluation mode
    predictions = []

    with torch.no_grad():  # No gradient computation needed for inference
        for images in test_loader:
            images = images.to(device)

            # Forward pass to get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the class index with highest score

            predictions.extend(predicted.cpu().numpy())

    # Create a DataFrame for saving predictions
    test_ids = np.arange(1, len(predictions) + 1)  # IDs starting from 1
    predictions_df = pd.DataFrame({'ID': test_ids, 'Class': predictions})

    # Save to CSV
    predictions_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")


# Load the best model weights
model.load_state_dict(torch.load(best_model_path))

# Generate predictions for the test dataset
output_csv_path = f"test_predictions2_{selected_model}_{fc_str}.csv"
generate_test_predictions(test_loader, model, device, output_csv_path)
