import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import pickle
from timm import create_model
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

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
class_counts = counts.tolist()  # Class counts
print(f"Class counts: {class_counts}")

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

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
subset_indices = train_dataset.indices
subset_labels = [train_data['labels'][i] for i in subset_indices]
sample_weights = [class_weights[label] for label in subset_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize DeiT-Small for 1-channel input
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = create_model('deit_small_patch16_224', pretrained=True, num_classes=len(unique_labels))
model.patch_embed.proj = nn.Conv2d(1, model.patch_embed.proj.out_channels, kernel_size=16, stride=16)
model = model.to(device)

# Loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Optimizer and Scheduler
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Paths for saving
checkpoint_path = "checkpoint_deit_small.pth"
best_model_path = "best_model_deit_small.pth"

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/deit_small_experiment")

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

import time
# Training and validation function
def train_and_validate(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    checkpoint_path,
    best_model_path,
    writer,
    device,
    scoring_metric="accuracy",  # Choose between "accuracy" and "balanced_accuracy"
):
    """
    Train and validate the model with saving of the best model based on a chosen metric.

    Args:
        model (torch.nn.Module): The model to train and validate.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epochs (int): Number of training epochs.
        checkpoint_path (str): Path to save checkpoints.
        best_model_path (str): Path to save the best model.
        writer (SummaryWriter): TensorBoard writer.
        device (str): Device to use ('cuda' or 'cpu').
        scoring_metric (str): Metric to evaluate the best model. Options: "accuracy" or "balanced_accuracy".
    """
    def calculate_score(labels, preds, metric):
        if metric == "accuracy":
            return (labels == preds).sum() / len(labels)
        elif metric == "balanced_accuracy":
            return balanced_accuracy_score(labels, preds)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # Load checkpoint if available
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Load best model and its score
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Best model loaded from {best_model_path}")
        model.eval()
        val_labels, val_preds = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
        best_score = calculate_score(np.array(val_labels), np.array(val_preds), scoring_metric)
        print(f"Best {scoring_metric} score of the loaded model: {best_score:.4f}")
    else:
        best_score = 0.0
        print("No best model found. Starting fresh.")

    # Start training
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_labels, train_preds = [], []

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
        train_score = calculate_score(np.array(train_labels), np.array(train_preds), scoring_metric)
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar(f"{scoring_metric.capitalize()}/Train", train_score, epoch)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_labels, val_preds = [], []

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
        val_score = calculate_score(np.array(val_labels), np.array(val_preds), scoring_metric)
        writer.add_scalar("Loss/Validation", val_loss / len(val_loader), epoch)
        writer.add_scalar(f"{scoring_metric.capitalize()}/Validation", val_score, epoch)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train {scoring_metric.capitalize()}: {train_score:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val {scoring_metric.capitalize()}: {val_score:.4f}")

        # Step the scheduler
        scheduler.step(val_score)

        # Save the best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved to {best_model_path} with Val {scoring_metric.capitalize()}: {best_score:.4f}")

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        time.sleep(300) # Sleep for 5 minutes to let the GPU cool down

train_and_validate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100,
    checkpoint_path=checkpoint_path,
    best_model_path=best_model_path,
    writer=writer,
    device=device,
    scoring_metric="balanced_accuracy",  # Choose between "accuracy" or "balanced_accuracy"
)
# Function to generate test predictions
def generate_test_predictions(test_loader, model, device, output_csv_path):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # Save predictions to CSV
    test_ids = np.arange(1, len(predictions) + 1)
    predictions_df = pd.DataFrame({'ID': test_ids, 'Class': predictions})
    predictions_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

# Generate test predictions
model.load_state_dict(torch.load(best_model_path))
output_csv_path = "test_predictions_deit_small.csv"
generate_test_predictions(test_loader, model, device, output_csv_path)

# Close TensorBoard writer
writer.close()
