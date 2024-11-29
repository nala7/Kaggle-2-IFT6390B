from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau



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
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translate by 10% of image size
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))     # Normalize to [-1, 1]
])


# Dataset preparation
dataset = CustomDataset(train_data['images'], train_data['labels'], transform=None)
test_dataset = CustomDataset(test_data['images'], transform=None)

labels = train_data['labels']
train_indices, val_indices = train_test_split(
    range(len(labels)), test_size=0.2, stratify=labels, random_state=42, shuffle=True
)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# DataLoaders
batch_size = 64


# Compute class weights
class_counts = np.bincount(train_data['labels'])  # Count samples per class
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)

# Map subset indices to their corresponding labels in the full dataset
subset_indices = train_dataset.indices  # Indices in the original dataset
subset_labels = [train_data['labels'][i] for i in subset_indices]  # Labels for the subset

# Compute sample weights for the subset
sample_weights = [class_weights[label] for label in subset_labels]

# Create WeightedRandomSampler
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# DataLoader with sampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)# sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(unique_labels))
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
checkpoint_path = f"checkpoint_resnet18_{fc_str}.pth"
best_model_path = f"best_model_resnet18_{fc_str}.pth"

# TensorBoard writer
writer = SummaryWriter(log_dir="runs/resnet18_experiment")

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
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs, checkpoint_path, best_model_path, predictions_file='train_val_predictions.csv',writer=writer, device=device,start_epoch=0):
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
    best_val_accuracy = 0.0
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True)
    # For saving predictions
    all_train_predictions = []
    all_val_predictions = []


    # load best model and its accuracy
    if os.path.exists(best_model_path):
        val_labels = []
        val_preds = []
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
        best_score = balanced_accuracy_score(val_labels, val_preds)
        print(f"Best model loaded from {best_model_path} with best score: {best_score:.4f}")

    else:
        best_score = 0.0
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
        train_balanced_acc = balanced_accuracy_score(train_labels, train_preds)

        # Log train metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
        writer.add_scalar("Balanced Accuracy/Train", train_balanced_acc, epoch)

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
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds)
        this_score = val_balanced_acc

        # Log validation metrics to TensorBoard
        writer.add_scalar("Loss/Validation", val_loss / len(val_loader), epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
        writer.add_scalar("Balanced Accuracy/Validation", val_balanced_acc, epoch)

        for label, pred in zip(val_labels, val_preds):
            all_val_predictions.append((label, pred, "val"))

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Train Balanced Acc: {train_balanced_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")

        scheduler.step(val_balanced_acc)

        # Save the best model
        if this_score > best_score:
            best_score = this_score
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with Val Accuracy: {this_score:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    # Save train and validation predictions
    predictions_df = pd.DataFrame(
        all_train_predictions + all_val_predictions,
        columns=["True Label", "Predicted Label", "Set"]
    )
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Train and validation predictions saved to '{predictions_file}'")

import pandas as pd
import torch
from torch.utils.data import DataLoader

def generate_predictions(data, labels, subset, model, save_path, batch_size=64):
    """
    Generates predictions for the given data and saves them in a CSV file.

    Args:
        data (array-like): Input data (e.g., images) for predictions.
        labels (array-like or None): Ground truth labels. If None, no "True Label" column is added.
        subset (list or None): Subset list (e.g., "train" or "val") for indicating the data set. If None, no "Set" column is added.
        model (torch.nn.Module): Trained model to generate predictions.
        save_path (str): Path to save the predictions CSV file.
        batch_size (int): Batch size for the DataLoader (default: 64).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Convert data into a DataLoader
    dataset = CustomDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Collect predictions
    predictions = []
    true_labels = [] if labels is not None else None
    subsets = [] if subset is not None else None

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images = batch[0].to(device)  # Input images
            # Ensure images have 4D shape: (batch_size, channels, height, width)
            if images.dim() == 3:
                images = images.unsqueeze(0)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class
            predictions.extend(predicted.cpu().numpy())

            # Add true labels if provided
            if labels is not None:
                true_labels.extend(batch[1].cpu().numpy())
            
            # Add subset labels if provided
            if subset is not None:
                subsets.extend([subset[i] for i in range(len(batch[1]))])

    # Create DataFrame
    predictions_df = pd.DataFrame({'Predicted Label': predictions})
    if labels is not None:
        predictions_df['True Label'] = true_labels
    if subset is not None:
        predictions_df['Set'] = subsets

    # Save predictions
    predictions_df.to_csv(save_path, index=False)
    print(f"Predictions saved to '{save_path}'")
    return predictions_df

# Generate metrics and confusion matrices
def generate_metrics(predictions_file):
    predictions_df = pd.read_csv(predictions_file)
    for set_type in predictions_df["Set"].unique():
        subset = predictions_df if set_type == "both" else predictions_df[predictions_df["Set"] == set_type]
        true_labels = subset["True Label"]
        pred_labels = subset["Predicted Label"]

        print(f"\nMetrics for {set_type.capitalize()} Set:")
        print(confusion_matrix(true_labels, pred_labels))
        print(classification_report(true_labels, pred_labels, target_names=[f"Class {i}" for i in range(len(unique_labels))]))
    

# Train and validate the model
predictions_file = f"train_val_predictions_resnet18_{fc_str}.csv"
train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=100, checkpoint_path=checkpoint_path, best_model_path=best_model_path, predictions_file=predictions_file)

# get best model
model = models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, len(unique_labels))

model.load_state_dict(torch.load(best_model_path))

# Generate predictions for the test data
test_predictions_file = f"test_predictions_resnet18_{fc_str}.csv"
test_predictions = generate_predictions(test_data['images'], None, None, model, save_path=test_predictions_file)

# Generate predictions for the complete train data
train_val_predictions_file = f"train_complete_predictions_resnet18_{fc_str}.csv"
train_val_predictions = generate_predictions(train_data['images'], train_data['labels'], ["train"]*len(train_data['labels']), model, save_path=train_val_predictions_file)

# Generate predictions for split train and validation data
train_predictions_file = f"train_split_predictions_resnet18_{fc_str}.csv"
this_data = np.array(train_data['images'])[train_indices].tolist()
this_labels = np.array(train_data['labels'])[train_indices].tolist()
this_set = ["train"]*len(np.array(train_data['labels'])[train_indices].tolist())
train_predictions = generate_predictions(this_data, this_labels , this_set , model, save_path=train_predictions_file)

val_predictions_file = f"val_split_predictions_resnet18_{fc_str}.csv"
this_data = np.array(train_data['images'])[val_indices].tolist()
this_labels = np.array(train_data['labels'])[val_indices].tolist()
this_set = ["val"]*len(np.array(train_data['labels'])[val_indices].tolist())
val_predictions = generate_predictions(this_data, this_labels , this_set , model, save_path=val_predictions_file)

# Generate metrics
for f in [ train_val_predictions_file, train_predictions_file, val_predictions_file]:
    print(f"\nMetrics for {f}:")
    generate_metrics(f)

# Close TensorBoard writer
writer.close()
