import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import pickle
import os

# Load data function
def load_data(path_to_data):
    with open(path_to_data, 'rb') as f:
        return pickle.load(f)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0  # Normalize to [0, 1]
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

# Model setup
def get_pretrained_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Adjust for single channel
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for num_classes
    return model

# Function to save and load checkpoint
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch+1}")
        return model, optimizer, epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting from scratch.")
        return model, optimizer, 0

# Training function
def train_model(train_dataset, val_dataset, best_model_path, checkpoint_path, num_classes=2, epochs=10, batch_size=64, learning_rate=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = get_pretrained_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if available
    model, optimizer, start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # Best model tracking
    best_acc = 0.0
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with Val Acc: {val_acc:.4f}")

        # Save the checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

    return None

def multi_stage_pipeline_train(train_data, model_paths):
    # Stage 1: Healthy vs Non-Healthy
    stage1_labels = [0 if label == 3 else 1 for label in train_data['labels']]
    stage1_dataset = CustomDataset(train_data['images'], stage1_labels)
    train_indices, val_indices = train_test_split(range(len(stage1_dataset)), test_size=0.2, stratify=stage1_labels)
    train_dataset = Subset(stage1_dataset, train_indices)
    val_dataset = Subset(stage1_dataset, val_indices)

    # Train Stage 1
    train_model(train_dataset, val_dataset, model_paths['best_stage1'], model_paths['checkpoint_stage1'], num_classes=2)

    # Stage 2: Choroidal Neovascularization vs Others
    stage2_labels = [0 if label == 0 else 1 for i, label in enumerate(train_data['labels']) if label != 3]
    stage2_dataset = CustomDataset(
        [train_data['images'][i] for i in range(len(train_data['labels'])) if train_data['labels'][i] != 3],
        stage2_labels
    )
    train_indices, val_indices = train_test_split(range(len(stage2_dataset)), test_size=0.2, stratify=stage2_labels)
    train_dataset = Subset(stage2_dataset, train_indices)
    val_dataset = Subset(stage2_dataset, val_indices)

    # Train Stage 2
    train_model(train_dataset, val_dataset, model_paths['best_stage2'], model_paths['checkpoint_stage2'], num_classes=2)

    # Stage 3: Diabetic Macular Edema vs Drusen
    stage3_labels = [0 if label == 1 else 1 for i, label in enumerate(train_data['labels']) if label in [1, 2]]
    stage3_dataset = CustomDataset(
        [train_data['images'][i] for i in range(len(train_data['labels'])) if train_data['labels'][i] in [1, 2]],
        stage3_labels
    )
    train_indices, val_indices = train_test_split(range(len(stage3_dataset)), test_size=0.2, stratify=stage3_labels)
    train_dataset = Subset(stage3_dataset, train_indices)
    val_dataset = Subset(stage3_dataset, val_indices)

    # Train Stage 3
    train_model(train_dataset, val_dataset, model_paths['best_stage3'], model_paths['checkpoint_stage3'], num_classes=2)
    return None

def multi_stage_pipeline_predict(test_data, model_paths,output_path='test_predictions.csv'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Stage 1: Healthy vs Non-Healthy
    print("Predicting Stage 1: Healthy vs Non-Healthy...")
    stage1_model = get_pretrained_model(2).to(device)
    stage1_model.load_state_dict(torch.load(model_paths['best_stage1']))
    stage1_model.eval()

    test_loader = DataLoader(CustomDataset(test_data['images']), batch_size=64, shuffle=False)
    stage1_preds = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = stage1_model(images)
            _, predicted = torch.max(outputs, 1)
            stage1_preds.extend(predicted.cpu().numpy())

    # Filter Non-Healthy samples for Stage 2
    non_healthy_indices = [i for i, pred in enumerate(stage1_preds) if pred == 1]
    non_healthy_images = [test_data['images'][i] for i in non_healthy_indices]

    # Stage 2: Choroidal Neovascularization vs Others
    print("Predicting Stage 2: Choroidal Neovascularization vs Others...")
    stage2_model = get_pretrained_model(2).to(device)
    stage2_model.load_state_dict(torch.load(model_paths['best_stage2']))
    stage2_model.eval()

    stage2_preds = []
    test_loader_stage2 = DataLoader(CustomDataset(non_healthy_images), batch_size=64, shuffle=False)
    with torch.no_grad():
        for images in test_loader_stage2:
            images = images.to(device)
            outputs = stage2_model(images)
            _, predicted = torch.max(outputs, 1)
            stage2_preds.extend(predicted.cpu().numpy())

    # Filter Others (Non-Choroidal Neovascularization) for Stage 3
    others_indices = [i for i, pred in enumerate(stage2_preds) if pred == 1]
    others_images = [non_healthy_images[i] for i in others_indices]

    # Stage 3: Diabetic Macular Edema vs Drusen
    print("Predicting Stage 3: Diabetic Macular Edema vs Drusen...")
    stage3_model = get_pretrained_model(2).to(device)
    stage3_model.load_state_dict(torch.load(model_paths['best_stage3']))
    stage3_model.eval()

    stage3_preds = []
    test_loader_stage3 = DataLoader(CustomDataset(others_images), batch_size=64, shuffle=False)
    with torch.no_grad():
        for images in test_loader_stage3:
            images = images.to(device)
            outputs = stage3_model(images)
            _, predicted = torch.max(outputs, 1)
            stage3_preds.extend(predicted.cpu().numpy())

    # Combine predictions for all stages
    print("Combining predictions...")
    final_predictions = []
    for i, pred in enumerate(stage1_preds):
        if pred == 0:
            final_predictions.append(3)  # Healthy
        elif stage2_preds[i - len(stage1_preds[:i])] == 0:
            final_predictions.append(0)  # Choroidal Neovascularization
        else:
            final_predictions.append(1 if stage3_preds[i - len(stage2_preds[:i])] == 0 else 2)  # Diabetic Macular Edema or Drusen

    # Save predictions to CSV
    ids = np.arange(1, len(final_predictions) + 1)
    output_df = pd.DataFrame({'ID': ids, 'Class': final_predictions})
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'")


# Paths to data
test_path = 'data/test_data.pkl'
train_path = 'data/train_data.pkl'
test_data = load_data(test_path)
train_data = load_data(train_path)

# Model paths
model_paths = {
    'best_stage1': 'best_stage1_model.pth',
    'checkpoint_stage1': 'checkpoint_stage1.pth',
    'best_stage2': 'best_stage2_model.pth',
    'checkpoint_stage2': 'checkpoint_stage2.pth',
    'best_stage3': 'best_stage3_model.pth',
    'checkpoint_stage3': 'checkpoint_stage3.pth'
}

# Train the multi-stage pipeline
multi_stage_pipeline_train(train_data, model_paths)

# Predict using the multi-stage pipeline
multi_stage_pipeline_predict(test_data, model_paths, output_path='test_predictions_multistage.csv')

