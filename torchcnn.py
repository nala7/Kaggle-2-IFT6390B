import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
def load_data(path_to_data):
    with open(path_to_data, 'rb') as f:
        return pickle.load(f)

test_path= 'data/test_data.pkl'
train_path= 'data/train_data.pkl'

test_data= load_data(test_path)
train_data= load_data(train_path)

unique_labels, counts = np.unique(train_data['labels'], return_counts=True)
label_counts = dict(zip(unique_labels, counts))

idx_dict = {}

for label in unique_labels:
    idx_dict[label] = np.where(train_data['labels'] == label)[0].tolist()



# Dataset Preparation (assuming train_data is a dictionary with 'images' and 'labels')
# CustomDataset should accept a `transform` parameter for augmentation
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
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
class CustomDatasetOLD(Dataset):
    def __init__(self, images, labels):
        self.images = torch.tensor(np.array(images)).float().unsqueeze(1) / 255.0  # Normalize to [0, 1] and add channel dim
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


import torchvision.transforms as transforms

# Define data augmentation transformations
train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=20),  # Random rotation by ±20 degrees
    transforms.RandomHorizontalFlip(),      # Random horizontal flip
    transforms.ToTensor(),                  # Convert to Tensor
    transforms.Normalize((0.5,), (0.5,))    # Normalize to [-1, 1] range
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Creating PyTorch Dataset
dataset = CustomDataset(train_data['images'], train_data['labels'], transform=train_transforms)

# Extract images and labels
images = train_data['images']
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


# Initialize TensorBoard writer
writer = SummaryWriter('runs/classification_experiment_with_balanced_accuracy')

# Defining the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x14x14

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64x7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Instantiate the model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




# Function to save a checkpoint
def save_checkpoint(model, optimizer, epoch, best_val_bal_acc, checkpoint_path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_bal_acc': best_val_bal_acc,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# Function to load a checkpoint
def load_checkpoint(model, optimizer, checkpoint_path='checkpoint.pth'):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_bal_acc = checkpoint['best_val_bal_acc']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}")
        return model, optimizer, start_epoch, best_val_bal_acc
    else:
        print("No checkpoint found. Starting training from scratch.")
        return model, optimizer, 0, 0.0

# Updated Training Function with Checkpoint Handling
def train_and_evaluate_with_checkpoint(model, train_loader, val_loader, criterion, optimizer, epochs=10, checkpoint_path='checkpoint.pth',writer=writer,device=device, best_path='best_model.pth'):
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    train_bal_acc_history, val_bal_acc_history = [], []

    # Load from checkpoint if available
    model, optimizer, start_epoch, best_val_bal_acc = load_checkpoint(model, optimizer, checkpoint_path)

    # Load the best weights (from 'best_model.pth') at the start of training
    if os.path.exists(best_path):
        model = load_model(model, best_path)  # Use the function to load best weights
        best_model_wts = copy.deepcopy(model.state_dict())  # Save best weights into memory
        print("Best model weights loaded before starting training.")
    else:
        best_model_wts = copy.deepcopy(model.state_dict())  # If no best weights file exists
        print("No best model weights found. Starting fresh.")
    for epoch in range(start_epoch, epochs):
        # Training Phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_train_labels = []
        all_train_preds = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect labels and predictions for balanced accuracy
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predicted.cpu().numpy())

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_bal_acc = balanced_accuracy_score(all_train_labels, all_train_preds)

        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        train_bal_acc_history.append(train_bal_acc)

        # Validation Phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect labels and predictions for balanced accuracy
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        val_bal_acc = balanced_accuracy_score(all_val_labels, all_val_preds)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        val_bal_acc_history.append(val_bal_acc)


        # Save best model weights based on validation balanced accuracy
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_path)  # Save best weights to a file
            print(f"Best model updated and saved at epoch {epoch + 1} with Val Bal Acc: {val_bal_acc:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, best_val_bal_acc, checkpoint_path)

        # Log metrics
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Balanced Accuracy/Train', train_bal_acc, epoch)
        writer.add_scalar('Balanced Accuracy/Validation', val_bal_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Bal Acc: {train_bal_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Bal Acc: {val_bal_acc:.4f}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'train_bal_acc': train_bal_acc_history,
        'val_bal_acc': val_bal_acc_history,
    }

# Training
checkpoint_path = 'checkpoint-torchcnn.pth'
best_path = 'best_model-torchcnn.pth'
epochs = 1000
results = train_and_evaluate_with_checkpoint(model, train_loader, val_loader, criterion, optimizer, epochs, checkpoint_path,best_path=best_path,writer=writer,device=device)
# Close TensorBoard writer
writer.close()

# Visualize Metrics
plt.figure(figsize=(12, 10))

epochs=len(results['train_loss'])
# Loss
plt.subplot(3, 1, 1)
plt.plot(range(1, epochs+1), results['train_loss'], label='Train Loss')
plt.plot(range(1, epochs+1), results['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

# Accuracy
plt.subplot(3, 1, 2)
plt.plot(range(1, epochs+1), results['train_acc'], label='Train Accuracy')
plt.plot(range(1, epochs+1), results['val_acc'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

# Balanced Accuracy
plt.subplot(3, 1, 3)
plt.plot(range(1, epochs+1), results['train_bal_acc'], label='Train Balanced Accuracy')
plt.plot(range(1, epochs+1), results['val_bal_acc'], label='Validation Balanced Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Balanced Accuracy')
plt.title('Balanced Accuracy vs Epochs')
plt.legend()

plt.tight_layout()
plt.show()


import torch
import pandas as pd
import numpy as np

# Function to load a saved model
def load_model(model, checkpoint_path='best_model.pth'):
    if not torch.cuda.is_available():
        # If using CPU, map the weights appropriately
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {checkpoint_path}")
    return model

# Assuming `SimpleCNN` is the model class and test_data is already loaded
model = SimpleCNN(num_classes=4).to(device)
model = load_model(model, best_path)  # Load the best saved model

# Convert test images to a PyTorch Tensor and normalize
test_images = torch.tensor(np.array(test_data['images'])).float().unsqueeze(1) / 255.0  # Add channel dim

# Create a DataLoader for the test data
batch_size = 64
test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=False)

# Make predictions
predictions = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        outputs = model(batch)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        predictions.extend(predicted.cpu().numpy())

# Create a DataFrame for the predictions
ids = np.arange(1, len(predictions) + 1)  # IDs starting from 1
output_df = pd.DataFrame({'ID': ids, 'Class': predictions})

# Save the DataFrame to a CSV file
output_csv_path = 'test_predictions_torchcnn.csv'
output_df.to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
