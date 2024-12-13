import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from cnn import SimpleCNN
from svm2 import MulticlassSVM


def pca(X, n_components):
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute covariance matrix efficiently
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components]

    # Project the data
    X_reduced = np.dot(X_centered, components)

    return X_reduced, components


def stratified_sampling(X, y, sample_size, random_seed=42):
    """
    Performs stratified sampling that handles imbalanced classes.
    If requested samples per class > available samples, takes all available samples.
    """
    np.random.seed(random_seed)
    classes, class_counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # Calculate target samples per class while accounting for class sizes
    total_samples = len(y)
    class_ratios = class_counts / total_samples
    target_samples_per_class = np.floor(sample_size * class_ratios).astype(int)

    # Distribute any remaining samples
    remaining_samples = sample_size - np.sum(target_samples_per_class)
    if remaining_samples > 0:
        # Add remaining samples to largest classes first
        sorted_class_indices = np.argsort(class_counts)[::-1]
        for idx in sorted_class_indices:
            if remaining_samples <= 0:
                break
            available_samples = class_counts[idx] - target_samples_per_class[idx]
            samples_to_add = min(remaining_samples, available_samples)
            target_samples_per_class[idx] += samples_to_add
            remaining_samples -= samples_to_add

    # Collect samples from each class
    indices = []
    for i, cls in enumerate(classes):
        cls_indices = np.where(y == cls)[0]
        # Take minimum of target samples and available samples
        n_samples = min(target_samples_per_class[i], len(cls_indices))
        sampled_indices = np.random.choice(cls_indices, size=n_samples, replace=False)
        indices.extend(sampled_indices)

    # Shuffle the final indices
    indices = np.array(indices)
    np.random.shuffle(indices)

    return X[indices], y[indices]


def process_data(X_train, y_train, X_test, sample_size=None, n_components=None):
    # Normalize features
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_test = (X_test - X_min) / (X_max - X_min + 1e-8)

    if sample_size:
        X_train, y_train = stratified_sampling(X_train, y_train, sample_size)

    if n_components:
        X_train, components = pca(X_train, n_components)
        X_test = np.dot(X_test - np.mean(X_test, axis=0), components)

    return X_train, y_train, X_test


def save(submission_filename, y_test_pred):
    with open(submission_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Class'])
        for i, pred in enumerate(y_test_pred, start=1):
            writer.writerow([i, int(pred)])
    print(f"Submission file created: {submission_filename}")


def callSVM(X_train, y_train, X_test):
    # Train model
    print("Training model...")
    svm = MulticlassSVM(kernel='rbf', C=1.0, tol=1e-4, max_iter=2000)
    svm.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_test_pred = svm.predict(X_test)

    # Calculate training accuracy
    train_accuracy = svm.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Save predictions
    save('submission.csv', y_test_pred)


def callCNN(X_train, y_train, X_test):
    # Initialize and train CNN
    print("Training model...")
    cnn = SimpleCNN(input_shape=(28, 28), num_classes=4, lr=0.01, epochs=5, batch_size=32)
    cnn.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_test_pred = cnn.predict(X_test)

    train_accuracy = cnn.score(X_train, y_train)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Save predictions
    save('cnn_submission.csv', y_test_pred)

def analyze_epochs(X_train, y_train, X_val, y_val, epoch_range=range(1, 16)):
    train_accuracies = []
    val_accuracies = []

    for epochs in epoch_range:
        print(f"Training with {epochs} epochs...")

        # Create and train model
        cnn = SimpleCNN(input_shape=(28, 28), num_classes=4,
                        lr=0.01, epochs=epochs, batch_size=32)
        cnn.fit(X_train, y_train)

        # Compute accuracies
        train_acc = cnn.score(X_train, y_train)
        val_acc = cnn.score(X_val, y_val)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    return train_accuracies, val_accuracies


def plot_epoch_performance(train_accuracies, val_accuracies, epoch_range = range(10, 13)):
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(epoch_range, val_accuracies, marker='o', label='Validation Accuracy')
    plt.title('Accuracy vs. Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # # Plot Loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(epoch_range, train_losses, marker='o', label='Training Loss')
    # plt.plot(epoch_range, val_losses, marker='o', label='Validation Loss')
    # plt.title('Loss vs. Number of Epochs')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

# Load data
print("Loading data...")
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

    # Prepare data
    print("Preparing data...")
    X = np.array(train_data['images']).reshape(len(train_data['images']), 28, 28)
    y = np.array(train_data['labels'])

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Analyze epochs
    train_accuracies, val_accuracies = analyze_epochs(X_train, y_train, X_val, y_val)

    # Plot results
    plot_epoch_performance(train_accuracies, val_accuracies)

    # Print best epoch
    best_epoch = val_accuracies.index(max(val_accuracies)) + 1
    print(f"\nBest number of epochs: {best_epoch}")
    print(f"Best validation accuracy: {max(val_accuracies) * 100:.2f}%")