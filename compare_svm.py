import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from svm2 import MulticlassSVM


def stratified_sampling(X, y, sample_percentage, random_seed=42):
    np.random.seed(random_seed)
    classes, class_counts = np.unique(y, return_counts=True)

    indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        # Calculate the number of samples for the current class
        cls_sample_count = max(1, int(len(cls_indices) * sample_percentage))
        # Sample indices for the current class
        sampled_indices = np.random.choice(cls_indices, size=cls_sample_count, replace=False)
        indices.extend(sampled_indices)

    # Shuffle the final indices
    indices = np.array(indices, dtype=int)  # Ensure indices are integers
    np.random.shuffle(indices)

    return X[indices], y[indices]


def test_hyperparameters(X_train, y_train, X_val, y_val, hyperparams):
    results = []
    for params in hyperparams:
        print(f"Testing with hyperparameters: {params}")

        # Perform sampling if sampling percentage is specified
        if 'sample_percentage' in params:
            X_sampled, y_sampled = stratified_sampling(X_train, y_train, params['sample_percentage'])
        else:
            X_sampled, y_sampled = X_train, y_train

        # Initialize the model with given hyperparameters
        model = MulticlassSVM(kernel=params['kernel'], C=params['C'], tol=params['tol'], max_iter=params['max_iter'])
        model.fit(X_sampled, y_sampled)

        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        # Store results
        results.append({'params': params, 'accuracy': accuracy})
    return results


def plot_results(results):
    hyperparams = [res['params'] for res in results]
    accuracy_score = [res['accuracy'] for res in results]

    x_labels = [
        f"{params['kernel'].upper()} (C={params['C']})" for params in hyperparams
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, accuracy_score, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title('SVM Hyperparameter Performance', fontsize=16)
    plt.xlabel('Hyperparameter Configuration', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i, accuracy in enumerate(accuracy_score):
        plt.text(i, accuracy, f'{accuracy}%',
                 ha='center', va='bottom', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



# Load your dataset
print("Loading data...")
with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

X = np.array(train_data['images']).reshape(len(train_data['images']), -1)
y = np.array(train_data['labels'])

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define hyperparameters to test
# hyperparams = [
#     {'kernel': 'linear', 'C': 0.1, 'tol': 1e-4, 'max_iter': 1000, 'sample_percentage': 0.3},
#     {'kernel': 'linear', 'C': 1.0, 'tol': 1e-4, 'max_iter': 1000, 'sample_percentage': 0.3},
#     {'kernel': 'rbf', 'C': 1.0, 'tol': 1e-3, 'max_iter': 2000, 'sample_percentage': 0.3},
#     {'kernel': 'rbf', 'C': 10.0, 'tol': 1e-4, 'max_iter': 2000, 'sample_percentage': 0.3}
# ]
#
# # Test the hyperparameters
# print("Testing hyperparameters...")
# results = test_hyperparameters(X_train, y_train, X_val, y_val, hyperparams)
#
# # Plot the results
# print("Plotting results...")
# plot_results(results)

def max_iters_plot_results(results):
    hyperparams = [res['params'] for res in results]
    accuracy_score = [res['accuracy'] for res in results]

    x_labels = [
        f"{params['kernel'].upper()} (Max Iters={params['max_iter']})" for params in hyperparams
    ]
    plt.figure(figsize=(10, 6))
    plt.plot(x_labels, accuracy_score, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.title('SVM Hyperparameter Performance', fontsize=16)
    plt.xlabel('Hyperparameter Configuration', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    for i, accuracy in enumerate(accuracy_score):
        plt.text(i, accuracy, f'{accuracy}%',
                 ha='center', va='bottom', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


hyperparams_max_iters_2 = [
    {'kernel': 'rbf', 'C': 1.0, 'tol': 1e-3, 'max_iter': 1000, 'sample_percentage': 0.3},
    {'kernel': 'rbf', 'C': 1.0, 'tol': 1e-3, 'max_iter': 1500, 'sample_percentage': 0.3},
    {'kernel': 'rbf', 'C': 1.0, 'tol': 1e-3, 'max_iter': 2000, 'sample_percentage': 0.3}
]

# Test the hyperparameters
print("Testing hyperparameters...")
results2 = test_hyperparameters(X_train, y_train, X_val, y_val, hyperparams_max_iters_2)

# Plot the results
print("Plotting results...")
max_iters_plot_results(results2)