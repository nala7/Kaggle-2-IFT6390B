import csv
import numpy as np
import pickle
from svm import MySVM



with open('train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

def save(submission_filename, y_test_pred):
    # Create a CSV with predictions
    with open(submission_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Class'])  # Header row
        for i, pred in enumerate(y_test_pred, start=1):
            # Write the ID (starting from 1) and the predicted class
            writer.writerow([i, int(pred)])
    print(f"Submission file created: {submission_filename}")

# Extract images and labels
X_train = np.array(train_data['images']).reshape(len(train_data['images']), -1)  # Flatten 28x28 images
y_train = np.array(train_data['labels'])

X_test = np.array(test_data['images']).reshape(len(test_data['images']), -1)  # Flatten 28x28 images

# Normalize features (min-max scaling)
X_train = (X_train - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)
X_test = (X_test - X_train.min(axis=0)) / (X_train.max(axis=0) - X_train.min(axis=0) + 1e-8)

# Train the SVM model
svm = MySVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
print("Fitting...")
svm.fit(X_train, y_train)

# Predict the labels for the test set
print("Predicting...")
y_test_pred = svm.predict(X_test)

train_accuracy = svm.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")


# Prepare the submission file
submission_filename = 'submission.csv'
save(submission_filename, y_test_pred)
