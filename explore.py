import pickle
import numpy as np
def load_data(path_to_data):
    with open(path_to_data, 'rb') as f:
        return pickle.load(f)

test_path= 'data/test_data.pkl'
train_path= 'data/train_data.pkl'

test_data= load_data(test_path)
train_data= load_data(train_path)

len(train_data['images'])
len(test_data['images'])

len(train_data['labels'])

# save the first 100 of each class

unique_labels, counts = np.unique(train_data['labels'], return_counts=True)
label_counts = dict(zip(unique_labels, counts))

idx_dict = {}

for label in unique_labels:
    idx_dict[label] = np.where(train_data['labels'] == label)[0].tolist()


import matplotlib.pyplot as plt

# Plot class distribution
classes = list(label_counts.keys())
counts = list(label_counts.values())

plt.bar(classes, counts)
plt.xlabel('Class Labels')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Training Data')
plt.show()


import matplotlib.pyplot as plt

# Function to display images
def plot_samples(images, labels, unique_labels, n_samples=5):
    plt.figure(figsize=(10, len(unique_labels) * 2))
    for i, label in enumerate(unique_labels):
        idx = np.random.choice(idx_dict[label], n_samples, replace=False)
        for j, img_idx in enumerate(idx):
            plt.subplot(len(unique_labels), n_samples, i * n_samples + j + 1)
            plt.imshow(images[img_idx], cmap='gray')
            plt.axis('off')
            plt.title(f'Class {label}')
    plt.tight_layout()
    plt.show()

plot_samples(train_data['images'], train_data['labels'], unique_labels)
