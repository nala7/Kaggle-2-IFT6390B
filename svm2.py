import numpy as np
from scipy.spatial.distance import cdist


class LabelBinarizer:
    def __init__(self):
        self.classes_ = None

    def fit_transform(self, y):
        self.classes_ = np.unique(y)
        n_samples = len(y)
        n_classes = len(self.classes_)
        Y = np.zeros((n_samples, n_classes))

        for i, cls in enumerate(self.classes_):
            Y[:, i] = (y == cls)
        return Y

    def inverse_transform(self, Y):
        return self.classes_[np.argmax(Y, axis=1)]


class BinarySVM:
    def __init__(self, kernel, C=1, tol=1e-3, max_iter=1000, class_weight=None):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None

    def _compute_sample_weights(self, y):
        if self.class_weight is None:
            return np.ones(len(y))
        elif self.class_weight == 'balanced':
            # Compute balanced weights
            unique_classes = np.unique(y)
            class_weights = {}
            n_samples = len(y)
            for cls in unique_classes:
                class_weights[cls] = n_samples / (len(unique_classes) * np.sum(y == cls))
        else:
            class_weights = self.class_weight

        # Apply weights to each sample
        sample_weights = np.ones(len(y))
        for cls, weight in class_weights.items():
            sample_weights[y == cls] = weight
        return sample_weights

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.b = 0

        # Compute sample weights
        sample_weights = self._compute_sample_weights(y)

        # Precompute kernel matrix
        K = self.kernel(X, X)

        # Optimize using coordinate descent with vectorized operations
        y = y.astype(float)
        for iteration in range(self.max_iter):
            alpha_prev = self.alpha.copy()

            # Compute predictions and margins for all samples at once
            predictions = np.dot(K, self.alpha * y) + self.b
            margins = y * predictions

            # Compute updates for all samples at once
            mask = margins < 1
            updates = np.zeros(n_samples)
            # Apply sample weights to updates
            updates[mask] = self.C * sample_weights[mask] * (1 - margins[mask])
            self.alpha += updates

            # Project alphas to [0, C * sample_weight]
            self.alpha = np.minimum(np.maximum(self.alpha, 0), self.C * sample_weights)

            # Update bias using weighted average
            weight_sum = np.sum(sample_weights)
            self.b = np.sum(sample_weights * (y - np.dot(K, self.alpha * y))) / weight_sum

            # Check convergence
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                print(f"Converged in {iteration} iterations")
                break

        # Store support vectors
        sv_mask = self.alpha > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.alpha = self.alpha[sv_mask]
        return self

    def predict(self, X):
        if self.support_vectors is None:
            raise ValueError("Model not trained yet!")
        K = self.kernel(X, self.support_vectors)
        return np.sign(np.dot(K, self.alpha * self.support_vector_labels) + self.b)


class MulticlassSVM:
    def __init__(self, kernel='linear', C=1, tol=1e-3, max_iter=500, class_weight=None):
        self.kernel_name = kernel
        self.kernel = self._get_kernel(kernel)
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.label_binarizer = LabelBinarizer()
        self.classifiers = []

    def _get_kernel(self, kernel_name):
        if kernel_name == 'linear':
            return lambda x, y: np.dot(x, y.T)
        elif kernel_name == 'polynomial':
            return lambda x, y, Q=3: (1 + np.dot(x, y.T)) ** Q
        elif kernel_name == 'rbf':
            return lambda x, y, γ=1: np.exp(-γ * cdist(x, y, 'sqeuclidean'))
        else:
            raise ValueError(f"Invalid kernel: {kernel_name}")

    def fit(self, X, y):
        # Convert to binary problems all at once
        y_binary = self.label_binarizer.fit_transform(y)
        n_classes = y_binary.shape[1]

        # Calculate class weights for multiclass case if 'balanced'
        if self.class_weight == 'balanced':
            n_samples = len(y)
            class_weights = {}
            for i, cls in enumerate(self.label_binarizer.classes_):
                class_weights[1] = n_samples / (2 * np.sum(y == cls))  # weight for positive class
                class_weights[-1] = n_samples / (2 * np.sum(y != cls))  # weight for negative class
        else:
            class_weights = self.class_weight

        # Train all binary classifiers
        self.classifiers = []
        for i in range(n_classes):
            print(f"Training classifier for class {i}")
            clf = BinarySVM(
                kernel=self.kernel,
                C=self.C,
                tol=self.tol,
                max_iter=self.max_iter,
                class_weight=class_weights
            )
            clf.fit(X, 2 * y_binary[:, i] - 1)  # Convert to {-1, 1}
            self.classifiers.append(clf)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classifiers)
        predictions = np.zeros((n_samples, n_classes))

        for i, clf in enumerate(self.classifiers):
            predictions[:, i] = clf.predict(X)

        return self.label_binarizer.inverse_transform(predictions > 0)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)