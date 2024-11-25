import numpy as np

class MySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = {}  # Dictionary to store weights for each class
        self.biases = {}   # Dictionary to store biases for each class
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            print(f"Training classifier for class {c}...")
            y_binary = np.where(y == c, 1, -1)  # Treat class `c` as positive, others as negative
            self.weights[c] = np.zeros(X.shape[1])
            self.biases[c] = 0

            w, b = self._train_binary_classifier(X, y_binary)
            self.weights[c] = w
            self.biases[c] = b

    def _train_binary_classifier(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, w) - b) >= 1
                if condition:
                    w -= self.lr * (2 * self.lambda_param * w)
                else:
                    w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, y[idx]))
                    b -= self.lr * y[idx]
        return w, b

    def predict(self, X):
        scores = []
        for c in self.classes:
            decision = np.dot(X, self.weights[c]) - self.biases[c]
            scores.append(decision)
        scores = np.array(scores).T  # Shape: (n_samples, n_classes)
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy
