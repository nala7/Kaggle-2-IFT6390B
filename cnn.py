import numpy as np

# Define the CNN as implemented previously
class SimpleCNN:
    def __init__(self, input_shape, num_classes, lr=0.01, epochs=50, batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.params = self._initialize_weights()

    def _initialize_weights(self):
        np.random.seed(42)
        filter_size = 3
        num_filters = 8
        hidden_units = 64

        # Calculate the size of the flattened layer after pooling
        conv_output_height = self.input_shape[0] - filter_size + 1  # After convolution
        conv_output_width = self.input_shape[1] - filter_size + 1
        pool_output_height = conv_output_height // 2  # After pooling
        pool_output_width = conv_output_width // 2
        flattened_size = num_filters * pool_output_height * pool_output_width

        weights = {
            "conv_filters": np.random.randn(num_filters, filter_size, filter_size) * 0.1,
            "conv_bias": np.zeros(num_filters),
            "fc_weights": np.random.randn(flattened_size, hidden_units) * 0.1,
            "fc_bias": np.zeros(hidden_units),
            "out_weights": np.random.randn(hidden_units, self.num_classes) * 0.1,
            "out_bias": np.zeros(self.num_classes)
        }
        return weights

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _conv_forward(self, X):
        n_samples, height, width = X.shape
        num_filters, filter_size, _ = self.params["conv_filters"].shape
        output_size = height - filter_size + 1
        conv_output = np.zeros((n_samples, num_filters, output_size, output_size))

        for i in range(output_size):
            for j in range(output_size):
                region = X[:, i:i + filter_size, j:j + filter_size]
                conv_output[:, :, i, j] = np.tensordot(region, self.params["conv_filters"], axes=([1, 2], [1, 2])) + self.params["conv_bias"]
        return self._relu(conv_output)

    def _pool_forward(self, X):
        n_samples, num_filters, height, width = X.shape
        output_size = height // 2
        pool_output = np.zeros((n_samples, num_filters, output_size, output_size))

        for i in range(output_size):
            for j in range(output_size):
                region = X[:, :, i * 2:i * 2 + 2, j * 2:j * 2 + 2]
                pool_output[:, :, i, j] = np.max(region, axis=(2, 3))
        return pool_output

    def _flatten(self, X):
        return X.reshape(X.shape[0], -1)

    def _fc_forward(self, X, weights, bias):
        return np.dot(X, weights) + bias

    def fit(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled, y_shuffled = X[shuffled_indices], y[shuffled_indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                conv_output = self._conv_forward(X_batch)
                pool_output = self._pool_forward(conv_output)
                flat_output = self._flatten(pool_output)
                fc_output = self._relu(self._fc_forward(flat_output, self.params["fc_weights"], self.params["fc_bias"]))
                logits = self._fc_forward(fc_output, self.params["out_weights"], self.params["out_bias"])
                probs = self._softmax(logits)

                y_one_hot = np.eye(self.num_classes)[y_batch]
                loss = -np.sum(y_one_hot * np.log(probs + 1e-8)) / self.batch_size

                grad_logits = (probs - y_one_hot) / self.batch_size
                grad_fc_weights_out = np.dot(fc_output.T, grad_logits)
                grad_fc_bias_out = np.sum(grad_logits, axis=0)

                grad_fc_output = np.dot(grad_logits, self.params["out_weights"].T) * self._relu_derivative(fc_output)
                grad_fc_weights = np.dot(flat_output.T, grad_fc_output)
                grad_fc_bias = np.sum(grad_fc_output, axis=0)

                self.params["fc_weights"] -= self.lr * grad_fc_weights
                self.params["fc_bias"] -= self.lr * grad_fc_bias
                self.params["out_weights"] -= self.lr * grad_fc_weights_out
                self.params["out_bias"] -= self.lr * grad_fc_bias_out

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        conv_output = self._conv_forward(X)
        pool_output = self._pool_forward(conv_output)
        flat_output = self._flatten(pool_output)
        fc_output = self._relu(self._fc_forward(flat_output, self.params["fc_weights"], self.params["fc_bias"]))
        logits = self._fc_forward(fc_output, self.params["out_weights"], self.params["out_bias"])
        probs = self._softmax(logits)

        return np.argmax(probs, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
