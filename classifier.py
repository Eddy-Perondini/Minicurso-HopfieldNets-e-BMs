import numpy as np


class SoftmaxClassifier:

    def __init__(self, n_features, n_classes=10, seed=0):
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        return self._softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def fit(self, X, y, epochs=20, lr=0.1, batch_size=200, l2=1e-4):
        y = np.asarray(y).astype(int)
        Y = np.eye(self.W.shape[1])[y]
        n = len(X)
        for _ in range(epochs):
            idx = self.rng.permutation(n)
            for i in range(0, n - batch_size + 1, batch_size):
                j = idx[i:i + batch_size]
                err = self.predict_proba(X[j]) - Y[j]
                self.W -= lr * (X[j].T @ err / len(j) + l2 * self.W)
                self.b -= lr * err.mean(axis=0)
        return self