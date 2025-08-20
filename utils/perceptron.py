import streamlit as st
# utils/perceptron.py
# utils/perceptron.py
import numpy as np

def unit_step_func(x):
    return np.where(x >= 0, 1, 0)

def misclassification_rate(y_true, y_pred):
    return 1.0 - (np.sum(y_true == y_pred) / len(y_true))

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iters=100, init_range=(-1.0, 1.0), shuffle=True):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
        self.init_range = init_range
        self.shuffle = shuffle

    def initialize(self, n_features, seed=42):
        """Initialize weights/bias so we can draw an initial boundary before training."""
        rng = np.random.default_rng(seed)
        lo, hi = self.init_range
        self.weights = rng.uniform(lo, hi, size=n_features)
        self.bias = rng.uniform(lo, hi)

    def predict(self, X):
        z = X @ self.weights + self.bias
        return self.activation_func(z)

    def fit(self, X_train, y_train, X_val=None, y_val=None, seed=42):
        """Yield after each epoch: dict(epoch, W, b, train_loss, val_loss, train_acc, val_acc, mistakes)"""
        rng = np.random.default_rng(seed)
        n_samples, n_features = X_train.shape

        # If not already initialized (e.g., from Interactive Demo), initialize now
        if self.weights is None or self.bias is None:
            self.initialize(n_features, seed=seed)

        y_tr = (y_train > 0).astype(int)
        y_va = (y_val > 0).astype(int) if y_val is not None else None

        for epoch in range(1, self.n_iters + 1):
            if self.shuffle:
                idx = rng.permutation(n_samples)
                Xe, ye = X_train[idx], y_tr[idx]
            else:
                Xe, ye = X_train, y_tr

            mistakes = 0
            for xi, yi in zip(Xe, ye):
                yhat = self.activation_func(xi @ self.weights + self.bias)
                update = self.lr * (yi - yhat)
                if update != 0:
                    mistakes += 1
                    self.weights += update * xi
                    self.bias += update

            # Metrics
            yhat_tr = self.predict(X_train)
            train_loss = misclassification_rate(y_tr, yhat_tr)
            train_acc = 1.0 - train_loss

            if X_val is not None:
                yhat_va = self.predict(X_val)
                val_loss = misclassification_rate(y_va, yhat_va)
                val_acc = 1.0 - val_loss
            else:
                val_loss = None
                val_acc = None

            yield {
                "epoch": epoch,
                "W": self.weights.copy(),
                "b": float(self.bias),
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": None if val_acc is None else float(val_acc),
                "mistakes": mistakes,
            }

            if mistakes == 0:
                break
