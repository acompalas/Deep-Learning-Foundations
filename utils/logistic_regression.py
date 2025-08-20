import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, log_interval=100):
        y_train = np.array(y_train).flatten()
        y_val = np.array(y_val).flatten() if y_val is not None else None
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(1, self.n_iters + 1):
            linear_pred = np.dot(X_train, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot(X_train.T, (predictions - y_train))
            db = (1 / n_samples) * np.sum(predictions - y_train)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % log_interval == 0:
                # Compute training loss
                train_loss = -np.mean(
                    y_train * np.log(predictions + 1e-15) + (1 - y_train) * np.log(1 - predictions + 1e-15)
                )

                # Compute validation loss if provided
                val_loss = None
                if X_val is not None and y_val is not None:
                    val_pred = sigmoid(np.dot(X_val, self.weights) + self.bias)
                    val_loss = -np.mean(
                        y_val * np.log(val_pred + 1e-15) + (1 - y_val) * np.log(1 - val_pred + 1e-15)
                    )

                yield {
                    "iteration": i,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "weights": self.weights.copy(),
                    "bias": self.bias
                }

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)
    
def plot_losses(train_losses, val_losses, log_interval):
    """
    Plot train and validation losses vs iterations.
    """
    iterations = np.arange(log_interval, log_interval * len(train_losses) + 1, log_interval)
    fig, ax = plt.subplots()
    ax.plot(iterations, train_losses, label="Train Loss")
    ax.plot(iterations, val_losses, label="Val Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Loss vs Iterations")
    ax.legend()
    return fig

