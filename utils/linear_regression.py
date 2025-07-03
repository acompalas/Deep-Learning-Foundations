from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from scipy.stats import norm
import streamlit as st

def load_california_housing():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df = df.dropna()  # Drop any NaNs
    return df

def plot_box_plot(df):
    features = df.drop(columns=["MedHouseVal"])

    # Compute range for each feature
    ranges = features.max() - features.min()
    sorted_features = ranges.sort_values(ascending=False).index.tolist()

    # Reorder columns by range
    features_sorted = features[sorted_features]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(features_sorted.values, vert=False, patch_artist=True)
    ax.set_yticks(range(1, len(features_sorted.columns) + 1))
    ax.set_yticklabels(features_sorted.columns)
    ax.set_xlabel("Feature Values")
    ax.set_title("Distribution of Feature Ranges")
    plt.tight_layout()
    return fig

def standardize_df(df):
    df_scaled = df.copy()
    features = df_scaled.drop(columns=["MedHouseVal"])
    df_scaled[features.columns] = (features - features.mean()) / features.std()
    return df_scaled

def minmax_scale_df(df):
    df_scaled = df.copy()
    features = df_scaled.drop(columns=["MedHouseVal"])
    df_scaled[features.columns] = (features - features.min()) / (features.max() - features.min())
    return df_scaled

def plot_density_panel(df):
    features = df.drop(columns=["MedHouseVal"])
    cols = features.columns

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = features[col]
        mean = data.mean()
        std = data.std()
        x_vals = np.linspace(data.min(), data.max(), 200)
        normal_curve = norm.pdf(x_vals, mean, std)

        # Plot KDE
        sns.kdeplot(data=data, ax=axes[i], fill=True, linewidth=1.5)

        # Overlay normal distribution
        axes[i].plot(x_vals, normal_curve, 'r--', label='Normal Dist')
        axes[i].set_title(col)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
        axes[i].legend()

    # Remove unused subplots if any
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit_stepwise(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
            residuals = y_pred - y

            dw = (1 / n_samples) * np.dot(X.T, residuals)
            db = (1 / n_samples) * np.sum(residuals)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(residuals ** 2)
            self.loss_history.append(loss)

            yield i + 1, self.weights.copy(), self.bias, loss

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


def compute_loss_landscape(X, y, theta1_range, theta2_range, bias, steps=50):
    theta1_vals = np.linspace(*theta1_range, steps)
    theta2_vals = np.linspace(*theta2_range, steps)
    loss_vals = np.zeros((steps, steps))

    for i, t1 in enumerate(theta1_vals):
        for j, t2 in enumerate(theta2_vals):
            y_pred = X[:, 0] * t1 + X[:, 1] * t2 + bias
            loss = np.mean((y - y_pred) ** 2)
            loss_vals[j, i] = loss

    return theta1_vals, theta2_vals, loss_vals


def plot_3d_hyperplane_with_mse(X, y, weights, bias, feature_names, theta1_vals, theta2_vals, loss_vals):
    fig = plt.figure(figsize=(14, 6))

    # 3D Regression Hyperplane
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], y, alpha=0.3)

    x_surf, y_surf = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    )
    z_surf = weights[0] * x_surf + weights[1] * y_surf + bias
    ax1.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.set_zlabel("MedHouseVal")
    ax1.set_title("Regression Hyperplane")

    # MSE Contour
    T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(T1, T2, loss_vals, cmap="viridis", edgecolor='none', alpha=0.8)

    current_loss = np.mean((y - (X @ weights + bias)) ** 2)
    ax2.scatter(weights[0], weights[1], current_loss, color='red', s=50, label='Current Weights')

    ax2.set_xlabel("Theta 1")
    ax2.set_ylabel("Theta 2")
    ax2.set_zlabel("MSE")
    ax2.set_title("Loss Landscape")
    ax2.legend()

    return fig

def plot_2d_hyperplane_with_contour(X, y, weights, bias, feature_names,
                                    theta1_vals, theta2_vals, loss_vals,
                                    weight_trajectory=None):
    fig = plt.figure(figsize=(14, 6))

    # --- Left: 3D Regression Hyperplane ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], y, alpha=0.3)

    x_surf, y_surf = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
    )
    z_surf = weights[0] * x_surf + weights[1] * y_surf + bias
    ax1.plot_surface(x_surf, y_surf, z_surf, color='red', alpha=0.5)

    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.set_zlabel("MedHouseVal")
    ax1.set_title("Regression Hyperplane")

    # --- Right: 2D Contour Plot with MSE Landscape ---
    ax2 = fig.add_subplot(122)
    T1, T2 = np.meshgrid(theta1_vals, theta2_vals)
    contour = ax2.contourf(T1, T2, loss_vals, levels=30, cmap="viridis")
    fig.colorbar(contour, ax=ax2, label="MSE")

    # Optional: Draw optimization path
    if weight_trajectory is not None and len(weight_trajectory) > 1:
        traj = np.array(weight_trajectory)
        ax2.plot(traj[:, 0], traj[:, 1], color='yellow', lw=2, label='Trajectory')
        ax2.scatter(traj[-1, 0], traj[-1, 1], color='red', s=50, label='Final Weights')

    else:
        ax2.scatter(weights[0], weights[1], color='red', s=50, label='Current Weights')

    ax2.set_xlabel("Theta 1")
    ax2.set_ylabel("Theta 2")
    ax2.set_title("Loss Contour")
    ax2.legend()

    return fig