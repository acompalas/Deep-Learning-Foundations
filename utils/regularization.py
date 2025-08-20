import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import streamlit as st

class RidgeRegression:
    def __init__(self, learning_rate=0.01, lambda_=0.1, n_iters=10000):
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _mse_with_ridge(self, X, y, weights, bias):
        n = X.shape[0]
        y_pred = X.dot(weights) + bias
        error = y - y_pred
        mse = (1 / n) * np.sum(error ** 2)
        ridge_penalty = self.lambda_ * np.sum(weights ** 2)
        return mse + ridge_penalty

    def fit_stepwise(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(1, self.n_iters + 1):
            y_pred = X.dot(self.weights) + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * (X.T.dot(error) + self.lambda_ * self.weights)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._mse_with_ridge(X, y, self.weights, self.bias)
            self.loss_history.append(loss)

            yield i, self.weights.copy(), self.bias, loss
            
