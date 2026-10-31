import numpy as np


def sigmoid(x):
    # Clip only for numerical stability during training.
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    H = np.tanh(Z1)
    Z2 = H @ W2 + b2
    y_hat = sigmoid(Z2)
    return y_hat, H
