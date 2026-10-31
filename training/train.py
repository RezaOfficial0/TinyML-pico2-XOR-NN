import os
import sys
import numpy as np

from dataset import X, y
from model import forward

# Reproducible initialization.
rng = np.random.default_rng(42)

input_size = 2
hidden_size = 4
output_size = 1

# Xavier-style initialization for tanh hidden layer.
W1 = rng.normal(
    0.0, np.sqrt(1.0 / input_size), (input_size, hidden_size)
).astype(np.float32)
b1 = np.zeros((1, hidden_size), dtype=np.float32)

W2 = rng.normal(
    0.0, np.sqrt(1.0 / hidden_size), (hidden_size, output_size)
).astype(np.float32)
b2 = np.zeros((1, output_size), dtype=np.float32)

learning_rate = 0.1
epochs = 20000
epsilon = 1e-7
n = X.shape[0]

for epoch in range(epochs):
    y_hat, H = forward(X, W1, b1, W2, b2)

    # Binary cross-entropy.
    loss = -np.mean(
        y * np.log(y_hat + epsilon)
        + (1.0 - y) * np.log(1.0 - y_hat + epsilon)
    )

    # Backpropagation.
    # For sigmoid + binary cross-entropy, dZ2 = (y_hat - y) / n.
    dZ2 = (y_hat - y) / n
    dW2 = H.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dZ1 = (dZ2 @ W2.T) * (1.0 - H ** 2)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

predictions, _ = forward(X, W1, b1, W2, b2)
labels = (predictions >= 0.5).astype(np.int32)
accuracy = np.mean(labels == y)

print("Final loss:", float(loss))
print("Training accuracy:", f"{accuracy * 100:.1f}%")
print("Predictions:")
for row, probability, label in zip(X, predictions, labels):
    print(row.astype(int).tolist(), "->", int(label[0]),
          "(probability:", round(float(probability[0]), 6), ")")

if accuracy != 1.0:
    print("Training did not reach 100%. Try increasing epochs or changing the seed.")
    sys.exit(1)

os.makedirs("../results", exist_ok=True)
np.savez(
    "../results/xor_model.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2,
)
print("Saved model to results/xor_model.npz")
