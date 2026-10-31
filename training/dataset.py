import numpy as np

# XOR truth table: inputs are two binary values, target is one binary value.
X = np.array(
    [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ],
    dtype=np.float32,
)

# Keep y as a column vector with shape (4, 1).
# A shape of (4,) can cause unintended NumPy broadcasting.
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
