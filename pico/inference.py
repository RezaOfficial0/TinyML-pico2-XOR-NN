import math
from model_params import W1, b1, W2, b2


def sigmoid(x):
    # The trained XOR model produces values in a safe range for exp().
    return 1.0 / (1.0 + math.exp(-x))


def forward(x):
    # Hidden layer: 2 inputs -> 4 tanh neurons.
    hidden = []

    for j in range(len(W1[0])):
        z = b1[0][j]

        for i in range(len(x)):
            z += x[i] * W1[i][j]

        hidden.append(math.tanh(z))

    # Output layer: 4 hidden values -> 1 sigmoid output.
    output = b2[0][0]

    for j in range(len(hidden)):
        output += hidden[j] * W2[j][0]

    return sigmoid(output)


def predict(x):
    if len(x) != 2:
        raise ValueError("Expected exactly two input values.")

    if x[0] not in (0, 1) or x[1] not in (0, 1):
        raise ValueError("Inputs must be binary: 0 or 1.")

    probability = forward(x)
    label = 1 if probability >= 0.5 else 0

    return label, probability
