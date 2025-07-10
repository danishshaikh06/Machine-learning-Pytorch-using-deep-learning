import numpy as np

# f= w * x
# f= 2 * x
# loss = y_pred - y_actual

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([1, 4, 6, 8], dtype=np.float32)

w = 0.0  # Initialize weight

# Model prediction
def forward_phase(x):
    return w * x

# Loss function 
def loss(y_pred, y_actual):
    return ((y_pred - y_actual) ** 2).mean()

# Compute gradient
# MSE = (1/N) * Σ (w*x - y)^2
# dJ/dw = (1/N) * Σ [2 * x * (w*x - y)]
def gradient(x, y_actual, y_pred):
    return np.mean(2 * x * (y_pred - y_actual))

print(f'Prediction before training: f(5)={forward_phase(5):.3f}')

# Training parameters
learning_rate = 0.01
epochs = 20

# Training loop
for epoch in range(epochs):
    y_prediction = forward_phase(X)  # Forward pass

    l = loss(y_prediction, Y)  # Compute loss

    dw = gradient(X, Y, y_prediction)  # Compute gradient

    w -= learning_rate * dw  # Update weight

    if epoch % 2 == 0:
        print(f'epoch {epoch+1} loss={l:.8f} w={w:.3f}')

print(f'Prediction after training: f(5)={forward_phase(5):.3f}')
