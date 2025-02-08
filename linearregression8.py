import torch
import torch.nn as nn
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
##model.parameters==Weights 
## In essence, my code trains a linear model to find the best-fit line that describes the relationship between the input feature and the target variable, enabling predictions on new data points.
# 1) Generate synthetic regression data using sklearn's datasets
# n_samples: number of data points, n_features: number of features (1 for simple linear regression)
# noise: standard deviation of Gaussian noise added to the output, random_state: seed for reproducibility
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

# Convert numpy arrays to PyTorch tensors
# X is the input feature, y is the target output
X = torch.from_numpy(X_numpy.astype(np.float32))  # Convert to float32 for PyTorch compatibility
print(X.shape)
y = torch.from_numpy(y_numpy.astype(np.float32))  # Convert to float32 for PyTorch compatibility
print(y.shape)

# Reshape y to be a column vector (100, 1) instead of a 1D array (100,)
y = y.view(y.shape[0], 1)

# Define a test input for prediction after training
X_test = torch.tensor([[5]], dtype=torch.float32)

# Get the number of samples and features from the input data
no_samples, no_features = X.shape
print(f"Number of samples: {no_samples}, Number of features: {no_features}")

# Define the input and output sizes for the linear model
input_size = no_features  # Number of input features (1 in this case)
output_size = 1  # Number of output features (1 for regression)

# Define the model: a simple linear regression model (y = wx + b)
model = nn.Linear(input_size, output_size)

# 2) Define the loss function and optimizer
# Mean Squared Error (MSE) loss for regression
losses = nn.MSELoss()
learning_rate=0.01
# Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
epochs = 100  # Number of iterations over the entire dataset

for epoch in range(epochs):
    # Forward phase: compute predicted y by passing X to the model
    y_prediction = model(X)

    # Compute the loss: difference between predicted and actual y
    loss = losses(y, y_prediction)

    # Backward phase: compute gradients of the loss with respect to model parameters
    loss.backward()

    # Update model parameters (weights and biases) using the computed gradients
    optimizer.step()

    # Zero the gradients before the next iteration to avoid accumulation
    optimizer.zero_grad()

    # Print the loss and model weights every 10 epochs
    if epoch % 10 == 0:
        print(f'epoch {epoch + 1} loss={loss.item():.3f}')
        for w in model.parameters():
            print(f"weight = {w.item():.4f}")

# 4) Plotting the results
# Detach the predicted values from the computation graph and convert to numpy for plotting
predicted = model(X).detach().numpy()

# Plot the original data points in red
plt.plot(X_numpy, y_numpy, '.', color='r')

# Plot the predicted regression line in blue
plt.plot(X_numpy, predicted, 'b')

# Show the plot
plt.show()

# 5) Make a prediction for a new input value (X = 5)
y_test = model(X_test)
print(f'Prediction for X=5: {y_test.item():.3f}')
