import torch
import torch.nn as nn

# Define a custom model with 4 layers
class LinearRegression(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(LinearRegression, self).__init__()
        
        # Define the layers
        self.layer1 = nn.Linear(input_dim, hidden_dim1)  # Input to 1st hidden layer
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)  # 1st hidden layer to 2nd hidden layer
        self.layer3 = nn.Linear(hidden_dim2, hidden_dim3)  # 2nd hidden layer to 3rd hidden layer
        self.layer4 = nn.Linear(hidden_dim3, output_dim)   # 3rd hidden layer to output layer

    def forward(self, x):
        # Forward pass through each layer with ReLU activations
        x = torch.relu(self.layer1(x))  # Apply ReLU after each layer to introduce non-linearity
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)  # No activation after the output layer (linear output)
        return x

# Initialize the model with 1 input feature and 4 hidden layers
input_size = 1  # 1 input feature
hidden_dim1 = 8  # 1st hidden layer with 8 units
hidden_dim2 = 8  # 2nd hidden layer with 8 units
hidden_dim3 = 8  # 3rd hidden layer with 8 units
output_size = 1  # 1 output value

# Create an instance of the model
model = LinearRegression(input_size, hidden_dim1, hidden_dim2, hidden_dim3, output_size)

# Print the model to verify its architecture
print(model)

# Example data
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # Features
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)  # Targets

# Define the loss function and optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD optimizer

# Training loop
epochs = 10
for epoch in range(epochs):
    # Forward pass: Compute predicted Y by passing X to the model
    y_pred = model(X)

    # Compute the loss
    loss = loss_fn(y_pred, Y)

    # Backward pass: Compute gradient of the loss with respect to model parameters
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Zero the gradients for the next step
    optimizer.zero_grad()

    # Print the loss and model parameters every 10 epochs
    if epoch % 2 == 0:
        # Instead of unpacking, you can iterate over the parameters
        for param in model.parameters():
            print(f'Param: {param.data}.item()')  # Print the parameters

# After training, make a prediction
X_test = torch.tensor([[5]], dtype=torch.float32)
y_test = model(X_test)
print(f'Prediction for X=5: {y_test.item():.3f}')
