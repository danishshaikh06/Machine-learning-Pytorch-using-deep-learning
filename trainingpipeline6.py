# 1) design model (input size,output size,forward phase)
# 2) construct loss and optimizer
# 3) training loop
#    -forward phase:compute prediction
#    -backward phase=compute gradient
#    -update weights

import torch
import torch.nn as nn  # Import PyTorch's neural network module

# 1️⃣ Prepare the dataset (Inputs & Targets)
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # Features (4 samples, 1 feature each)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)  # Targets (True values)
X_test = torch.tensor([5], dtype=torch.float32)  # A test input for prediction after training

# Get the number of samples and features
no_samples, no_feature = X.shape  
print(no_samples, no_feature)  # Output: (4,1) → 4 training examples, 1 feature each

# 2️⃣ Define the Model (Linear Regression)
input_size = no_feature   # 1 input feature
output_size = no_feature  # 1 output

# PyTorch's Linear model: y = w * x + b
#model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, outut_dim):
        super(LinearRegression,self).__init__()
        self.layer1=nn.Linear(input_dim,outut_dim)

    def forward(self,x):
        return self.layer1(x)
    

model = LinearRegression(input_size, output_size)
# Check the model's initial prediction before training
print(f'Prediction before training: {model(X_test).item():.3f}')

# 3️⃣ Training Parameters
learning_rate = 0.01  # Small step size for gradient descent
epochs = 100          # Number of training iterations

# Define Loss Function (Mean Squared Error - MSE)
loss_fn = nn.MSELoss()

# Define Optimizer (Stochastic Gradient Descent - SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # model.parameters() = [weights, bias]

# 4️⃣ Training Loop
for epoch in range(epochs):
    # Forward pass: Compute prediction
    y_prediction = model(X)  

    # Compute loss (How far the prediction is from the actual values)
    l = loss_fn(y_prediction, Y)  

    # Backward pass: Compute gradients (dl/dw and dl/db)
    l.backward()  

    # Update model weights using gradients
    optimizer.step()  

    # Reset gradients before the next iteration (Otherwise, they accumulate)
    optimizer.zero_grad()  

    # Print loss and weight every 10 epochs
    if epoch % 10 == 0:
        [w, b] = model.parameters()  # Extract model's learned weight & bias
        print(f'epoch {epoch+1} loss={l:.8f} w={w[0][0].item():.3f}')

# 5️⃣ Model Prediction After Training
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
