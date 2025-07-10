# 1) design model (input size,output size,forward phase)
# 2) construct loss and optimizer
# 3) training loop
#    -forward phase:compute prediction
#    -backward phase=compute gradient
#    -update weights

# Import necessary libraries
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 0 - Load and prepare the dataset
bc = datasets.load_breast_cancer()  # Load Breast Cancer dataset from sklearn
X, y = bc.data, bc.target           # X = features, y = labels
n_samples, n_features = X.shape     # Get number of samples and features

# Split the dataset into training and testing sets
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (standardization for better training)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
x_test = sc.transform(x_test)

# Convert NumPy arrays to PyTorch tensors and reshape labels
X_train = torch.from_numpy(X_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).view(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).view(-1, 1)

# 1 - Define the logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.layer1 = nn.Linear(n_input_features, 1)  # One layer outputting a single value

    def forward(self, x):
        return torch.sigmoid(self.layer1(x))  # Sigmoid activation for binary classification

# Initialize the model with input size
model = LogisticRegression(n_features)

# 2 - Define loss function and optimizer
learning_rate = 0.01
loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss for binary classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3 - Training loop
epochs = 100
loss_list = []  # To track loss for plotting

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train)
    
    # Compute loss
    loss = loss_fn(y_pred, y_train)
    
    # Backward pass (compute gradients)
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Clear previous gradients
    optimizer.zero_grad()

    # Store loss value
    loss_list.append(loss.item())
    
    # Print loss every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# Plot the loss over epochs
plt.figure(figsize=(8, 4))
plt.plot(range(epochs), loss_list, label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Evaluate the model's accuracy on the test set
with torch.no_grad():
    y_predicted = model(x_test)
    y_predicted_cls = y_predicted.round()  # Threshold predictions at 0.5
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc:.4f}')

    # 4 - Confusion Matrix
    y_pred_numpy = y_predicted_cls.numpy()  # Convert tensors to NumPy arrays
    y_test_numpy = y_test.numpy()
    cm = confusion_matrix(y_test_numpy, y_pred_numpy)  # Compute confusion matrix

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malignant'], 
                yticklabels=['Benign', 'Malignant'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
