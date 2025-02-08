import torch
import numpy as np

# Create a 4x4 tensor with random values between 0 and 1
z = torch.rand(4, 4)

# Create a 2x2 tensor filled with zeros
a = torch.zeros(2, 2)

# Create a 2x2 tensor filled with ones
b = torch.ones(2, 2)

# Attempt to use a non-existent method 'torch_add' on tensor 'a'
# This will raise an AttributeError because 'torch_add' is not a valid method
# a.torch_add(b)

# Correct way to add tensors 'a' and 'b'; result is stored in 'd'
d = torch.add(a, b)  # Alternatively, d = a + b

# Reshape 'z' into a 1D tensor with 16 elements
e = z.view(16)

# Reshape 'z' into a 2D tensor with 2 rows and 8 columns
e = z.view(-1, 8)  # '-1' infers the size of the dimension automatically

print(z)  # Print the original 4x4 tensor 'z'

print(z[1:])  # Print all rows of 'z' starting from the second row

# Attempt to access a slice that results in an empty tensor and call 'item()' on it
# This will raise a ValueError because 'item()' cannot be called on an empty tensor
# print(z[1:1].item())

# Convert a NumPy array of ones to a PyTorch tensor
a = np.ones(5)

# 'requires_grad' is an attribute for PyTorch tensors, not NumPy arrays
a = torch.ones(5, requires_grad=True)

print(a)  # Print the NumPy array 'a'

# Convert the NumPy array 'a' to a PyTorch tensor
b = torch.from_numpy(a)

print(b)  # Print the PyTorch tensor 'b'







