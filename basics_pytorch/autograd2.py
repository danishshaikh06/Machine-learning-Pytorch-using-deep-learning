import torch

# Create a tensor 'a' with four elements, all initialized to 1.0.
# Setting requires_grad=True enables PyTorch to track operations on this tensor,
# allowing automatic differentiation to compute gradients during backpropagation.
a = torch.ones(4, requires_grad=True)

# Perform an element-wise addition: each element of 'a' is increased by 2.
# The result is stored in tensor 'z'.
z = a + 2
print(z)  # Output: tensor([3., 3., 3., 3.], grad_fn=<AddBackward0>)

# Compute 'y' by squaring each element of 'z' and then multiplying by 2.
# This results in another tensor where each element is (3^2) * 2 = 18.
y = z * z * 2
print(y)  # Output: tensor([18., 18., 18., 18.], grad_fn=<MulBackward0>)

# Calculate the mean of all elements in 'y' to obtain a scalar value 'b'.
# This scalar represents the average of the elements in 'y'.
b = y.mean()
print(b)  # Output: tensor(18., grad_fn=<MeanBackward0>)

# Perform backpropagation to compute the gradients of 'b' with respect to 'a'.
# Since 'b' is a scalar, this operation calculates the gradient of 'b' w.r.t. each element in 'a'.
b.backward()

# Print the computed gradients stored in 'a.grad'.
# Each element in 'a.grad' represents the partial derivative of 'b' with respect to the corresponding element in 'a'.
print(a.grad)  # Output: tensor([12., 12., 12., 12.])

# Iterate over a range of 3 epochs to simulate training iterations.
for epoch in range(3):
    # Compute 'model_output' by multiplying each element of 'a' by 3 and summing the results.
    # The sum() function ensures that 'model_output' is a scalar, which is necessary for backpropagation.
    model_output = (a * 3).sum()

    # Perform backpropagation to compute the gradients of 'model_output' with respect to 'a'.
    model_output.backward()

    # Print the computed gradients after this iteration.
    print(a.grad)  # Output: tensor([3., 3., 3., 3.])

    # Reset the gradients in 'a' to zero to prevent accumulation in the next iteration.
    # This is crucial because, by default, PyTorch accumulates gradients on subsequent backward passes.
    a.grad.zero_()

