import torch  # Import PyTorch for tensor operations and neural network components
import torch.nn as nn  # Import neural network modules from PyTorch

# Input string
text = "hello"  # Define the input string to be converted

# Create vocabulary
chars = sorted(list(set(text)))  # Get unique characters, sort them for consistency
char_to_idx = {ch: i for i, ch in enumerate(chars)}  # Map each character to an index
vocab_size = len(chars)  # Calculate the size of the vocabulary

# Convert string to indices
indices = torch.tensor([char_to_idx[char] for char in text], dtype=torch.long)  # Convert characters to their indices

# Define embedding layer
embedding_dim = 64  # Set the dimensionality of the embedding vectors
embedding = nn.Embedding(vocab_size, embedding_dim)  # Create an embedding layer

# Get embedding vectors
embedding_vectors = embedding(indices)  # Pass indices through the embedding layer
print(embedding_vectors.shape)  # Print the shape of the resulting tensor
print(embedding_vectors)  # Print the embedding vectors