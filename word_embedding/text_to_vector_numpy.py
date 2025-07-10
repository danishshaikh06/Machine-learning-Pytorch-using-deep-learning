import numpy as np  # Import NumPy for efficient array operations

# Input string
text = "hello"  # Define the input string to be converted

# Create vocabulary
chars = sorted(list(set(text)))  # Get unique characters, sort them for consistency
char_to_idx = {ch: i for i, ch in enumerate(chars)}  # Map each character to an index
vocab_size = len(chars)  # Calculate the size of the vocabulary

# Convert string to one-hot vectors
def string_to_one_hot(text, char_to_idx, vocab_size):  # Define a function to create one-hot vectors
    one_hot = np.zeros((len(text), vocab_size))  # Initialize a zero matrix for one-hot vectors
    for i, char in enumerate(text):  # Iterate over each character with its index
        one_hot[i, char_to_idx[char]] = 1  # Set the corresponding index to 1 for the character
    return one_hot  # Return the one-hot encoded matrix

one_hot_vectors = string_to_one_hot(text, char_to_idx, vocab_size)  # Call the function to get vectors
print(one_hot_vectors)  # Print the resulting one-hot vectors