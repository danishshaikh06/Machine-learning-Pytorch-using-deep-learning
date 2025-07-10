"""
GPT Basic Implementation
This file contains the foundational concepts on how to structure and use GPT for generating text.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTBlock(nn.Module):
    """A basic block for GPT language model"""
    def __init__(self, embed_size, heads):
        super(GPTBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x):
        # Attention and Add & Norm
        x = self.attention(x, x, x) + x
        x = self.norm1(x)
        # Feed Forward and Add & Norm
        x = self.ff(x) + x
        x = self.norm2(x)
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(1000, embed_size)
        self.layers = nn.ModuleList([
            GPTBlock(embed_size, heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        out = self.token_embedding(x) + self.position_embedding(positions)
        for layer in self.layers:
            out = layer(out)
        logits = self.fc_out(out)
        return logits

    def generate(self, x, max_length):
        for _ in range(max_length):
            logits = self(x)
            next_token = torch.argmax(logits, dim=2)[:, -1]
            x = torch.cat((x, next_token.unsqueeze(1)), dim=1)
        return x

def demonstrate_gpt():
    """Setup a simple GPT model and demonstrate text generation"""
    vocab_size = 100
    embed_size = 256
    num_layers = 6
    heads = 8
    max_length = 10
    
    gpt = GPT(vocab_size, embed_size, num_layers, heads)
    
    # Random input simulating sequence of tokenized indices
    x = torch.randint(0, vocab_size, (1, 5))
    
    print("Input token indices:", x)
    output = gpt.generate(x, max_length=10)
    print("Generated token indices:", output)


if __name__ == "__main__":
    print("=== GPT Implementation Demo ===")
    demonstrate_gpt()
