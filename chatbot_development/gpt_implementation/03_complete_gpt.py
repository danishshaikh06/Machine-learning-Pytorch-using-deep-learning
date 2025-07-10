"""
Complete GPT Implementation for Text Generation
This file demonstrates a full GPT model with tokenization and text generation capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import Counter

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_size
        )
        
        return self.W_o(attention_output)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :].transpose(0, 1)

class GPTBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(GPTBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, max_len=512, dropout=0.1):
        super(GPT, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        
        self.layers = nn.ModuleList([
            GPTBlock(embed_size, heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_causal_mask(self, seq_len):
        """Create causal mask to prevent attention to future tokens"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        
        # Token embeddings
        token_emb = self.token_embedding(x)
        
        # Add positional encoding
        x = self.position_encoding(token_emb)
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len).to(x.device)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.fc_out(x)
        
        return logits

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50):
        """Generate text using the trained model"""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions for the last token
                logits = self(input_ids)
                last_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)
                    last_token_logits = torch.full_like(last_token_logits, -float('inf'))
                    last_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(last_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we've reached max length
                if input_ids.size(1) >= self.max_len:
                    break
        
        return input_ids

class SimpleTokenizer:
    """Simple tokenizer for demonstration purposes"""
    
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        
    def fit(self, texts):
        """Build vocabulary from texts"""
        # Simple word-level tokenization
        words = []
        for text in texts:
            words.extend(re.findall(r'\b\w+\b|[.,!?;]', text.lower()))
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Add special tokens
        self.word_to_id = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.id_to_word = {0: '<pad>', 1: '<unk>', 2: '<sos>', 3: '<eos>'}
        
        # Add words to vocabulary
        for word, count in word_counts.most_common():
            if count >= 2:  # Only include words that appear at least twice
                self.word_to_id[word] = len(self.word_to_id)
                self.id_to_word[len(self.id_to_word)] = word
        
        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text):
        """Convert text to token IDs"""
        words = re.findall(r'\b\w+\b|[.,!?;]', text.lower())
        return [self.word_to_id.get(word, self.word_to_id['<unk>']) for word in words]
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        words = [self.id_to_word.get(id, '<unk>') for id in token_ids]
        return ' '.join(words)

def train_simple_gpt():
    """Demonstrate training a simple GPT model"""
    
    # Sample training data
    texts = [
        "Hello, how are you today?",
        "I am doing well, thank you for asking.",
        "What is your favorite color?",
        "My favorite color is blue.",
        "Tell me a story about a cat.",
        "Once upon a time, there was a cat named Whiskers.",
        "The cat loved to play with yarn.",
        "Every day, the cat would chase mice.",
        "The weather is nice today.",
        "I love sunny days.",
    ]
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Model parameters
    embed_size = 128
    num_layers = 4
    heads = 8
    max_len = 64
    
    # Initialize model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        max_len=max_len
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Generate sample text
    prompt = "hello how"
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    print(f"\\nPrompt: '{prompt}'")
    print(f"Input IDs: {input_ids}")
    
    # Generate text
    generated = model.generate(input_ids, max_length=10, temperature=1.0)
    generated_text = tokenizer.decode(generated[0].tolist())
    
    print(f"Generated: '{generated_text}'")
    
    return model, tokenizer

def demonstrate_chatbot_basics():
    """Demonstrate basic chatbot functionality"""
    print("=== GPT-based Chatbot Demonstration ===")
    
    # Create and demonstrate the model
    model, tokenizer = train_simple_gpt()
    
    print("\\n=== Chatbot Ready ===")
    print("Note: This is a demonstration model without proper training.")
    print("In a real implementation, you would train on large text datasets.")
    
    # Example interactions
    prompts = [
        "what is",
        "how are",
        "tell me"
    ]
    
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        generated = model.generate(input_ids, max_length=8, temperature=0.8)
        response = tokenizer.decode(generated[0].tolist())
        print(f"User: {prompt}")
        print(f"Bot: {response}")
        print()

if __name__ == "__main__":
    demonstrate_chatbot_basics()
