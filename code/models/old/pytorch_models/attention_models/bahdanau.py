import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Bahdanau-style (Additive) Attention mechanism.
    Used to calculate a weighted sum of hidden states from an RNN,
    where weights are learned based on relevance.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # W_h * h_t (from encoder)
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        # V * tanh(W_h*h_t + W_s*s_t) -> context vector
        self.Va = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_output):
        # rnn_output: (batch_size, sequence_length, hidden_dim * num_directions)
        
        # We want to learn to score each hidden state h_i for attention
        # Equation: score(h_i) = V * tanh(W_h * h_i)
        # Apply Wa to each hidden state
        attn_scores = self.Wa(rnn_output) # (batch_size, sequence_length, hidden_dim)
        
        # Apply Va to reduce to a single score per sequence step
        attn_scores = self.Va(torch.tanh(attn_scores)).squeeze(2) 
        # attn_scores: (batch_size, sequence_length) - unnormalized scores

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=1)
        # attn_weights: (batch_size, sequence_length) - sum to 1 across sequence

        # Compute the weighted sum of rnn_output
        context_vector = torch.bmm(attn_weights.unsqueeze(1), rnn_output).squeeze(1)
        # context_vector: (batch_size, hidden_dim * num_directions)
        # bmm is batch matrix multiplication: (batch, 1, seq_len) * (batch, seq_len, hidden_dim) -> (batch, 1, hidden_dim)
        
        return context_vector, attn_weights # Return weights too for interpretability/visualization if needed