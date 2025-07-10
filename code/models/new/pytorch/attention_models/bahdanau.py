import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BahdanauAttention(nn.Module):
    """
    Bahdanau-style (Additive) Attention mechanism.
    Used to calculate a weighted sum of hidden states from an RNN,
    where weights are learned based on relevance.
    """
    def __init__(self, hidden_dim, dropout_prob=0.0):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim)
        self.Va = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, rnn_output):
        attn_scores = self.Wa(rnn_output)
        
        attn_scores = self.Va(torch.tanh(attn_scores)).squeeze(2) 

        attn_weights = F.softmax(attn_scores, dim=-1)

        context_vector = torch.bmm(attn_weights.unsqueeze(1), rnn_output).squeeze(1)
        
        return context_vector, attn_weights