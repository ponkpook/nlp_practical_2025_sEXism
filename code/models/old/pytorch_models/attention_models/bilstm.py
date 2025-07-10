import torch
import torch.nn as nn
import torch.nn.functional as F
from .bahdanau import BahdanauAttention

# Re-use the BahdanauAttention class defined above

class BiLSTMAttentionClassifier(nn.Module):
    """
    A Bidirectional LSTM model for text classification with an Attention mechanism.
    Supports learning/pretrained embeddings.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 dropout_prob=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        """
        Initializes the BiLSTMAttentionClassifier model.
        """
        super().__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=True, 
                            dropout=dropout_prob if 1 > 0 else 0)
        
        # Attention layer operates on the output of the Bi-LSTM
        # The hidden_dim for attention is hidden_dim * 2 because it's bidirectional
        self.attention = BahdanauAttention(hidden_dim * 2) 
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text):
        # text: (batch_size, sequence_length)

        embedded = self.embedding(text)
        # embedded: (batch_size, sequence_length, embedding_dim)

        output, (hidden, cell) = self.lstm(embedded)
        # output: (batch_size, sequence_length, num_directions * hidden_size)
        
        # Apply attention to the LSTM output sequence
        context_vector, attn_weights = self.attention(output)
        # context_vector: (batch_size, num_directions * hidden_size)
        
        cat = self.dropout(context_vector)
        # cat: (batch_size, num_directions * hidden_size)

        logits = self.fc(cat)
        # logits: (batch_size, num_classes)

        return logits