import torch
import torch.nn as nn
import torch.nn.functional as F
from .bahdanau import BahdanauAttention

class BiGRUAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, 
                 dropout_prob=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.gru = nn.GRU(embedding_dim, 
                          hidden_dim, 
                          num_layers=1, 
                          bidirectional=True, 
                          batch_first=True, 
                          dropout=dropout_prob if 1 > 0 else 0)
        
        self.attention = BahdanauAttention(hidden_dim * 2) 
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.gru(embedded)        
        context_vector, attn_weights = self.attention(output)        
        cat = self.dropout(context_vector)
        logits = self.fc(cat)
        return logits