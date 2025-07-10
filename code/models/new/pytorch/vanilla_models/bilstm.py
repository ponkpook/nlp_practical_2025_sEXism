import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=1,
                 dropout_prob=0.5, pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers,
                            bidirectional=True, 
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        pooled = F.max_pool1d(output.permute(0, 2, 1), output.size(1)).squeeze(2)
        cat = self.dropout(pooled)
        logits = self.fc(cat)
        return logits