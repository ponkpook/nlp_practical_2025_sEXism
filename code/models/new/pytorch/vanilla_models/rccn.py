import torch
import torch.nn as nn
import torch.nn.functional as F

class RCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes,
                 rnn_type='lstm', pretrained_embeddings=None, freeze_embeddings=False, 
                 dropout_prob=0.5, num_filters=100, kernel_size=3):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.conv = nn.Conv1d(in_channels=hidden_dim * 2,
                              out_channels=num_filters,
                              kernel_size=kernel_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)                      # (batch, seq_len, emb_dim)
        rnn_out, _ = self.rnn(embedded)                   # (batch, seq_len, hidden_dim*2)
        rnn_out = rnn_out.permute(0, 2, 1)                # (batch, hidden_dim*2, seq_len)
        conv_out = F.relu(self.conv(rnn_out))             # (batch, num_filters, seq_len-kernel+1)
        pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # (batch, num_filters)
        output = self.fc(self.dropout(pooled))
        return output