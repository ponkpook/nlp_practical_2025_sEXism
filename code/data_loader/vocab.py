from collections import Counter
import numpy as np
import torch
import re

def build_vocab(texts, embedding=None, min_freq=2, embedding_dim=200):
    all_words = []
    for text in texts:
        words = re.findall(r'\b\w+\b', str(text).lower())
        all_words.extend(words)

    word_counts = Counter(all_words)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    embeddings = [
        np.zeros(embedding_dim, dtype=np.float32),  # <PAD>
        np.random.normal(scale=0.6, size=embedding_dim).astype(np.float32)  # <UNK>
    ]

    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
            if embedding and word in embedding.key_to_index:
                embeddings.append(embedding[word].astype(np.float32))
            else:
                embeddings.append(np.random.normal(scale=0.6, size=embedding_dim).astype(np.float32))

    #embedding_matrix = torch.tensor(np.stack(embeddings), dtype=torch.float32)
    embedding_matrix = torch.tensor(np.array(embeddings), dtype=torch.float32)
    return vocab, embedding_matrix