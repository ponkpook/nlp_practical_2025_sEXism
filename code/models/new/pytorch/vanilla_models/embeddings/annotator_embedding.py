import torch
import torch.nn as nn


class AnnotatorEmbedding(nn.Module):
    def __init__(self, attribute_vocabs, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.attributes = list(attribute_vocabs.keys())

        self.vocab_maps = {
            attr: {val: i for i, val in enumerate(vocab)}
            for attr, vocab in attribute_vocabs.items()
        }
        self.embeddings = nn.ModuleDict({
            attr: nn.Embedding(len(vocab), embed_dim)
            for attr, vocab in attribute_vocabs.items()
        })

    def forward(self, annotator_batch_dict):
        batch_size = len(next(iter(annotator_batch_dict.values())))
        num_annotators = len(annotator_batch_dict[self.attributes[0]][0])
        device = next(self.parameters()).device

        total_vec = torch.zeros(batch_size, num_annotators, self.embed_dim, device=device)

        for attr in self.attributes:
            # Convert to index tensor
            vocab_map = self.vocab_maps[attr]
            attr_indices = [
                [vocab_map[val] for val in annotators]
                for annotators in annotator_batch_dict[attr]
            ]
            attr_tensor = torch.tensor(attr_indices, dtype=torch.long, device=device)  # [B, 6]
            attr_embed = self.embeddings[attr](attr_tensor)  # [B, 6, embed_dim]
            total_vec += attr_embed

        # Mean over annotators: [B, embed_dim]
        return total_vec.mean(dim=1)
