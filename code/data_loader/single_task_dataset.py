import torch
from torch.utils.data import Dataset
import re
import ast

class SingleTaskDataset(Dataset):
    def __init__(self, df, vocab, max_length, label_column, text_column='tweet', task="binary"):
        """
        Args:
            df (pd.DataFrame): DataFrame with at least 'text' and one label column.
            vocab (dict): Token-to-index mapping.
            max_length (int): Maximum sequence length.
            label_column (str): Name of the label column.
            task (str): One of 'binary', 'multiclass', 'multilabel'.
        """
        self.texts = df[text_column].values
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token_id = self.vocab.get("<PAD>", 0)
        self.task = task.lower()

        # Label preparation
        if self.task == "binary":
            self.labels = df[label_column].values.astype(bool)
        elif self.task == "multiclass":
            self.labels = df[label_column].astype(int).values
        elif self.task == "multilabel":
            self.labels = df[label_column].apply(ast.literal_eval).tolist()
        else:
            raise ValueError(f"Unknown task type: {self.task}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenize_text(text)
        label = self.labels[idx]

        item = {"input_ids": torch.tensor(tokens, dtype=torch.long)}

        if self.task == "binary":
            item["labels"] = torch.tensor(label, dtype=torch.long)
        elif self.task == "multiclass":
            item["labels"] = torch.tensor(label, dtype=torch.long)
        elif self.task == "multilabel":
            item["labels"] = torch.tensor(label, dtype=torch.float)

        return item

    def tokenize_text(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        tokens = [self.vocab.get(word, self.vocab.get('<UNK>', 1)) for word in words]

        if len(tokens) < self.max_length:
            tokens.extend([self.pad_token_id] * (self.max_length - len(tokens)))
        else:
            tokens = tokens[:self.max_length]
        return tokens