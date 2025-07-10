import torch
from torch.utils.data import Dataset
import re

class MultiTaskDataset(Dataset):
    def __init__(self, df, vocab, max_length, task_columns: dict, text_column='tweet'):
        """
        task_columns: dict like {
            "binary": "label_bin",
            "multiclass": "label_mc",
            "multilabel": "label_ml"
        }
        """
        self.texts = df[text_column].values
        self.vocab = vocab
        self.max_length = max_length
        self.pad_token_id = self.vocab.get("<PAD>", 0)
        self.task_columns = task_columns

        # Preprocess label tensors
        self.labels = {}
        for task, column in task_columns.items():
            if task == "binary":
                self.labels[task] = df[column].astype(float).values
            elif task == "multiclass":
                self.labels[task] = df[column].astype(int).values
            elif task == "multilabel":
                self.labels[task] = df[column].tolist()
            else:
                raise ValueError(f"Unknown task: {task}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenize_text(text)

        output = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": {}
        }

        for task, values in self.labels.items():
            if task == "binary":
                output["labels"][task] = torch.tensor(values[idx], dtype=torch.float)
            elif task == "multiclass":
                output["labels"][task] = torch.tensor(values[idx], dtype=torch.long)
            elif task == "multilabel":
                output["labels"][task] = torch.tensor(values[idx], dtype=torch.float)

        return output

    def tokenize_text(self, text):
        words = re.findall(r"\\b\\w+\\b", text.lower())
        tokens = [self.vocab.get(word, self.vocab.get("<UNK>", 1)) for word in words]
        tokens = tokens[:self.max_length]
        padding = [self.pad_token_id] * (self.max_length - len(tokens))
        return tokens + padding