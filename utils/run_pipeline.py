import os
import yaml
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# --- imports from your modules ---
from preprocessing.normalization import (
    lowercase, remove_punctuation, expand_contractions,
    append_negation_suffix, handle_hashtags,
    normalize_whitespace, remove_accents, replace_numbers
)
from preprocessing.morphology import (
    count_vectorize, tfidf_vectorize,
    word2vec_vectorize, glove_vectorize
)

# Registries
PREPROCESSING_REGISTRY = {
    "lowercase": lowercase,
    "remove_punctuation": remove_punctuation,
    "expand_contractions": expand_contractions,
    "append_negation_suffix": append_negation_suffix,
    "handle_hashtags": handle_hashtags,
    "normalize_whitespace": normalize_whitespace,
    "remove_accents": remove_accents,
    "replace_numbers": replace_numbers,
}
FEATURE_REGISTRY = {
    "count_vectorize": count_vectorize,
    "tfidf_vectorize": tfidf_vectorize,
    "word2vec_vectorize": word2vec_vectorize,
    "glove_vectorize": glove_vectorize,
}

# Simple PyTorch model example
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    def forward(self, x):
        return self.net(x)

MODEL_REGISTRY = {
    "simple_mlp": SimpleMLP,
    # add more models here
}

def run_pipeline(config_path: str):
    # 1) load config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    df = pd.read_csv(cfg["dataset"])
    texts = df["text"]

    # 2) preprocessing
    for name, params in cfg.get("preprocessing", {}).items():
        func = PREPROCESSING_REGISTRY[name]
        p = params or {}        # treat None as {}
        print(f">>> Preprocessing: {name} {p}")
        texts = func(texts, **p)

    # 3) feature extraction (take each in order, store them)
    features = {}
    for name, params in cfg.get("feature_extraction", {}).items():
        func = FEATURE_REGISTRY[name]
        p = params or {}
        print(f">>> Feature extraction: {name} {p}")
        result = func(texts, **p)
        # unpack vectorizer if returned
        if isinstance(result, tuple):
            mat, meta = result
            features[name] = {"matrix": mat, "meta": meta}
        else:
            features[name] = {"matrix": result}

    # 4) model build / train / eval
    mcfg = cfg["model"]
    ModelClass = MODEL_REGISTRY[mcfg["type"]]
    model = ModelClass(**mcfg["params"])
    # assume single feature extractor for input dimension
    X = features[next(iter(features))]["matrix"]
    if hasattr(X, "toarray"): X = X.toarray()
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=mcfg["training"]["batch_size"], shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=mcfg["training"]["lr"])
    loss_fn = nn.CrossEntropyLoss()

    # training loop
    for epoch in range(1, mcfg["training"]["epochs"] + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{mcfg['training']['epochs']} — loss: {total_loss/len(loader):.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        preds = logits.argmax(dim=1).cpu()
    acc = (preds == y).float().mean().item()
    print(f"Final accuracy: {acc:.4f}")

    # save outputs
    torch.save(model.state_dict(), os.path.join(cfg["out_dir"], "model.pt"))
    pd.to_pickle(features, os.path.join(cfg["out_dir"], "features.pkl"))
    with open(os.path.join(cfg["out_dir"], "metrics.txt"), "w") as f:
        f.write(f"accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    run_pipeline("config.yaml")
