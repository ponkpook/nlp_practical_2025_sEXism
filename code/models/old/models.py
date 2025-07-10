from registry import register_model
# --- Register CNN_basic ---
from .basic_model import CNNBasicModel
#import CNNBasicModel

# --- 1. Scikit-learn Model Builder ---
@register_model("LogisticRegression")
def build_logistic_regression(params):
    from sklearn.linear_model import LogisticRegression

    # Convert the CfgNode to a regular dictionary.
    param_dict = dict(params)
    print(f"  -> Building sklearn.LogisticRegression with params: {param_dict}")
    
    # Use keyword argument unpacking (**).
    # Sklearn will use its own defaults for any params not in our dict.
    return LogisticRegression(**param_dict)

# --- 2. PyTorch Model Builder ---
@register_model("SimpleCNN")
def build_simple_cnn(params):
    import torch
    
    # Define your PyTorch model class here or import it
    class SimpleCNN(torch.nn.Module):
        def __init__(self, num_classes, vocab_size, embedding_dim=100):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            # ... other layers ...
            self.fc = torch.nn.Linear(embedding_dim, num_classes)
            print(f"Initialized SimpleCNN with num_classes={num_classes}, vocab_size={vocab_size}")

        def forward(self, x):
            # ... forward pass ...
            return self.fc(self.embedding(x))

    param_dict = dict(params)
    print(f"  -> Building PyTorch SimpleCNN with params: {param_dict}")
    
    # Your PyTorch class is instantiated just like the sklearn model.
    return SimpleCNN(**param_dict)

# --- 3. Hugging Face Model Builder ---
@register_model("HuggingFace")
def build_huggingface_model(cfg_node):
    # This builder needs more than just `params`, it needs `model_name_or_path`.
    # So we pass the entire `cfg.model` node to it.
    from transformers import AutoConfig, AutoModelForSequenceClassification

    model_name = cfg_node.model_name_or_path
    param_dict = dict(cfg_node.params)
    
    print(f"  -> Building HuggingFace model '{model_name}' with overrides: {param_dict}")

    # 1. Load the model's default config and apply our overrides from `params`.
    #    This is the correct way to adjust a HF model's configuration.
    hf_config = AutoConfig.from_pretrained(model_name, **param_dict)

    # 2. Load the pretrained model with our custom configuration.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=hf_config)
    
    return model


@register_model("CNN_basic")
def build_cnn_basic(params=None):
    # You can add params if you want to support hyperparameters
    return CNNBasicModel()

# --- Register custom ---
@register_model("custom")
def build_custom_model(cfg_node):
    layers_cfg = cfg_node.layers
    layers = []
    
    import torch.nn as nn
    
    for layer_def in layers_cfg:
        layer_name, layer_params = list(layer_def.items())[0]
        layer_class = getattr(nn, layer_name)
        layers.append(layer_class(**layer_params))
    
    model = nn.Sequential(*layers)
    return model