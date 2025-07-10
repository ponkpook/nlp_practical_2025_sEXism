from .sklearn_models import (
    logistic_regression,
    naive_bayes,
    decision_tree,
    svm,
)
#from .pytorch_models import simple_mlp
#from .hf_models import hf_sequence_classifier

# Single registry merging them all:
MODEL_REGISTRY = {
    # scikit-learn
    "logistic_regression": logistic_regression,
    "naive_bayes": naive_bayes,
    "decision_tree": decision_tree,
    "svm": svm,
    # PyTorch
    #"simple_mlp": simple_mlp,
    # HuggingFace
    #"hf_sequence_classifier": hf_sequence_classifier,
}
