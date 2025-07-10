from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def logistic_regression(**params):
    """
    Returns a scikit-learn LogisticRegression initialized with params.
    """
    return LogisticRegression(**params)

def naive_bayes(**params):
    """
    Returns a scikit-learn MultinomialNB initialized with params.
    """
    return MultinomialNB(**params)

def decision_tree(**params):
    """
    Returns a scikit-learn DecisionTreeClassifier initialized with params.
    """
    return DecisionTreeClassifier(**params)

def svm(**params):
    """
    Returns a scikit-learn SVC initialized with params.
    """
    return SVC(**params)
