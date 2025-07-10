import logging
from sklearn.metrics import accuracy_score, classification_report


def trainx(loggers, data, model, optimizer, scheduler, cfg):
    """
    Trains a scikit-learn model.

    Args:
        loggers: Logging tools (not usually used in sklearn, but left for consistency).
        data: A tuple (X_train, X_val, y_train, y_val).
        model: An instance of an sklearn estimator.
        optimizer, scheduler: Ignored (for API compatibility).
        cfg: Configuration object.

    Returns:
        Trained model.
    """
    X_train, X_val, y_train, y_val = data

    logging.info(f"Training scikit-learn model: {type(model).__name__}...")
    model.fit(X_train, y_train)

    logging.info("Evaluating on validation data...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    logging.info(f"Validation Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{report}")

    return model

def train(data, model, cfg):
    """
    Trains a scikit-learn model.

    Args:
        loggers: Logging tools (not usually used in sklearn, but left for consistency).
        data: A tuple (X_train, X_val, y_train, y_val).
        model: An instance of an sklearn estimator.
        optimizer, scheduler: Ignored (for API compatibility).
        cfg: Configuration object.

    Returns:
        Trained model.
    """
    X_train, X_val, y_train, y_val = data

    logging.info(f"Training scikit-learn model: {type(model).__name__}...")
    model.fit(X_train, y_train)

    logging.info("Evaluating on validation data...")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    logging.info(f"Validation Accuracy: {acc:.4f}")
    logging.info(f"Classification Report:\n{report}")

    return model
