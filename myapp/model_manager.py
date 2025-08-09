# myapp/model_manager.py
# this script is for model training, loading, and evaluation

import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelBinarizer

MODEL_DIR = "models"


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(X_train, y_train, model_type: str, label_column: str, save=True):
    """
    Train and save a classification model. Assumes X_train is fully encoded.
    """
    ensure_model_dir()
    if model_type == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == "dtc":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "nbc":
        model = GaussianNB()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    filename = f"{MODEL_DIR}/{model_type}_{label_column}.pkl"
    if save:
        joblib.dump(model, filename)
    return model, filename


def load_model(model_type: str, label_column: str):
    """
    Load a previously trained model by type and label column.
    """
    filename = f"{MODEL_DIR}/{model_type}_{label_column}.pkl"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model {filename} not found. Train the model first.")
    model = joblib.load(filename)
    return model


def get_model_info(model_type: str, label_column: str) -> Optional[str]:
    """
    Returns a summary of the model (if available).
    """
    try:
        model = load_model(model_type, label_column)
        return str(model)
    except Exception as e:
        return f"Could not load model: {e}"


def delete_model(model_type: str, label_column: str) -> bool:
    filename = f"{MODEL_DIR}/{model_type.lower()}_{label_column}.pkl"
    if os.path.exists(filename):
        os.remove(filename)
        return True
    return False


def list_models():
    """
    List all saved models.
    """
    ensure_model_dir()
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Compute and return core classification metrics.
    """
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    # Add ROC AUC if possible (binary/multiclass)
    if hasattr(model, "predict_proba"):
        try:
            lb = LabelBinarizer()
            y_test_bin = lb.fit_transform(y_test)
            y_score = model.predict_proba(X_test)
            if y_score.shape[1] == 2:  # binary
                results["roc_auc"] = roc_auc_score(y_test_bin, y_score[:, 1])
            else:  # multiclass
                results["roc_auc"] = roc_auc_score(
                    y_test_bin, y_score, average="weighted", multi_class="ovr"
                )
        except Exception as e:
            results["roc_auc"] = None
    return results


def predict_instance_label(
    model,
    X_encoded: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    instance_idx: int = None,
    test_sample_path: str = None,
    label_column: str = None,
    training_columns: list = None,
):
    """
    Predict the label for a specific instance or from a test sample file.
    Return formatted string with prediction, actual label, and instance details.
    """
    if test_sample_path is not None:
        test_df = pd.read_csv(test_sample_path)
        if test_df.shape[0] != 1:
            return f"Test sample must contain exactly one row. Found {test_df.shape[0]} rows."
        # drop label column if it exists
        if label_column and label_column in test_df.columns:
            test_df = test_df.drop(columns=[label_column])
        # one-hot encode to match training columns
        cat_cols = [
            c
            for c in test_df.columns
            if test_df[c].dtype == "object" or test_df[c].dtype.name == "category"
        ]
        test_encoded = pd.get_dummies(test_df, columns=cat_cols)
        test_encoded = test_encoded.reindex(columns=training_columns, fill_value=0)
        pred_label = model.predict(test_encoded)[0]
        instance_details = "\n".join(
            f"{col}: {val}" for col, val in test_df.iloc[0].items()
        )
        return (
            f"Prediction for the user given test sample in {test_sample_path}:\n"
            f"- Predicted label: {pred_label}\n"
            f"- Input values:\n{instance_details}"
        )

    elif instance_idx is not None:
        # get encoded features
        instance_features = X_encoded.iloc[instance_idx]
        # model prediction (this will work only if the model has .predict method)
        pred_label = model.predict([instance_features])[0]
        # actual label
        actual_label = y.iloc[instance_idx]
        # original (not encoded) instance for readability
        original_instance = df.iloc[instance_idx]
        instance_details = "\n".join(
            f"{col}: {val}" for col, val in original_instance.items()
        )
        return (
            f"Prediction for index {instance_idx}: \n"
            f"- Predicted label: {pred_label}\n"
            f"- Actual label: {actual_label}\n"
            f"- Original instance values:\n{instance_details}"
        )
    else:
        return "Please provide either a valid instance index or test sample path."


def plot_confusion_matrix(
    model, X_test: pd.DataFrame, y_test: pd.Series, class_names=None
) -> plt.Figure:
    """
    Generate and return a confusion matrix plot.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_roc_auc(model, X_test: pd.DataFrame, y_test: pd.Series) -> plt.Figure:
    """
    Plot ROC curve for binary or multiclass classifier.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model does not support probability predictions required for ROC curve."
        )
    y_score = model.predict_proba(X_test)
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    fig, ax = plt.subplots()
    if y_score.shape[1] == 2:  # binary
        fpr, tpr, _ = roc_curve(y_test_bin, y_score[:, 1])
        ax.plot(fpr, tpr, label="ROC curve")
    else:  # multiclass
        for i in range(y_score.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            ax.plot(fpr, tpr, label=f"Class {i}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    return fig
