# myapp/method_registry.py

from __future__ import annotations

# This script maintains a single source of truth for all available methods.
# A new method always need to be included here.
# Each entry has:
# - description: internal description
# - user_friendly: a user-friendly description of the method that will be shown to users
# - examples: natural-language examples users might want to type
# - schema_hint: JSON fragment for the agentic prompt examples
# - params_note: additional notes for parameters (index, file path, etc.)

METHODS = {
    "describe_data": {
        "description": "Summarize the dataset and its columns.",
        "user_friendly": "Get a plain-English summary of your dataset and what each column represents.",
        "examples": [
            "Show me the columns in the dataset.",
            "Give me a summary of the dataset.",
        ],
        "schema_hint": '{"method": "describe_data"}',
    },
    "basic_stats": {
        "description": "Report basic statistics (mean, median, min, max, etc) for all or specified numerical columns.",
        "user_friendly": "See quick stats like min/mean/median for one or more columns.",
        "examples": [
            "What is the mean and median of Age?",
            "Show basic stats for Credit amount and Duration.",
        ],
        "schema_hint": '{"method": "basic_stats", "features": ["Age"], "stats": ["mean", "median"]}',
    },
    "show_sample": {
        "description": "Show a table sample (e.g., first 5 rows).",
        "user_friendly": "Preview a few rows of your data to make sure it looks right.",
        "examples": ["Display a sample of the dataset."],
        "schema_hint": '{"method": "show_sample"}',
    },
    "plot_distribution": {
        "description": "Show a histogram/distribution plot for a feature.",
        "user_friendly": "Visualize how a specific column is distributed.",
        "examples": ["Plot a histogram for Credit amount."],
        "schema_hint": '{"method": "plot_distribution", "feature": "Credit amount"}',
        "params_note": "feature (string) required",
    },
    "train_model": {
        "description": 'Train a specific model ("KNN", "DTC", "NBC") with the current label column.',
        "user_friendly": "Train a classifier (e.g., KNN/Decision Tree/Naive Bayes) on your data.",
        "examples": ["Train a decision tree model for this data."],
        "schema_hint": '{"method": "train_model", "model_type": "dtc"}',
        "params_note": "Available model types with the current setting: {knn, dtc, nbc}",
    },
    "evaluate_model": {
        "description": "Show accuracy, precision, recall, F1, ROC-AUC.",
        "user_friendly": "Get a quick report on how well the current model performs.",
        "examples": ["What is the accuracy of the model?"],
        "schema_hint": '{"method": "evaluate_model"}',
    },
    "plot_confusion_matrix": {
        "description": "Plot the confusion matrix.",
        "user_friendly": "See where the model is getting predictions right/wrong by class.",
        "examples": ["Show me the confusion matrix"],
        "schema_hint": '{"method": "plot_confusion_matrix"}',
    },
    "plot_roc_auc": {
        "description": "Plot the ROC curve (if possible).",
        "user_friendly": "Visualize the trade-off between true- and false-positive rates.",
        "examples": ["Plot the ROC curve"],
        "schema_hint": '{"method": "plot_roc_auc"}',
    },
    "load_model": {
        "description": 'Load a trained model by type (e.g., "KNN", "DTC", "NBC").',
        "user_friendly": "Load a previously trained model from disk (Note: You have to train the model first to load it).",
        "examples": ["Load the k-nearest neighbors classifier."],
        "schema_hint": '{"method": "load_model", "model_type": "knn"}',
    },
    "list_models": {
        "description": "List all available trained models on disk.",
        "user_friendly": "Show me which saved models are available to load.",
        "examples": ["Show available trained models."],
        "schema_hint": '{"method": "list_models"}',
    },
    "delete_model": {
        "description": "Delete a trained model by type and label column.",
        "user_friendly": "Remove a saved model file if you no longer need it.",
        "examples": ["Delete the model for KNN with label 'target'."],
        "schema_hint": '{"method": "delete_model", "model_type": "knn", "label_column": "target"}',
        "params_note": "Requires user's input to specify model_type and label_column; the label_column defaults to current label if omitted.",
    },
    "feature_importance": {
        "description": "Show global feature importances (model-based).",
        "user_friendly": "Find which columns matter most overall (if the model supports it).",
        "examples": ["What features are most important for this model?"],
        "schema_hint": '{"method": "feature_importance"}',
    },
    "global_shap_summary": {
        "description": "Show global SHAP feature importances (mean |SHAP| over all instances).",
        "user_friendly": "Get a model-agnostic ranking of important features using SHAP.",
        "examples": ["Give me global feature importance using SHAP."],
        "schema_hint": '{"method": "global_shap_summary"}',
    },
    "predict_label": {
        "description": "Predict the label for a specified row and compare to its actual label.",
        "user_friendly": "Ask for a prediction on a specific row and see if it matches the true label.",
        "examples": [
            "What is the predicted label for index 10?",
            "Predict the label for index 0.",
        ],
        "schema_hint": '{"method": "predict_label", "instance_idx": 10}',
    },
    "shap": {
        "description": "Explain a prediction with SHAP (row-level).",
        "user_friendly": "Explain why the model predicted what it did for a given row using SHAP.",
        "examples": [
            "Explain the prediction for row 2 using SHAP.",
            "Explain the prediction for test_set/sample1.csv with SHAP.",
        ],
        "schema_hint": '{"method": "shap", "instance_idx": 2}',
    },
    "lime": {
        "description": "Explain a prediction with LIME (row-level).",
        "user_friendly": "Explain why the model predicted what it did for a given row using LIME.",
        "examples": [
            "Explain row 3 with LIME.",
            "Explain the prediction for test_set/sample1.csv with LIME.",
        ],
        "schema_hint": '{"method": "lime", "instance_idx": 3}',
    },
    "dice": {
        "description": "Generate counterfactuals (row-level).",
        "user_friendly": "Show how to minimally change inputs to flip the prediction.",
        "examples": [
            "Generate some counterfactuals for instance 7.",
            "Generate 5 counterfactuals for index 37 and show the original data.",
        ],
        "schema_hint": '{"method": "dice", "instance_idx": 7, "total_CFs": 3}',
    },
}


def agentic_methods_block() -> str:
    """
    agentic_methods_block function is to build the "Possible methods" section for the agentic routing prompt
    from this registry, including one JSON schema_hint per method when available.
    """
    lines = ["Possible methods:"]
    for name, meta in METHODS.items():
        desc = meta.get("description", "")
        lines.append(f'- "{name}" — {desc}')
    lines.append("")  # spacer

    # Include examples as JSON hints (1 per method)
    lines.append("Here are some examples:")
    for name, meta in METHODS.items():
        exs = meta.get("examples", [])
        hint = meta.get("schema_hint")
        # Only include one example to keep prompt concise
        if exs:
            lines.append(f'User: "{exs[0]}"')
            if hint:
                lines.append(hint)
    return "\n".join(lines)


def user_help_text() -> str:
    """
    user_help_text function is to build a 'help' or 'what can you do?' guide from the same registry.
    """
    lines = ["Here's what I can help you do:\n"]
    for name, meta in METHODS.items():
        title = meta.get("user_friendly") or meta.get("description") or name
        desc = meta.get("user_friendly") or meta.get("description", "")
        lines.append(f"• {title}")
        if meta.get("examples"):
            examples = "; ".join(meta["examples"])
            lines.append(f"  e.g., {examples}")
        lines.append("")  # spacer
    return "\n".join(lines)
