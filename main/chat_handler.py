# main/chat_handler.py
import os

print("Current working directory:", os.getcwd())
# this is for absolute path to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project root path:", PROJECT_ROOT)
import json
import logging
import re
import numpy as np
from utils.data_utils import *
from myapp.model_manager import *
from myapp.explanation_engine import ExplanationEngine
from sklearn.model_selection import train_test_split

# Convert matplotlib Figure to PIL Image
import io
from PIL import Image
import base64
import matplotlib.pyplot as plt


# --- Helper to extract JSON from LLM output ---
def extract_json_from_llm(llm_response):
    llm_response = llm_response.strip()
    llm_response = re.sub(r"^```.*?\n", "", llm_response)  # remove start ```json or ```
    llm_response = re.sub(r"\n```$", "", llm_response)  # remove end ```
    return llm_response


def load_test_sample(test_sample_path):
    # If only a file name is provided, currently assuming it's in test_set/
    if not os.path.isabs(test_sample_path):
        abs_path = os.path.join(PROJECT_ROOT, test_sample_path)
    else:
        abs_path = test_sample_path
    print("Loading test sample from:", abs_path)
    return pd.read_csv(abs_path)


# --- Main handler ---
def handle_user_input(
    user_input,
    conversation_history,
    explainer,
    llm_client,
    df,  # The full dataframe
    model,  # The current model object (or None)
    X_train,  # Current X_train if available
    X_test,  # Current X_test if available
    y_train,  # Current y_train if available
    y_test,  # Current y_test if available
    label_column,  # Label column string (must be present)
    state,  # The full session state dict for updating as needed
):
    # Prepare the agentic prompt for the LLM
    history_prompt = ""
    for msg in conversation_history:
        if msg["role"] == "user":
            history_prompt += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_prompt += f"Bot: {msg['content']}\n"
    prompt = llm_client.format_agentic_prompt(
        f"{history_prompt}\nUser asked: {user_input}"
    )
    llm_decision = llm_client.query(prompt)
    logging.info("Raw LLM response: %s", llm_decision)

    try:
        decision = json.loads(extract_json_from_llm(llm_decision))
        method = decision["method"].lower()
        response, fig_or_img = (
            None,
            None,
        )  # initialize response and image so when there is no image returned from the functions, this will prevent errors

        # --- Data exploration ---
        if method == "describe_data":
            response = get_dataset_description(df, label_column)
        elif method == "basic_stats":
            columns = decision.get("features") or None
            stats = get_basic_stats(df, columns)
            response = stats.to_markdown()
        elif method == "show_sample":
            response = get_table_sample(df)
        elif method == "plot_distribution":
            feature = decision.get("feature")
            if not feature or feature not in df.columns:
                response = "Please specify a valid feature to plot."
            else:
                fig_or_img = plot_feature_distribution(df, feature)
                response = f"Distribution plot for '{feature}':"

        # --- Model training & management ---
        elif method == "train_model":
            model_type = decision.get("model_type")
            if not model_type:
                response = "Please specify which model to train (e.g., KNN, DTC, NBC)."
            else:
                # Prepare train/test split
                X = state["X_encoded"]
                y = state["y"]
                state["explainer"] = ExplanationEngine(
                    model=state["model"],
                    X=X,
                    y=y,
                    feature_names=list(X.columns),
                    categorical_features=[],  # update as needed
                    class_names=list(map(str, sorted(np.unique(y)))),
                )

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                state["X_train"], state["X_test"] = X_train, X_test
                state["y_train"], state["y_test"] = y_train, y_test

                model, filename = train_model(
                    X_train, y_train, model_type.lower(), label_column
                )
                state["model"] = model
                state["model_type"] = model_type.lower()
                state["label_column"] = label_column
                response = (
                    f"Model '{model_type.upper()}' trained and saved as {filename}."
                )
        elif method == "load_model":
            model_type = decision.get("model_type")
            if not model_type:
                response = "Please specify which model to load."
            else:
                try:
                    X = state["X_encoded"]
                    y = state["y"]
                    model = load_model(model_type.lower(), label_column)
                    state["model"] = model
                    state["model_type"] = model_type.lower()
                    state["label_column"] = label_column
                    response = f"Model '{model_type.upper()}' loaded successfully."
                    state["explainer"] = ExplanationEngine(
                        model=state["model"],
                        X=X,
                        y=y,
                        feature_names=list(X.columns),
                        categorical_features=[],  # update as needed
                        class_names=list(map(str, sorted(np.unique(y)))),
                    )
                except Exception as e:
                    response = f"Failed to load model: {e}"
        elif method == "list_models":
            available = list_models()
            response = "Available models:\n" + "\n".join(available)
        elif method == "delete_model":
            model_type = decision.get("model_type")
            target_label = decision.get(
                "label_column", label_column
            )  # use current label if not specified
            if not model_type:
                response = "Please specify which model to delete."
            else:
                deleted = delete_model(model_type.lower(), target_label)
                if deleted:
                    response = f"Model '{model_type.upper()}' deleted successfully."
                    # remove from state if it was the current model
                    if (
                        state.get("model")
                        and state.get("model_type", "").lower() == model_type.lower()
                        and state.get("label_column") == target_label
                    ):
                        state["model"] = None
                        state["model_type"] = None
                        state["explainer"] = None
                else:
                    response = f"Model '{model_type.upper()}' not found or could not be deleted."

        # --- Model evaluation ---
        elif method == "evaluate_model":
            if not model or X_test is None or y_test is None:
                response = "Please train or load a model first."
            else:
                metrics = evaluate_model(model, X_test, y_test)
                model_name = state.get("model_type", "unknown").upper()
                label = state.get("label_column", "unknown")
                response = (
                    f"Evaluating with model: {model_name} (target label: {label})\n"
                    + "\n".join(
                        f"{k.title()}: {v:.3f}"
                        for k, v in metrics.items()
                        if v is not None
                    )
                )
        elif method == "predict_label":
            instance_idx = decision.get("instance_idx")
            test_sample_path = decision.get("test_sample_path")
            label_column = state.get("label_column")
            training_columns = (
                list(state["X_encoded"].columns)
                if state.get("X_encoded") is not None
                else None
            )

            if (
                model is None
                or state.get("X_encoded") is None
                or state.get("y") is None
            ):
                response = "Please train or load a model first."

            elif test_sample_path:  # when the user provides a test sample path
                response = predict_instance_label(
                    model,
                    state["X_encoded"],
                    state["y"],
                    df,
                    instance_idx=None,
                    test_sample_path=test_sample_path,
                    label_column=label_column,
                    training_columns=training_columns,
                )

            elif (
                instance_idx is None
                or instance_idx < 0
                or instance_idx >= len(state["X_encoded"])
            ):
                response = f"Please provide a valid instance index (0 - {len(state['X_encoded'])-1})."
            else:  # when the user asks for a specific instance index
                response = predict_instance_label(
                    model, state["X_encoded"], state["y"], df, instance_idx=instance_idx
                )

        elif method == "plot_confusion_matrix":
            if not model or X_test is None or y_test is None:
                response = "Please train or load a model first."
            else:
                model_name = state.get("model_type", "unknown").upper()
                label = state.get("label_column", "unknown")
                fig_or_img = plot_confusion_matrix(model, X_test, y_test)
                response = f"Here is the confusion matrix for model: {model_name} (target label: {label})"
        elif method == "plot_roc_auc":
            if not model or X_test is None or y_test is None:
                response = "Please train or load a model first."
            else:
                try:
                    model_name = state.get("model_type", "unknown").upper()
                    label = state.get("label_column", "unknown")
                    fig_or_img = plot_roc_auc(model, X_test, y_test)
                    response = f"Here is the ROC curve for model: {model_name} (target label: {label})"
                except Exception as e:
                    response = f"ROC AUC plotting failed: {e}"

        # --- XAI explanation ---
        elif method in ["shap", "lime", "dice"]:
            instance_idx = decision.get("instance_idx")
            test_sample_path = decision.get("test_sample_path")
            # Build explainer if needed
            if not state.get("explainer"):
                # (Re)build explainer on demand for current model
                X = state["X_encoded"]
                y = state["y"]
                # print("model in state: ", state["model"])
                # print("explainer in state: ", state["explainer"])

                # For one-hot, categorical_features should be defined here
                state["explainer"] = ExplanationEngine(
                    model=state["model"],
                    X=X,
                    y=y,
                    feature_names=list(X.columns),
                    categorical_features=[],  # NOTE: this needs to be set correctly - i need to handle how to infer this
                    class_names=list(map(str, sorted(np.unique(y)))),
                )
            explainer = state["explainer"]
            if not model:
                response = "Please train or load a model first."
            elif test_sample_path:
                test_df = load_test_sample(test_sample_path)
                if test_df.shape[0] != 1:
                    response = f"Test sample must contain exactly one row. Found {test_df.shape[0]} rows."
                else:
                    cat_cols = [
                        c
                        for c in test_df.columns
                        if test_df[c].dtype == "object"
                        or test_df[c].dtype.name == "category"
                    ]
                    label_column = state.get("label_column")
                    if label_column and label_column in test_df.columns:
                        test_df = test_df.drop(columns=[label_column])
                    # one-hot encode to match training columns
                    test_encoded = pd.get_dummies(test_df, columns=cat_cols)
                    test_encoded = test_encoded.reindex(
                        columns=state["X_encoded"].columns, fill_value=0
                    )
                    # use only row 0 when the test sample is provided
                    if method == "shap":
                        explanation_result = explainer.shap_explainer(test_encoded)
                        prompt_content = build_instance_xai_prompt(
                            0,  # for test sample, always index 0
                            model,
                            explainer,
                            state,
                            test_df,  # for user-friendly display
                            method,
                            explanation_result,
                            pred_label=model.predict(test_encoded)[0],
                            actual_label=None,
                            test_sample_path=test_sample_path,
                        )
                    elif method == "lime":
                        explanation_result = explainer.lime_explainer.explain_instance(
                            data_row=test_encoded.iloc[0],
                            predict_fn=model.predict_proba,
                            num_features=5,  # default to 5 features
                        ).as_list()
                        prompt_content = build_instance_xai_prompt(
                            0,  # for test sample, always index 0
                            model,
                            explainer,
                            state,
                            test_df,  # for user-friendly display
                            method,
                            explanation_result,
                            pred_label=model.predict(test_encoded)[0],
                            actual_label=None,
                            test_sample_path=test_sample_path,
                        )
                    elif method == "dice":
                        explanation_result = explainer.generate_counterfactuals(
                            test_encoded, total_CFs=3
                        )
                        prompt_content = build_instance_xai_prompt(
                            0,  # for test sample, always index 0
                            model,
                            explainer,
                            state,
                            test_df,  # for user-friendly display
                            method,
                            explanation_result,
                            pred_label=model.predict(test_encoded)[0],
                            actual_label=None,
                            test_sample_path=test_sample_path,
                        )
                    response = llm_client.query(
                        llm_client.format_explanation_prompt(user_input, prompt_content)
                    )
            elif instance_idx is None or instance_idx >= len(df):
                response = f"Please provide a valid instance index (0 - {len(df)-1})."

            else:  # when the user asks for a specific instance index within the dataset
                # Handle each XAI method
                if method == "shap":
                    explanation_result = explainer.explain_with_shap(instance_idx)
                    # prompt_content = build_instance_xai_prompt(
                    #     instance_idx,
                    #     model,
                    #     explainer,
                    #     state,
                    #     df,
                    #     method,
                    #     explanation_result,
                    # )
                    # response = llm_client.query(
                    #     llm_client.format_explanation_prompt(user_input, prompt_content)
                    # )

                elif method == "lime":
                    explanation_result = explainer.explain_with_lime(instance_idx)
                    # prompt_content = build_instance_xai_prompt(
                    #     instance_idx,
                    #     model,
                    #     explainer,
                    #     state,
                    #     df,
                    #     method,
                    #     explanation_result,
                    # )
                    # response = llm_client.query(
                    #     llm_client.format_explanation_prompt(user_input, prompt_content)
                    # )

                elif method == "dice":  # dice requires original instance and cf table
                    explanation_result = explainer.generate_counterfactuals(
                        instance_idx
                    )
                prompt_content = build_instance_xai_prompt(
                    instance_idx,
                    model,
                    explainer,
                    state,
                    df,
                    method,
                    explanation_result,
                )
                response = llm_client.query(
                    llm_client.format_explanation_prompt(user_input, prompt_content)
                )

        # Global feature importance
        elif method in ["feature_importance", "global_shap_summary"]:
            # try model's built-in feature importance first
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                features = state["X_encoded"].columns
                importance_tuples = sorted(
                    zip(features, importances), key=lambda x: -abs(x[1])
                )
                prompt_content = (
                    "Model feature importances (highest to lowest):\n"
                    + "\n".join(
                        [f"{feat}: {imp:.3f}" for feat, imp in importance_tuples]
                    )
                )
                response = llm_client.query(
                    llm_client.format_summary_prompt(user_input, prompt_content)
                )
            else:
                # fallback to SHAP global summary
                if (
                    explainer is not None
                    and hasattr(explainer, "shap_explainer")
                    and explainer.shap_explainer is not None
                ):
                    shap_values = explainer.shap_explainer(explainer.X)
                    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                    mean_abs_shap = np.array(mean_abs_shap).flatten()
                    features = explainer.feature_names
                    importance_tuples = sorted(
                        zip(features, mean_abs_shap), key=lambda x: -x[1]
                    )
                    prompt_content = (
                        "Global SHAP feature importances (mean absolute SHAP values from all instances):\n"
                        + "\n".join(
                            [f"{feat}: {imp:.3f}" for feat, imp in importance_tuples]
                        )
                    )
                    response = llm_client.query(
                        llm_client.format_summary_prompt(user_input, prompt_content)
                    )
                else:
                    response = "Feature importance not available for this model type."
        else:
            response = "Sorry, I didn't understand your request."

        # If image/plot is generated, return it in second output slot
        return response, fig_or_img

    except json.JSONDecodeError as e:
        logging.error("LLM JSON Parsing error: %s", e)
        return "Sorry, I couldn't understand the question clearly.", None

    except Exception as e:
        # logging.error("Explanation generation failed: %s", e)
        logging.error("Explanation generation failed", exc_info=True)
        return f"Sorry, something went wrong generating the explanation: {e}", None


def build_instance_xai_prompt(
    instance_idx,
    model,
    explainer,
    state,
    df,
    method,
    explanation_result,
    pred_label=None,
    actual_label=None,
    test_sample_path=None,
):
    # this function is to build a prompt for LLM to explain the instance,
    # which also include the predicted vs actual label and original instance values
    pred_label = model.predict([explainer.X.iloc[instance_idx]])[0]
    actual_label = state["y"].iloc[instance_idx]
    original_instance = df.iloc[instance_idx]
    instance_details = "\n".join(
        f"{col}: {val}" for col, val in original_instance.items()
    )
    prompt_content = f"Instance details (index {instance_idx}"
    if test_sample_path:
        prompt_content += f", from file: {test_sample_path}"
    prompt_content += "):\n"
    prompt_content += instance_details + "\n"
    if pred_label is not None:
        prompt_content += f"- Predicted label: {pred_label}\n"
    if actual_label is not None:
        prompt_content += f"- Actual label: {actual_label}\n"
    prompt_content += f"\n\n{method.upper()} explanation:\n{generate_explanation_text(method, explanation_result)}"
    return prompt_content
    # return (
    #     f"Instance {instance_idx}:\n"
    #     f"- Predicted label: {pred_label}\n"
    #     f"- Actual label: {actual_label}\n"
    #     f"- Original instance values:\n{instance_details}\n\n"
    #     f"Explanation:\n"
    #     f"{generate_explanation_text(method, explanation_result)}"
    # )


def generate_explanation_text(method, explanation_result):
    # This function formats the XAI explanation result into a human-readable string
    if method in ["shap", "lime"]:
        return "\n".join(
            f"{feature}: {float(importance):.3f}"
            for feature, importance in explanation_result
        )
    elif method == "dice":
        if hasattr(explanation_result, "to_markdown"):
            return explanation_result.to_markdown()
        return str(explanation_result)
    else:
        return str(explanation_result)
