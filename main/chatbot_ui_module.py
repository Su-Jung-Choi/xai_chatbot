# main/chatbot_ui_module.py


import gradio as gr
import pandas as pd
from dotenv import load_dotenv
import os
from PIL import Image
import io
import base64

from main.chat_handler import handle_user_input
from myapp.llm_interface import LLMInterface
from myapp.explanation_engine import ExplanationEngine

# --- Load LLM API Key ---
load_dotenv(dotenv_path="env/credentials.env")
DEEPSEEK_API_KEY = os.getenv("deepseek_api")
if not DEEPSEEK_API_KEY:
    raise ValueError("DeepSeek API key must be provided in env/credentials.env")
llm_client = LLMInterface(api_key=DEEPSEEK_API_KEY)


# --- State Storage for Session ---
def get_default_state():
    # Holds: df, label_column, model, explainer, X_train, X_test, y_train, y_test, feature_names, class_names, categorical_features, history
    return {
        "df": None,
        "label_column": None,
        "model": None,
        "explainer": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "feature_names": None,
        "class_names": None,
        "categorical_features": None,
        "history": [],
    }


# --- File Upload Handler ---
def upload_file_handler(file_path, state, history):
    # NOTE: the history variable is not directly used here, but it needs to be passed for Gradio wiring
    # because to set chatbot as an output, it needs to be included in both input and output
    try:
        df = pd.read_csv(file_path)
        state["df"] = df
        state["history"].append(
            {
                "role": "assistant",
                "content": "CSV loaded! Please select the label (target) column from the dropdown.",
            }
        )
        # Reset state when new data loaded
        state["label_column"] = None
        state["model"] = None
        state["explainer"] = None
        state["X_train"] = None
        state["X_test"] = None
        state["y_train"] = None
        state["y_test"] = None
        state["feature_names"] = None
        state["categorical_features"] = None
        state["class_names"] = None

        return (
            "",
            state["history"],
            None,
            gr.update(choices=list(df.columns), interactive=True, value=None),
        )  # Return empty message, updated history, and no plot/image

    except Exception as e:
        state["history"].append(
            {
                "role": "assistant",
                "content": f"Failed to load CSV: {e}",
            }
        )
        return (
            "",
            state["history"],
            None,
            gr.update(choices=[], interactive=False, value=None),
        )


# --- Label Column Handler ---
def label_column_handler(label_name, state, history):
    df = state.get("df")
    if df is None:
        history.append(
            {"role": "assistant", "content": "Please upload a CSV file first."}
        )
        return "", history, None
    if label_name not in df.columns:
        history.append(
            {
                "role": "assistant",
                "content": f"Column '{label_name}' not found in the dataset. Please select a valid label column.",
            }
        )
        return "", history, None

    state["label_column"] = label_name

    # encoding categorical features
    X = df.drop(columns=[label_name])
    y = df[label_name]
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # one-hot encode
    X_encoded = pd.get_dummies(X, columns=cat_cols).astype(float)
    print(X_encoded.dtypes)

    # save encoded and original for future use
    state["X_encoded"] = X_encoded
    state["y"] = y

    history.append(
        {
            "role": "assistant",
            "content": f"Label column set to '{label_name}' and categorical features encoded. Now you can ask questions!",
        }
    )

    return "", history, None


def decode_base64_to_pil(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img = Image.open(io.BytesIO(img_bytes))
    return img


# --- Chat Interface Handler ---
def chat_interface(user_input, history, state):
    df = state.get("df")
    label_column = state.get("label_column")
    model = state.get("model")
    explainer = state.get("explainer")
    X_train = state.get("X_train")
    X_test = state.get("X_test")
    y_train = state.get("y_train")
    y_test = state.get("y_test")
    history = state.get("history", [])

    # Block all queries until CSV + label are set
    if df is None:
        history.append(
            {"role": "assistant", "content": "Please upload a CSV file first."}
        )

        return "", history, None

    if label_column is None:
        history.append(
            {
                "role": "assistant",
                "content": "Please specify the label column (target variable) before asking questions.",
            }
        )

        return "", history, None

    # --- NOTE: Dynamic Model/Explainer Sync Logic ---
    # If model or explainer is missing, certain actions are not allowed (e.g., XAI methods)
    # The chat_handler itself should raise "Please train or load a model first" for those cases

    # Route to chat_handler, pass only what is needed.
    response, fig_or_img = handle_user_input(
        user_input,
        history,
        explainer,
        llm_client,
        df,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        label_column,
        state,
    )
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    state["history"] = history

    # Plot/image handling
    if fig_or_img is None:
        return "", history, None
    # If it's a matplotlib Figure
    elif hasattr(fig_or_img, "savefig"):
        return "", history, fig_or_img

    else:
        return "", history, None


# --- Gradio UI ---
def start_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# ExplainMyModel Chatbot")
        state = gr.State(get_default_state())

        file_upload = gr.File(label="Upload your CSV file", type="filepath")
        label_dropdown = gr.Dropdown(
            label="Select label column (target variable)", choices=[], interactive=False
        )
        # label_box = gr.Textbox(label="Enter label column name (target variable)")
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(label="Ask a question about your data or model")
        plot_output = gr.Plot(label="Plot or Distribution Output")
        clear = gr.Button("Clear All")

        # File upload sets the dataframe
        file_upload.change(
            upload_file_handler,
            inputs=[file_upload, state, chatbot],
            outputs=[
                msg,
                chatbot,
                plot_output,
                label_dropdown,
            ],  # chatbot is where the status messages will appear
        )
        # Label column selection
        label_dropdown.change(
            label_column_handler,
            inputs=[label_dropdown, state, chatbot],
            outputs=[msg, chatbot, plot_output],
        )
        # Main chat interaction
        msg.submit(
            chat_interface,
            inputs=[msg, chatbot, state],
            outputs=[msg, chatbot, plot_output],
        )

        # Clear history and state
        def reset_state():
            return (
                "",
                get_default_state(),
                None,
                gr.update(choices=[], interactive=False, value=None),
            )

        clear.click(
            reset_state, [], [msg, state, plot_output, label_dropdown], queue=False
        )

    demo.launch()


if __name__ == "__main__":
    start_ui()
