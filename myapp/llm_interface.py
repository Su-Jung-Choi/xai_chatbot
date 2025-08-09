import logging
import os
from openai import OpenAI


class LLMInterface:
    def __init__(
        self,
        api_key=None,
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        if not self.api_key:
            raise ValueError("DeepSeek API key must be provided via argument")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logging.info("Initialized LLM interface with model %s", self.model_name)

    def query(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 512,  # to limit the number of tokens in the response
    ):
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            logging.info("LLM response: %s", answer)
            return answer

        except Exception as e:
            logging.error("LLM API call failed: %s", e)
            return "Sorry, I couldn't generate a response at the moment."

    def format_agentic_prompt(self, user_query: str, context_info: str = "") -> str:
        return f"""
    You are a decision-making agent for an XAI chatbot. The user will ask you questions about a machine learning model and its dataset.

    Your job is to select the correct method and parameters for the request.

    Possible methods:
    - "shap" — Explain a prediction with SHAP.
    - "lime" — Explain a prediction with LIME.
    - "dice" — Generate counterfactuals.
    - "describe_data" — Summarize the dataset and its columns.
    - "basic_stats" — Report basic statistics (mean, median, min, max, etc) for all or specified features.
    - "show_sample" — Show a table sample (e.g., first 5 rows).
    - "plot_distribution" — Show a histogram/distribution plot for a feature.
    - "train_model": train a specific model ("KNN", "DTC", "NBC") with given label column
    - "evaluate_model": show model accuracy, precision, recall, f1, ROC-AUC
    - "plot_confusion_matrix": show the confusion matrix as a plot
    - "plot_roc_auc": plot the ROC curve (if possible)
    - "load_model": load a trained model by type (e.g., "KNN", "DTC", "NBC")
    - "list_models": list all available trained models
    - "delete_model": delete a trained model by type and label column
    - "feature_importance": Show global feature importances (model-based).
    - "global_shap_summary": Show global SHAP feature importances (mean absolute SHAP values over all instances).
    - "predict_label": Predict the label for a specific row and compare it with the actual label.

    Return ONLY a JSON object in the following format, and NOTHING ELSE. Do not include explanations, comments, or markdown. 
    {{
        "method": "...",       // (see above)
        "instance_idx": int,   // (if required, e.g. for SHAP/LIME/DICE)
        "feature": str,       // (if required, e.g. for plot_distribution)
        "model_type": str,   // (if applicable, for training)
        "label_column": str,  // (if applicable)
        "num_features": int,  // (if lime)
        "total_CFs": int      // (if dice)
    }}

    Here are some examples:

    User: "Show me the columns in the dataset."
    {{"method": "describe_data"}}

    User: "What is the mean and median of Age?"
    {{"method": "basic_stats", "features": ["Age"], "stats": ["mean", "median"]}}

    User: "Display a sample of the dataset."
    {{"method": "show_sample"}}

    User: "Plot a histogram for Credit amount."
    {{"method": "plot_distribution", "feature": "Credit amount"}}

    User: "Explain the prediction for row 2 using SHAP."
    {{"method": "shap", "instance_idx": 2}}

    User: "What features were most important for this prediction?"
    {{"method": "shap", "instance_idx": 0}}

    User: "Train a decision tree model for this data."
    {{"method": "train_model", "model_type": "dtc"}}

    User: "Load the k-nearest neighbors classifier."
    {{"method": "load_model", "model_type": "knn"}}

    User: "Show available trained models."
    {{"method": "list_models"}}

    User: "Show summary statistics."
    {{"method": "basic_stats"}}

    User: "Explain row 3 with LIME."
    {{"method": "lime", "instance_idx": 3}}

    User: What is the accuracy of the model?
    {{"method": "evaluate_model"}}

    User: Show me the confusion matrix
    {{"method": "plot_confusion_matrix"}}

    User: Delete the model for KNN with label 'target'.
    {{"method": "delete_model", "model_type": "knn", "label_column": "target"}}

    User: What features are most important for this model?
    {{"method": "feature_importance"}}

    User: What is the predicted label for index 10?
    {{"method": "predict_label", "instance_idx": 10}}

    User: Explain the prediction for test_set/sample1.csv with SHAP.
    {{"method": "shap", "test_sample_path": "test_set/sample1.csv"}}
    
    User query: "{user_query}"
    {context_info}

    """

    def format_explanation_prompt(self, user_query: str, explanation_text: str) -> str:
        # This method is only for XAI explanation queries
        return f"""
        You are an XAI chatbot assistant. The user asked a question about a machine learning model.
        Here is the data for the original instance (input sample) and the model explanation result.
        Always include a summary of the original instance values before giving the explanation, even if the user did not requiest it.
        
        {explanation_text}

        Please provide a natural language summary that answers the user query:
        "{user_query}"
        """

    def format_summary_prompt(self, user_query: str, model_stats: str) -> str:
        return f"""
        You are an XAI chatbot assistant. The user asked a question about a machine learning model.
        Here is the model summary:

        {model_stats}

        Please provide a natural language summary that answers the user query:
        "{user_query}"
        """
