# myapp/llm_interface.py
from __future__ import annotations
import logging
from openai import OpenAI
from myapp.method_registry import agentic_methods_block


class LLMInterface:
    def __init__(
        self,
        api_key=None,
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-chat",
    ):
        """
        Initialize the LLM interface with the provided API key and model details.
        """
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
        """
        query method is to send a request to the LLM with the provided prompt and parameters.
        """
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
        """
        format_agentic_prompt method pulls the possible methods and examples from the method_registry.
        It provides a structured prompt for the LLM to understand the user's intent and the available actions.
        """
        # NOTE: a new method always need to be included in the method_registry.py
        methods_block = agentic_methods_block()

        return f"""
        You are a decision-making agent for an XAI chatbot. The user will ask you questions about a machine learning model and its dataset.

        Your job is to select the correct method and parameters for the request.
        {methods_block}

        Return ONLY a JSON object in the following format, and NOTHING ELSE. Do not include explanations, comments, or markdown. 
        {{
            "method": "...",       // (see above)
            "instance_idx": int,   // (if required, e.g. for SHAP/LIME/DICE)
            "feature": str,       // (if required, e.g. for plot_distribution)
            "model_type": str,   // (if applicable, for training)
            "label_column": str,  // (if applicable)
            "num_features": int,  // (if lime)
            "total_CFs": int,      // (if dice)
            "test_sample_path": str // (if the user points to a 1-row CSV for prediction/explanation)
        }}


        User query: "{user_query}"
        {context_info}

        """

    def format_explanation_prompt(self, user_query: str, explanation_text: str) -> str:
        """
        format_explanation_prompt method is to create a structured prompt for the LLM to generate explanations.
        This is only for XAI explanation queries.
        """

        return f"""
        You are an XAI chatbot assistant. The user asked a question about a machine learning model.
        Here is the data for the original instance (input sample) and the model explanation result.
        Always include a summary of the original instance values before giving the explanation, even if the user did not request it.
        
        {explanation_text}

        Please provide a natural language summary that answers the user query:
        "{user_query}"
        """

    def format_summary_prompt(self, user_query: str, model_stats: str) -> str:
        """
        format_summary_prompt method is to create a structured prompt for the LLM to generate a summary of the output from the model.
        """
        return f"""
        You are an XAI chatbot assistant. The user asked a question about a machine learning model.
        Here is the model summary:

        {model_stats}

        Please provide a natural language summary that answers the user query:
        "{user_query}"
        """
