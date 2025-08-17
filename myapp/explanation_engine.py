# NOTE: currently this is only compatible using the classification models with encoded datasets and without pipelines
# myapp/explanation_engine.py

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Any
import shap
import lime.lime_tabular
import dice_ml
from dice_ml import Dice


class ExplanationEngine:
    def __init__(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        categorical_features: List[int],
        class_names: List[str],
        backend: str = "sklearn",
    ):
        """
        Initialize the explanation engine with the provided model and dataset.
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names
        self.backend = backend

        # don't init explainers until all data/model present
        self.lime_explainer = None
        self.shap_explainer = None
        self.dice = None

        # Try to setup explainers, but allow for late binding if model/data missing
        self._try_initialize_explainers()

    def _try_initialize_explainers(self):
        """
        Set up LIME, SHAP, and DiCE explainers if all required fields are present.
        """
        # Check prerequisites
        if (
            self.model is not None
            and self.X is not None
            and self.y is not None
            and self.feature_names is not None
            and self.class_names is not None
            and len(self.X) > 0
        ):
            try:
                # LIME
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.array(self.X),
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    categorical_features=self.categorical_features,
                    mode="classification",
                )
            except Exception as e:
                print(f"[LIME Initialization Error]: {e}")

            try:
                # SHAP (use Kernel or Permutation explainer for full compatibility)
                self.shap_explainer = shap.Explainer(self.model.predict_proba, self.X)
            except Exception as e:
                print(f"[SHAP Initialization Error]: {e}")

            try:
                # DiCE
                df_for_dice = self.X.copy()
                df_for_dice["target"] = self.y.values

                continuous_features = [
                    name
                    for idx, name in enumerate(self.feature_names)
                    if idx
                    not in self.categorical_features  # NOTE: this should be a list of column indices for categorical features (as in the original DataFrame before one-hot encoding)
                ]
                self.dice_data = dice_ml.Data(
                    dataframe=df_for_dice,
                    continuous_features=continuous_features,
                    outcome_name="target",
                )
                self.dice_model = dice_ml.Model(model=self.model, backend=self.backend)
                self.dice = Dice(self.dice_data, self.dice_model, method="random")
            except Exception as e:
                print(f"[DiCE Initialization Error]: {e}")

    def update(self, model, X, y, feature_names, categorical_features, class_names):
        """
        update method is to update the engine with a new model and/or data.
        This is used when user trains a new model or uploads new data.
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names
        self._try_initialize_explainers()

    def explain_with_lime(
        self, instance_idx: int, num_features: int = 5
    ) -> List[Tuple[str, float]]:
        """
        explain_with_lime method is to generate explanations for a specific instance within the given dataset using LIME.
        """

        if self.lime_explainer is None:
            raise RuntimeError(
                "LIME explainer not initialized. Please train or load a model first."
            )
        instance = self.X.iloc[instance_idx].values
        explanation = self.lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=self.model.predict_proba,
            num_features=num_features,
        )
        return explanation.as_list()

    def explain_with_shap(
        self, instance_idx: int, class_idx: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        explain_with_shap method is to generate explanations for a specific instance within the given dataset using SHAP.
        """

        if self.shap_explainer is None:
            raise RuntimeError(
                "SHAP explainer not initialized. Please train or load a model first."
            )
        instance = self.X.iloc[[instance_idx]].astype(np.float64)
        shap_values = self.shap_explainer(instance)

        # Multi-class or binary handling
        if hasattr(shap_values, "values"):
            vals = None
            if shap_values.values.ndim == 1:
                vals = shap_values.values
            elif shap_values.values.ndim == 2:
                vals = shap_values.values[0]
            elif shap_values.values.ndim == 3:
                # Multi-class
                if class_idx is None:
                    proba = self.model.predict_proba(instance)[0]
                    class_idx = int(np.argmax(proba))
                vals = shap_values.values[0, :, class_idx]
            else:
                raise RuntimeError("Unexpected SHAP values shape.")
            vals = np.asarray(vals).flatten()
            return list(zip(self.feature_names, vals))
        else:
            raise RuntimeError("Could not interpret SHAP values.")

    def explain_with_shap_row(
        self,
        row,  # np.ndarray shape (n_features,) or (1, n_features) or 1-row DataFrame
        class_idx: Optional[int] = None,
        feature_names: Optional[list] = None,
    ):
        """
        explain_with_shap_row method is to generate explanations for a specific row provided by the user using SHAP.
        The file must be in the test_set folder. The columns must match the training set and it is expected to have only one row in the test set.
        """
        if self.shap_explainer is None:
            raise RuntimeError(
                "SHAP explainer not initialized. Please train or load a model first."
            )

        # Normalize row to a 1-row DataFrame with the same columns as training
        if isinstance(row, np.ndarray):
            row = row.reshape(1, -1)
            row_df = pd.DataFrame(row, columns=self.X.columns)
        elif isinstance(row, pd.DataFrame):
            assert row.shape[0] == 1, "row must be a single row DataFrame"
            # ensure same order/columns as training
            row_df = row.reindex(columns=self.X.columns)
        else:
            raise TypeError(f"Unsupported row type: {type(row)}")

        row_df = row_df.astype(np.float64)

        expl = self.shap_explainer(row_df)

        if hasattr(expl, "values"):
            vals = expl.values
            if vals.ndim == 1:  # (n_features,)
                flat = vals
            elif vals.ndim == 2:  # (1, n_features)
                flat = vals[0]
            elif vals.ndim == 3:  # (1, n_features, n_classes)
                if class_idx is None:
                    proba = self.model.predict_proba(row_df)[0]
                    class_idx = int(np.argmax(proba))
                flat = vals[0, :, class_idx]
            else:
                raise RuntimeError("Unexpected SHAP values shape.")
            flat = np.asarray(flat).flatten()
            names = feature_names if feature_names is not None else list(self.X.columns)
            return list(zip(names, flat))
        else:
            raise RuntimeError("Could not interpret SHAP values.")

    def generate_counterfactuals(
        self, instance_idx: int, total_CFs: int = 3
    ) -> pd.DataFrame:
        """
        generate_counterfactuals method is to generate counterfactual explanations for a specific instance within the given dataset using DiCE.
        """
        if self.dice is None:
            raise RuntimeError(
                "DiCE explainer not initialized. Please train or load a model first."
            )
        assert isinstance(
            self.X, pd.DataFrame
        ), f"self.X is not a DataFrame: {type(self.X)}"
        assert isinstance(
            instance_idx, int
        ), f"instance_idx is not an int: {type(instance_idx)}"
        assert (
            0 <= instance_idx < len(self.X)
        ), f"instance_idx out of range: {instance_idx}"

        query_instance = self.X.iloc[[instance_idx]]

        explanation = self.dice.generate_counterfactuals(
            query_instance, total_CFs=total_CFs
        )
        return explanation.cf_examples_list[0].final_cfs_df

    def explain(self, method: str, instance_idx: int, **kwargs):
        """
        explain method is to call the appropriate XAI method.
        """
        if method.lower() == "lime":
            return self.explain_with_lime(instance_idx, **kwargs)
        elif method.lower() == "shap":
            return self.explain_with_shap(instance_idx, **kwargs)
        elif method.lower() == "dice":
            return self.generate_counterfactuals(instance_idx, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
