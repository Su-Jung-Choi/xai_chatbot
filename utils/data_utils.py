# utils/data_utils.py
# this script includes functions the LLM can execute.

import pandas as pd
import numpy as np
from typing import List, Optional
import matplotlib.pyplot as plt


def get_dataset_description(df: pd.DataFrame, label_col: Optional[str] = None) -> str:
    """
    get_dataset_description function is to generate a plain-text description of the dataset including
    column names, data types, and unique label values (if label_col provided).
    """
    if df is None or len(df) == 0:
        return "No data loaded."

    desc = []
    desc.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.\n")
    desc.append("Columns and data types:")
    for col in df.columns:
        dtype = df[col].dtype
        desc.append(f"  - {col}: {dtype}")

    if label_col and label_col in df.columns:
        unique_vals = df[label_col].unique()
        desc.append(f"\nUnique values in '{label_col}': {unique_vals.tolist()}")
        desc.append(f"Counts: {df[label_col].value_counts().to_dict()}")
    return "\n".join(desc)


def get_basic_stats(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    get_basic_stats function is to get basic statistics (mean, median, mode, min, max, std) for all numeric columns or specified ones.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame({"info": ["No data loaded."]})

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = pd.DataFrame(
        {
            "mean": df[columns].mean(),
            "median": df[columns].median(),
            "mode": (
                df[columns].mode().iloc[0] if not df[columns].mode().empty else np.nan
            ),
            "min": df[columns].min(),
            "max": df[columns].max(),
            "std": df[columns].std(),
        }
    )
    stats.index.name = "feature"
    return stats.round(3)


def get_table_sample(df: pd.DataFrame, num_rows: int = 5) -> str:
    """
    get_table_sample function is to return the first n rows of the DataFrame as a markdown-formatted table (right now it returns the first 5 rows by default).
    """
    if df is None or len(df) == 0:
        return "No data loaded."
    try:
        return df.head(num_rows).to_markdown(index=False)
    except Exception as e:
        # Fallback to CSV if markdown fails
        return df.head(num_rows).to_csv(index=False)


def plot_feature_distribution(
    df: pd.DataFrame, feature: str, bins: int = 10
) -> plt.Figure:
    """
    plot_feature_distribution function is to generate a histogram of the specified feature's distribution.
    """
    if df is None or feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

    fig, ax = plt.subplots()
    if pd.api.types.is_numeric_dtype(df[feature]):
        df[feature].hist(ax=ax, bins=bins, color="skyblue", edgecolor="black")
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    else:  # Categorical feature
        df[feature].value_counts().plot(
            kind="bar", ax=ax, color="skyblue", edgecolor="black"
        )
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")

    plt.tight_layout()
    return fig
