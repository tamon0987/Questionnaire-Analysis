"""
Storage module for Survey Vector Pipeline.
Handles saving/loading of processed data, vectors, 3D coordinates, cluster labels, and metadata.
"""

import pandas as pd
import pickle

def save_dataframe(df, path, format="parquet"):
    if format == "parquet":
        df.to_parquet(path, index=False)
    elif format == "csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_dataframe(path, format="parquet"):
    if format == "parquet":
        return pd.read_parquet(path)
    elif format == "csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def save_metadata(metadata, path):
    with open(path, "wb") as f:
        pickle.dump(metadata, f)

def load_metadata(path):
    with open(path, "rb") as f:
        return pickle.load(f)