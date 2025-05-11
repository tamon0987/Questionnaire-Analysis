"""
Main entry point for the Survey Vector Pipeline.
Loads config, runs the pipeline, and saves processed data for the dashboard.
"""

import argparse
import yaml
import pandas as pd
from data_loader import SurveyDataLoader
from preprocessing import preprocess_text
from vectorizer import HashingVectorizerWrapper
from reducer import TruncatedSVDWrapper
from clustering import KMeansClusterer
from storage import save_dataframe

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Survey Vector Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Loaded config:", config)

    # 1. Load data
    loader = SurveyDataLoader(config)
    df = loader.load()
    print(f"Loaded {len(df)} rows from data source.")

    # 2. Preprocess text
    text_col = "answer" if "answer" in df.columns else df.columns[0]
    preprocessed = []
    for t in df[text_col]:
        p = preprocess_text(str(t), config.get("preprocessing", {}))
        preprocessed.append(p)
    df["preprocessed"] = preprocessed
    df = df[df["preprocessed"].notnull()]
    print(f"After preprocessing: {len(df)} rows remain.")

    # 3. Vectorize
    vectorizer = HashingVectorizerWrapper(config.get("vectorizer", {}))
    X = vectorizer.transform(df["preprocessed"])

    # 4. Reduce to 3D
    reducer = TruncatedSVDWrapper(config.get("reducer", {}))
    X_3d = reducer.fit(X).transform(X)
    df["x"] = X_3d[:, 0]
    df["y"] = X_3d[:, 1]
    df["z"] = X_3d[:, 2]

    # 5. Cluster
    clusterer = KMeansClusterer(config.get("clustering", {}))
    clusterer.fit(X_3d)
    df["cluster"] = clusterer.predict(X_3d)

    # 6. Save processed data for dashboard
    df.rename(columns={text_col: "original"}, inplace=True)
    out_cols = ["x", "y", "z", "cluster", "original", "preprocessed"]
    save_dataframe(df[out_cols], "processed_data.parquet", format="parquet")
    print("Saved processed data to processed_data.parquet")

if __name__ == "__main__":
    main()