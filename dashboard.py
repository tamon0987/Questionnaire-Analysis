"""
Streamlit dashboard for Survey Vector Pipeline.
Displays a 3D scatter plot of reduced survey vectors with interactive features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import os

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_processed_data():
    # Placeholder: expects a file 'processed_data.parquet' with columns ['x', 'y', 'z', 'cluster', 'original', 'preprocessed']
    if os.path.exists("processed_data.parquet"):
        return pd.read_parquet("processed_data.parquet")
    else:
        st.error("Processed data file 'processed_data.parquet' not found. Please run the pipeline first.")
        st.stop()

def main():
    st.set_page_config(page_title="Survey Vector 3D Explorer", layout="wide")
    st.title("Survey Vector 3D Explorer")

    config_path = st.sidebar.text_input("Config file", "config.yaml")
    if not os.path.exists(config_path):
        st.sidebar.error(f"Config file '{config_path}' not found.")
        st.stop()
    config = load_config(config_path)

    df = load_processed_data()

    st.sidebar.markdown("### Search")
    search_text = st.sidebar.text_input("Search for text (case-insensitive):", "")

    # Filter by search
    if search_text:
        mask = df['original'].str.contains(search_text, case=False, na=False) | \
               df['preprocessed'].str.contains(search_text, case=False, na=False)
        highlight_df = df[mask]
        st.sidebar.write(f"Found {len(highlight_df)} matches.")
    else:
        highlight_df = pd.DataFrame()

    # 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='cluster',
        hover_data=['original', 'preprocessed'],
        opacity=0.7,
        title="3D Survey Answer Embeddings"
    )

    # Highlight search results
    if not highlight_df.empty:
        fig.add_trace(
            px.scatter_3d(
                highlight_df, x='x', y='y', z='z',
                marker=dict(size=6, color='red', symbol='diamond'),
                hover_data=['original', 'preprocessed']
            ).data[0]
        )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Click a point in the plot to see details below (use Plotly's lasso or box select tool).")
    st.dataframe(df[['original', 'preprocessed', 'cluster']])

if __name__ == "__main__":
    main()