# Survey Vector Pipeline

Modular, scalable pipeline for large-scale open-ended survey answer analysis, vectorization, dimensionality reduction, clustering, and interactive dashboarding.
![3dgraph](https://github.com/user-attachments/assets/d4adc90d-4b81-4344-baa6-aff893c763a2)

## Directory Structure

```
survey_vector_pipeline/
├── config.yaml
├── main.py
├── data_loader.py
├── preprocessing.py
├── vectorizer.py
├── reducer.py
├── clustering.py
├── storage.py
├── dashboard.py
├── utils.py
└── README.md
```

## Quick Start

1. Edit `config.yaml` to set your data source and pipeline options.
2. Run the pipeline:
   ```
   python main.py --config config.yaml
   ```
3. Launch the dashboard:
   ```
   python dashboard.py --config config.yaml
   ```

## Features

- Pluggable data loaders (CSV, JSON, DB)
- Unicode normalization, PII masking (Presidio+SpaCy), language filtering
- HashingVectorizer with persistent seed, extensible vectorizer interface
- TruncatedSVD/IncrementalSVD, extensible reducer interface
- KMeans/HDBSCAN clustering, with scaling
- All pipeline steps configurable via YAML/ENV
- Plotly Dash/Streamlit dashboard with downsampling/WebGL options
- Reproducible: logs hash seed, vectorizer/reducer params, SVD components

See [`../survey_vector_pipeline_architecture.md`](../survey_vector_pipeline_architecture.md) for full architecture details.