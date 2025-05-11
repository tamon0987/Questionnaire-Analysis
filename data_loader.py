"""
Data loading module for Survey Vector Pipeline.
Supports CSV, JSON, and extensible database sources.
"""

import pandas as pd

class SurveyDataLoader:
    def __init__(self, config):
        self.config = config

    def load(self):
        src = self.config["data_source"]
        if src["type"] == "csv":
            return pd.read_csv(src["path"])
        elif src["type"] == "json":
            return pd.read_json(src["path"])
        # TODO: Add database support
        else:
            raise ValueError(f"Unsupported data source type: {src['type']}")