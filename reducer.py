"""
Dimensionality reduction module for Survey Vector Pipeline.
Implements a pluggable Reducer interface with TruncatedSVD and IncrementalTruncatedSVD.
"""

from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from sklearn.utils import check_random_state

class Reducer:
    def fit(self, X):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

class TruncatedSVDWrapper(Reducer):
    def __init__(self, config):
        self.n_components = config.get("n_components", 3)
        self.incremental = config.get("incremental", False)
        self.random_seed = config.get("random_seed", 42)
        if self.incremental:
            self.reducer = IncrementalPCA(n_components=self.n_components)
        else:
            self.reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_seed)

    def fit(self, X):
        return self.reducer.fit(X)

    def transform(self, X):
        return self.reducer.transform(X)