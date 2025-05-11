"""
Clustering module for Survey Vector Pipeline.
Implements KMeans (default) and HDBSCAN, with optional StandardScaler.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

class Clusterer:
    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

class KMeansClusterer(Clusterer):
    def __init__(self, config):
        self.n_clusters = config.get("n_clusters", 10)
        self.random_seed = config.get("random_seed", 42)
        self.scaler = None
        if config.get("scaler", "none") == "standard":
            self.scaler = StandardScaler(with_mean=False)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_seed)

    def fit(self, X):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.kmeans.fit(X)
        return self

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.kmeans.predict(X)

class HDBSCANClusterer(Clusterer):
    def __init__(self, config):
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan is not installed.")
        self.min_cluster_size = config.get("min_cluster_size", 5)
        self.scaler = None
        if config.get("scaler", "none") == "standard":
            self.scaler = StandardScaler(with_mean=False)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)

    def fit(self, X):
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.hdbscan.fit(X)
        return self

    def predict(self, X):
        if self.scaler:
            X = self.scaler.transform(X)
        return self.hdbscan.predict(X)