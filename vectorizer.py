"""
Vectorization module for Survey Vector Pipeline.
Implements a pluggable Vectorizer interface with HashingVectorizer as default.
"""

from sklearn.feature_extraction.text import HashingVectorizer

class Vectorizer:
    def fit(self, texts):
        pass  # Stateless for HashingVectorizer

    def transform(self, texts):
        raise NotImplementedError

class HashingVectorizerWrapper(Vectorizer):
    def __init__(self, config):
        self.n_features = config.get("n_features", 2**18)
        self.ngram_range = tuple(config.get("ngram_range", [1, 2]))
        self.binary = config.get("binary", False)
        # hash_seed is only supported in scikit-learn >=1.2
        # For compatibility, we omit it here
        self.vectorizer = HashingVectorizer(
            n_features=self.n_features,
            ngram_range=self.ngram_range,
            binary=self.binary,
            alternate_sign=False,
            norm='l2',
            input='content',
            dtype=float,
            lowercase=True,
            analyzer='word',
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b"
        )

    def fit(self, texts):
        # No fitting needed for HashingVectorizer
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)