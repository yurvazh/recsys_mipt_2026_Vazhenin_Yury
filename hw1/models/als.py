import numpy as np


class ExplicitALS:
    def __init__(self, factors=32, regularization=0.5, iterations=8, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.seed = seed

    def fit(self, interactions):
        x = interactions.tocsr().astype(float)
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.1, (x.shape[0], self.factors))
        self.item_factors = rng.normal(0, 0.1, (x.shape[1], self.factors))
        rows = x.tocsr()
        cols = x.tocsc()
        identity = np.eye(self.factors)
        for _ in range(self.iterations):
            for u in range(x.shape[0]):
                start, end = rows.indptr[u:u + 2]
                ids = rows.indices[start:end]
                if not len(ids):
                    continue
                vectors = self.item_factors[ids]
                target = rows.data[start:end]
                self.user_factors[u] = np.linalg.solve(
                    vectors.T @ vectors + self.regularization * len(ids) * identity,
                    vectors.T @ target)
            for i in range(x.shape[1]):
                start, end = cols.indptr[i:i + 2]
                ids = cols.indices[start:end]
                if not len(ids):
                    continue
                vectors = self.user_factors[ids]
                target = cols.data[start:end]
                self.item_factors[i] = np.linalg.solve(
                    vectors.T @ vectors + self.regularization * len(ids) * identity,
                    vectors.T @ target)
        return self

    def score(self, user):
        return self.user_factors[user] @ self.item_factors.T
