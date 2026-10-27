import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=80, shrink=20):
        self.neighbors = neighbors
        self.shrink = shrink

    def fit(self, interactions):
        x = interactions.tocsr().astype(float)
        norm = np.sqrt(np.asarray(x.power(2).sum(axis=0)).ravel())
        norm[norm == 0] = 1
        similarity = (x.T @ x).toarray()
        similarity /= norm[:, None] * norm[None, :] + self.shrink
        np.fill_diagonal(similarity, 0)
        n = similarity.shape[0]
        if self.neighbors < n:
            drop = np.argpartition(similarity, -self.neighbors, axis=1)[:, :-self.neighbors]
            similarity[np.arange(n)[:, None], drop] = 0
        self.similarity = sparse.csr_matrix(similarity)
        self.interactions = x
        return self

    def score(self, user):
        return np.asarray((self.interactions[user] @ self.similarity).toarray()).ravel()
