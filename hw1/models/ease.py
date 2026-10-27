import numpy as np


class EASE:
    def __init__(self, regularization=200):
        self.regularization = regularization

    def fit(self, interactions):
        x = interactions.tocsr().astype(float)
        gram = (x.T @ x).toarray()
        gram.flat[::len(gram) + 1] += self.regularization
        inverse = np.linalg.inv(gram)
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0)
        self.interactions = x
        return self

    def score(self, user):
        return np.asarray(self.interactions[user] @ self.weights).ravel()
