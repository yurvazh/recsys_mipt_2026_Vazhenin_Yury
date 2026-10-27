import numpy as np
from scipy import sparse

from .als import ExplicitALS


class TemporalTargetALS(ExplicitALS):
    def __init__(self, factors=32, regularization=0.5, iterations=8,
                 decay=0.4, seed=42):
        super().__init__(factors, regularization, iterations, seed)
        self.decay = decay

    def make_targets(self, frame, n_users, n_items):
        age_years = (
            (frame.timestamp.max() - frame.timestamp)
            .dt.total_seconds().to_numpy() / (365.25 * 86400)
        )
        targets = 1 + (frame.rating.to_numpy(dtype=float) - 1) * np.exp(
            -self.decay * age_years
        )
        return sparse.csr_matrix(
            (targets, (frame.u.to_numpy(), frame.i.to_numpy())),
            shape=(n_users, n_items),
        )

    def fit_frame(self, frame, n_users, n_items):
        return self.fit(self.make_targets(frame, n_users, n_items))
