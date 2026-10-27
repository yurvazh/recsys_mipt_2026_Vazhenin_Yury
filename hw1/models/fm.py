import numpy as np


class FactorizationMachine:
    def __init__(self, factors=12, learning_rate=0.01, regularization=0.001,
                 epochs=5, batch_size=2048, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, features, targets):
        x = features.tocsr().astype(np.float64)
        y = np.asarray(targets, dtype=np.float64).ravel()
        if len(y) != x.shape[0] or not len(y):
            raise ValueError('Feature rows must match non-empty targets')
        rng = np.random.default_rng(self.seed)
        self.bias = float(y.mean())
        self.linear = np.zeros(x.shape[1], dtype=np.float64)
        self.latent = rng.normal(0, 0.03, (x.shape[1], self.factors))
        linear_m = np.zeros_like(self.linear)
        linear_v = np.zeros_like(self.linear)
        latent_m = np.zeros_like(self.latent)
        latent_v = np.zeros_like(self.latent)
        bias_m = 0.0
        bias_v = 0.0
        step = 0
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        for _ in range(self.epochs):
            permutation = rng.permutation(x.shape[0])
            for offset in range(0, len(permutation), self.batch_size):
                indices = permutation[offset:offset + self.batch_size]
                batch = x[indices]
                squared = batch.multiply(batch)
                latent_sum = np.asarray(batch @ self.latent)
                latent_squares = np.asarray(squared @ (self.latent ** 2))
                interaction = 0.5 * np.sum(
                    latent_sum ** 2 - latent_squares, axis=1
                )
                predictions = self.bias + batch @ self.linear + interaction
                errors = np.clip(predictions - y[indices], -10, 10)
                count = len(indices)
                bias_gradient = float(errors.mean())
                linear_gradient = (
                    np.asarray(batch.T @ errors).ravel() / count
                    + self.regularization * self.linear
                )
                latent_gradient = (
                    np.asarray(batch.T @ (errors[:, None] * latent_sum))
                    - np.asarray(squared.T @ errors)[:, None] * self.latent
                ) / count + self.regularization * self.latent

                step += 1
                bias_m = beta1 * bias_m + (1 - beta1) * bias_gradient
                bias_v = beta2 * bias_v + (1 - beta2) * bias_gradient ** 2
                linear_m = beta1 * linear_m + (1 - beta1) * linear_gradient
                linear_v = beta2 * linear_v + (1 - beta2) * linear_gradient ** 2
                latent_m = beta1 * latent_m + (1 - beta1) * latent_gradient
                latent_v = beta2 * latent_v + (1 - beta2) * latent_gradient ** 2

                correction_m = 1 - beta1 ** step
                correction_v = 1 - beta2 ** step
                rate = self.learning_rate * np.sqrt(correction_v) / correction_m
                self.bias -= rate * bias_m / (np.sqrt(bias_v) + epsilon)
                self.linear -= rate * linear_m / (np.sqrt(linear_v) + epsilon)
                self.latent -= rate * latent_m / (np.sqrt(latent_v) + epsilon)
        return self

    def predict(self, features):
        x = features.tocsr()
        linear = np.asarray(x @ self.linear).ravel()
        first = np.asarray(x @ self.latent)
        second = np.asarray(x.power(2) @ (self.latent ** 2))
        return self.bias + linear + 0.5 * np.sum(first ** 2 - second, axis=1)

    def set_features(self, feature_builder):
        self.feature_builder = feature_builder
        return self

    def score(self, user):
        items = np.arange(self.feature_builder.n_items)
        users = np.full(len(items), user)
        return self.predict(self.feature_builder.transform(users, items))
