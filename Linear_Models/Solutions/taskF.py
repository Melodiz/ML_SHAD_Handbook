import numpy as np
from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1., delta_converged=1e-3, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.W = np.zeros(X.shape[1])
        self.b = 0.0

        steps_counter = 0
        while steps_counter < self.max_steps:
            indices = np.random.choice(X.shape[0], size=self.batch_size, replace=False)
            X_batch = X[indices]
            Y_batch = Y[indices]
            self.W_prev = self.W.copy()

            gradient_b = (2 / self.batch_size) * np.sum(X_batch @ self.W + self.b - Y_batch)
            gradient_W = (2 / self.batch_size) * X_batch.T @ (X_batch @ self.W + self.b - Y_batch) + 2 * self.regularization * self.W

            self.W -= self.lr * gradient_W
            self.b -= self.lr * gradient_b

            steps_counter += 1
            if np.sqrt(np.sum((self.W - self.W_prev) ** 2)) < self.delta_converged:
                break
        return self


    def predict(self, X):
        return X @ self.W + self.b