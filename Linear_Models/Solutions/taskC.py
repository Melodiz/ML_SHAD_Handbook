import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge

class ExponentialLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, **ridge_params):
        self.linear_model = Ridge(**ridge_params)  # Initialize Ridge regressor with parameters

    def fit(self, X, Y):
        # Apply logarithmic transformation to target
        Y_log = np.log(Y)
        self.linear_model.fit(X, Y_log)
        return self

    def predict(self, X):
        # Apply logarithmic transformation to target
        Y_log = self.linear_model.predict(X)
        # Apply exponential transformation to get original target
        return np.exp(Y_log)

    def get_params(self, *args, **kwargs):
        # Return the parameters of the model
        return self.linear_model.get_params(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        # Set the parameters of the model
        self.linear_model.set_params(*args, **kwargs)
