# MeanRegressor

from sklearn.base import RegressorMixin
import numpy as np

class MeanRegressor(RegressorMixin):
   # Predicts the mean of y_train
    def fit(self, X, y):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Training data features
        y : array like, shape = (_samples,)
            Training data targets
        '''
        self.mean_ = np.mean(y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
            Data to predict
        '''
        return np.full(shape=X.shape[0], fill_value=self.mean_)