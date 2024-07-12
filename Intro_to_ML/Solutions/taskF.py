# MostFrequentClassifier 
 
from sklearn.base import ClassifierMixin 
import numpy as np 
from scipy.stats import mode 
 
class MostFrequentClassifier(ClassifierMixin): 
    def fit(self, X, y): 
        self.mode_ = mode(y)[0] 
        self.is_fitted_ = True 
        return self 
 
    def predict(self, X): 
        return np.full(shape=X.shape[0], fill_value=self.mode_)