import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

class BaseDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]] = None):
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, *args):
        if self.needed_columns is not None:
            data = data[self.needed_columns]
        self.scaler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        if self.needed_columns is not None:
            data = data[self.needed_columns]
        data = self.scaler.transform(data)
        return data

class OneHotPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode: Optional[List[str]] = None, continue_columns: Optional[List[str]] = None):
        self.columns_to_encode = columns_to_encode
        self.continue_columns = continue_columns
        self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, *args):
        if self.columns_to_encode is not None:
            self.encoder.fit(data[self.columns_to_encode])
        if self.continue_columns is not None:
            self.scaler.fit(data[self.continue_columns])
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        encoded_data = np.array([]).reshape(len(data), 0)
        if self.columns_to_encode is not None:
            encoded_data = self.encoder.transform(data[self.columns_to_encode])
        scaled_data = np.array([]).reshape(len(data), 0)
        if self.continue_columns is not None:
            scaled_data = self.scaler.transform(data[self.continue_columns])
        return np.hstack((encoded_data, scaled_data))

def make_ultimate_pipeline(continuous_columns, categorical_columns):
    base_preprocessor = BaseDataPreprocessor(needed_columns=continuous_columns)
    one_hot_preprocessor = OneHotPreprocessor(columns_to_encode=categorical_columns, continue_columns=continuous_columns)
    pipeline = Pipeline([
        ('preprocessor', one_hot_preprocessor),
        ('regressor', Ridge())
    ])
    return pipeline