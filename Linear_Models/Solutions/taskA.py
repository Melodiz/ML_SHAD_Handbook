import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class BaseDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]] = None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.needed_columns = needed_columns
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        if self.needed_columns is not None:
            data = data[self.needed_columns]
        self.scaler.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        if self.needed_columns is not None:
            data = data[self.needed_columns]
        data = self.scaler.transform(data)
        return data
