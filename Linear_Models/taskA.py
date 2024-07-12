import numpy as np
import pandas as pd
from typing import Optional, List
from sklearn.model_selection import train_test_split

import sklearn.base
# import standartScaler
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
seed = 24

# data = pd.read_csv('./data.csv')
# target_column = "Sale_Price"
# np.random.seed(seed)

# test_size = 0.2
# data_train, data_test, Y_train, Y_test = train_test_split(
#     data[data.columns.drop("Sale_Price")],
#     np.array(data["Sale_Price"]),
#     test_size=test_size,
#     random_state=seed)

# print(f"Train : {data_train.shape} {Y_train.shape}")
# print(f"Test : {data_test.shape} {Y_test.shape}")

# continuous_columns = [key for key in data.keys(
# ) if data[key].dtype in ("int64", "float64")]
# categorical_columns = [
#     key for key in data.keys() if data[key].dtype == "object"]

# continuous_columns.remove(target_column)

# print(f"Continuous : {len(continuous_columns)}, Categorical : {len(categorical_columns)}")


class BaseDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, needed_columns: Optional[List[str]] = None):
        """
        :param needed_columns: if not None select these columns from the dataframe
        """
        self.needed_columns = needed_columns if needed_columns else data.columns.to_list()
        self.scaler = StandardScaler()

    def fit(self, data, *args):
        """
        Prepares the class for future transformations
        :param data: pd.DataFrame with all available columns
        :return: self
        """
        # Your code here
        data = data[self.needed_columns] # select only needed columns
        self.scaler.fit(data) # fit scaler to continuous columns
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transforms features so that they can be fed into the regressors
        :param data: pd.DataFrame with all available columns
        :return: np.array with preprocessed features
        """
        # Your code here
        data = data[self.needed_columns] # select only needed columns
        data[continuous_columns] = self.scaler.transform(data[continuous_columns]) # scale continuous columns
        return data.values # return np.array

# preprocessor = BaseDataPreprocessor(needed_columns=continuous_columns)

# X_train = preprocessor.fit_transform(data_train)
# X_test = preprocessor.transform(data_test)