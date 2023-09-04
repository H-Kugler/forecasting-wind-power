from xgboost import XGBRegressor
import pandas as pd
from typing import Literal
from src.Models.basemodel import Basemodel


class XGB(Basemodel):

    def __init__(self):
        self.model = XGBRegressor()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        n_samples = X.shape[0]
        y = y.iloc[-n_samples:]
        # check if shapes match
        assert X.shape[0] == y.shape[0]
        self.model.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
    


        