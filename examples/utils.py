import enum
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer


class DataSet(enum.Enum):
    Train = "Train"
    Valid = "Valid"
    Test = "Test"
    

def get_dataset(dataset: DataSet) -> pd.DataFrame:
    if dataset == DataSet.Train:
        return pd.read_csv('TRAIN.csv', sep='\t')
    elif dataset == DataSet.Valid:
        return pd.read_csv('VALID.csv', sep='\t')
    elif dataset == DataSet.Test:
        return pd.read_csv('TEST.csv', sep='\t')
    else:
        raise ValueError('Wrong argument.')

        
def cost(true: np.ndarray, pred: np.ndarray) -> float:
    return np.sqrt(
        np.mean(
            (np.log1p(true) - np.log1p(pred)) ** 2
        )
    )

class MercariTranformer:
    def __init__(
        self,
        vectorizer_name,
        vectorizer_desc,
    ):
        self._vectorizer_name = vectorizer_name
        self._vectorizer_desc = vectorizer_desc
        self._vectorizer_brand = LabelBinarizer(sparse_output=True)
        self._vectorizer_cond = LabelBinarizer()
        self._unique_categories = []
        self._category_map = {}
        self._m_category = None
        
    def fit(self, df: pd.DataFrame):
        self._vectorizer_name.fit(df["name"].fillna(""))        
        self._vectorizer_desc.fit(df["item_description"].fillna(""))
        self._vectorizer_brand.fit(df["brand_name"].fillna("Missing"))
        self._vectorizer_cond.fit(df["item_condition_id"])
        
        self._unique_categories = reduce(
            lambda x, y: set(x).union(y),
            df.loc[~df["category_name"].isnull(), "category_name"].str.split("/").values
        )
        self._category_map = dict([
            (c, i) for i, c in enumerate(self._unique_categories, 1)
        ])
        self._m_category = df["category_name"].fillna("").str.split("/").apply(len).max()
        
    def transform(self, df):
        return {
            "name": self._vectorizer_name.transform(df["name"].fillna("")),
            "item_description": self._vectorizer_desc.transform(df["item_description"].fillna('')),
            "brand_name": self._vectorizer_brand.transform(df["brand_name"].fillna("Missing")),
            "item_condition_id": self._vectorizer_cond.transform(df["item_condition_id"]),
            "shipping": df["shipping"],
            "category_name": np.vstack(self._transform_category(df))
        }
    
    def _transform_category(self, df):
        def f(lst):
            v = [0] * self._m_category
            for i, x in enumerate(lst):
                if x in self._category_map:
                    v[i] = self._category_map[x]
            return v
        return df["category_name"].fillna("").str.split("/").apply(f)
