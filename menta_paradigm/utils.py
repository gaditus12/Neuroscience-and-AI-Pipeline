# combat_transformer.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from neuroHarmonize.harmonizationLearn import harmonizationLearn
from neuroHarmonize.harmonizationApply import harmonizationApply

class CombatTransformer(BaseEstimator, TransformerMixin):
    """Batch-harmonise numeric features with neuroHarmonize ComBat.

    Parameters
    ----------
    feature_cols : list[str]
        Names of columns to harmonise.
    site_col : str, default='session'
        Column that defines the batch / site / session.
    """

    def __init__(self, feature_cols, site_col="session"):
        self.feature_cols = feature_cols
        self.site_col = site_col
        self.model_ = None        # trained ComBat model

    # ---------------------------------------------------------------------
    def fit(self, X, y=None):
        covars = pd.DataFrame({"SITE": X[self.site_col].values})
        data   = X[self.feature_cols].values
        self.model_, _ = harmonizationLearn(data, covars, eb=True)
        return self

    # ---------------------------------------------------------------------
    def transform(self, X):
        data = X[self.feature_cols].values
        adj  = harmonizationApply(data, self.model_)
        X_out = X.copy()
        X_out[self.feature_cols] = adj
        return X_out
