# ---------------------------------------------------------------------------
#  leakage-safe ComBat that never crashes on missing batches
# ---------------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from neuroHarmonize.harmonizationLearn import harmonizationLearn
from neuroHarmonize.harmonizationApply import harmonizationApply


class CombatTransformer(BaseEstimator, TransformerMixin):
    """
    *   Fits ComBat on the training rows of a fold (no leakage)
    *   Transforms any subset of rows later **without IndexError**:
        we keep the full list of batch levels so the design matrix
        shape matches the model expectation.
    """

    def __init__(self, feature_cols, site_col="session"):
        self.feature_cols = feature_cols
        self.site_col = site_col
        self.model_ = None
        self._site_levels = None        # list[str]

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        # remember the complete set & order of batches
        self._site_levels = pd.Categorical(
            X[self.site_col]).categories.tolist()

        covars = pd.DataFrame(
            {"SITE": pd.Categorical(X[self.site_col],
                                    categories=self._site_levels)},
            index=X.index
        )
        data = X[self.feature_cols].values
        self.model_, _ = harmonizationLearn(data, covars, eb=True)
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X):
        if self.model_ is None:
            raise RuntimeError("CombatTransformer.transform() before fit()")

        # ① rebuild covariate table with *all* levels from training
        covars = pd.DataFrame(
            {"SITE": pd.Categorical(X[self.site_col],
                                    categories=self._site_levels)},
            index=X.index
        )

        # ② harmonise numeric features
        data = X[self.feature_cols].values
        adjusted = harmonizationApply(data, covars, self.model_)

        # ③ return a DataFrame that keeps the index and column names
        return pd.DataFrame(adjusted,
                            columns=self.feature_cols,
                            index=X.index)
