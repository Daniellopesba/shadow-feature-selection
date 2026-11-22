"""Input handling shared by the selectors.

Both selectors accept what a data scientist actually has: NumPy arrays, SciPy
sparse matrices, and pandas DataFrames with missing values, booleans, nullable
integer columns and categoricals. Numeric input is kept in its own floating
dtype (float32 stays float32); anything the downstream learner can read is
passed through unchanged.
"""

from __future__ import annotations

import inspect

import numpy as np
from scipy import sparse
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, column_or_1d

_FINITE_KW = (
    "ensure_all_finite"
    if "ensure_all_finite" in inspect.signature(check_array).parameters
    else "force_all_finite"
)


def is_dataframe(X) -> bool:
    return hasattr(X, "columns") and hasattr(X, "iloc") and hasattr(X, "dtypes")


def _is_numeric_numpy_dtype(dtype) -> bool:
    # plain NumPy numeric or bool dtypes (not pandas extension dtypes)
    return isinstance(dtype, np.dtype) and (dtype.kind in "biuf")


def _is_nullable_numeric(dtype) -> bool:
    # pandas nullable Int64 / Float64 / boolean
    import pandas as pd

    return isinstance(dtype, pd.api.extensions.ExtensionDtype) and (
        pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    )


def set_feature_names(est, X) -> None:
    """Record (or forget, on refit) string column names, as scikit-learn does."""
    names = None
    if is_dataframe(X):
        cols = np.asarray(X.columns, dtype=object)
        if len(cols) and all(isinstance(c, str) for c in cols):
            names = cols
    if names is not None:
        est.feature_names_in_ = names
    elif hasattr(est, "feature_names_in_"):
        del est.feature_names_in_


def check_target(y, n_samples: int):
    """Validate a classification target with at least two classes."""
    y = column_or_1d(y, warn=True)
    if len(y) != n_samples:
        raise ValueError(f"X has {n_samples} rows but y has {len(y)}")
    check_classification_targets(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("y must contain at least two classes; got 1 class")
    return y, classes, counts


def prepare_X(X):
    """Return ``(X, kind)`` with ``kind`` in ``{"dense", "sparse", "frame"}``.

    * ``"frame"``: a pandas DataFrame with at least one non-numeric column
      (category, object, string). It is returned unchanged, with a fresh
      0..n-1 index, so that dtypes and categories reach the learner intact.
    * ``"dense"``: a 2-D floating ndarray (float32 kept, other numerics cast to
      float64). Numeric DataFrames, booleans and nullable columns land here,
      with ``pd.NA`` as ``NaN``.
    * ``"sparse"``: a CSC matrix.

    Missing values are allowed everywhere; infinite values are not.
    """
    if is_dataframe(X):
        dtypes = list(X.dtypes)
        if all(_is_numeric_numpy_dtype(t) or _is_nullable_numeric(t) for t in dtypes):
            if any(not isinstance(t, np.dtype) for t in dtypes):
                X = X.astype("float64").to_numpy(dtype=np.float64, na_value=np.nan)
            else:
                X = X.to_numpy()
        else:
            if X.shape[1] == 0 or X.shape[0] == 0:
                raise ValueError("X must have at least one row and one column")
            X = X.reset_index(drop=True)
            num = [c for c, t in X.dtypes.items() if _is_numeric_numpy_dtype(t)]
            if num:
                vals = X[num].to_numpy(dtype=float)
                if np.isinf(vals).any():
                    raise ValueError("Input X contains infinity")
            return X, "frame"
    if sparse.issparse(X):
        X = check_array(
            X,
            accept_sparse="csc",
            dtype=(np.float64, np.float32),
            **{_FINITE_KW: "allow-nan"},
        )
        return X, "sparse"
    if isinstance(X, np.ndarray) and X.dtype.kind == "b":
        X = X.astype(np.float64)
    X = check_array(X, dtype=(np.float64, np.float32), **{_FINITE_KW: "allow-nan"})
    return X, "dense"


def numeric_columns(X, kind):
    """Yield ``(j, values_or_None)``: float values for numeric columns, ``None``
    for categorical/object columns of a DataFrame."""
    import pandas as pd

    d = X.shape[1]
    for j in range(d):
        if kind == "dense":
            yield j, X[:, j]
        elif kind == "sparse":
            yield j, X[:, [j]].toarray().ravel()
        else:
            col = X.iloc[:, j]
            t = col.dtype
            if _is_numeric_numpy_dtype(t) or _is_nullable_numeric(t):
                yield j, pd.to_numeric(col).to_numpy(dtype=float, na_value=np.nan)
            else:
                yield j, None


def standardized_mean_difference(X, kind, minority_mask) -> np.ndarray:
    """``(mean_minority - mean_rest) / pooled SD`` per column, ignoring NaN.

    ``NaN`` for categorical columns and for columns without variance.
    """
    d = X.shape[1]
    out = np.full(d, np.nan)
    pos = np.asarray(minority_mask, dtype=bool)
    for j, v in numeric_columns(X, kind):
        if v is None:
            continue
        a, b = v[pos], v[~pos]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        if sp2 > 0:
            out[j] = (a.mean() - b.mean()) / np.sqrt(sp2)
    return out


class FrameTransformMixin:
    """``transform`` that keeps DataFrames with categorical columns intact.

    scikit-learn's ``SelectorMixin.transform`` converts its input to an array,
    which fails on categorical columns. A DataFrame with any non-numeric
    column is instead returned as a column subset, with its dtypes.
    """

    def transform(self, X):
        if is_dataframe(X) and not all(
            _is_numeric_numpy_dtype(t) or _is_nullable_numeric(t) for t in X.dtypes
        ):
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(self)
            if X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                    f"{self.n_features_in_} features as input."
                )
            names = getattr(self, "feature_names_in_", None)
            if names is not None and list(X.columns) != list(names):
                raise ValueError(
                    "The feature names should match those that were passed during fit."
                )
            return X.iloc[:, self.get_support()]
        return super().transform(X)
