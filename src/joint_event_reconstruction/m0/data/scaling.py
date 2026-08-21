import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, offset: float = 1e-3):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log(X + self.offset)

class SinCosScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sin_transformed = np.sin(X)
        cos_transformed = np.cos(X)
        return np.hstack((sin_transformed, cos_transformed))

class PIDScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.id_mapping = {1 : "JET", 2: "ELECTRON", 3: "MUON", 4: "MET", np.nan: "UNKNOWN"}
        self.one_hot_mapping = {
            "JET": [1, 0, 0, 0],
            "ELECTRON": [0, 1, 0, 0],
            "MUON": [0, 0, 1, 0],
            "MET": [0, 0, 0, 1],
            "UNKNOWN": [0, 0, 0, 0]
        }

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        transformed = []
        for val in X.flatten():
            if val in self.id_mapping:
                transformed.append(self.one_hot_mapping[self.id_mapping[val]])
            elif math.isnan(val):
                transformed.append(self.one_hot_mapping["UNKNOWN"])
            else:
                raise ValueError(f"Unknown particle type '{val}' for PIDScaler")
        return np.array(transformed)

class StringScaler(BaseEstimator, TransformerMixin):
    def __init__(self, expression: str):
        self.expression = expression

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Create a safe evaluation environment
        allowed_names = {"x": X}
        return eval(self.expression, {"__builtins__": None}, allowed_names)


def get_transformer_by_name(name: str):
    normalized = name.strip().lower()
    if normalized == "log":
        return LogScaler()
    if normalized == "sincos":
        return SinCosScaler()
    if normalized == "standard":
        return StandardScaler()
    if normalized == "minmax":
        return MinMaxScaler()
    if normalized == "pid":
        return PIDScaler()
    if normalized == "none":
        return "passthrough"
    try: # Try to evaluate simple scaling expressions like "x/4"
        return StringScaler(normalized)
    except Exception:
        raise ValueError(f"Unsupported transformation '{name}'")


def build_feature_scaler(
    feature_key_column_dict: dict[str, int],
    scaler_column_dict: dict[str, str]
) -> tuple[ColumnTransformer, list[str]]:
    transformers = []

    for feature_key, column_idx in feature_key_column_dict.items():
        try:
            transformation_name = scaler_column_dict[feature_key]
        except KeyError:
            raise ValueError(f"Feature key '{feature_key}' not found in scaler column dictionary.")
        name = f"obs_{feature_key}"
        transformers.append((name, get_transformer_by_name(transformation_name), [column_idx]))
        column_idx += 1

    return ColumnTransformer(transformers, remainder="drop")
