"""
Registry of scalers (standard, minmax, none) applied to the embedding space
"""
from typing import Optional

from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCALER_REGISTRY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "none": None,
}


def get_scaler(name: str) -> Optional[object]:
    if name not in SCALER_REGISTRY:
        raise ValueError(f"Unknown scaler: '{name}'. Options: {list(SCALER_REGISTRY)}")
    cls = SCALER_REGISTRY[name]
    return cls() if cls is not None else None
