"""
Loads UCR/aeon .arff datasets, encodes labels, optionally merges and re-splits train/test,
and provides the stratified train/validation split
"""
import json

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from aeon.datasets import load_from_arff_file


def _load_single_arff_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".arff":
        X, y = load_from_arff_file(str(path))
        X = X.squeeze(1)  # (N, 1, seq_len) -> (N, seq_len)
        return X.astype(np.float32), y
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def load_dataset(name: str, data_dir: str, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expected naming convention for .arff files in `data_dir`:
        {name}_TRAIN.arff
        {name}_TEST.arff
    Last column: label

    Returns X_train, y_train, X_test, y_test (X: float32, y: int64 (encoded labels))
    """
    data_dir = Path(data_dir)

    train_path = data_dir / f"{name}_TRAIN.arff"
    test_path = data_dir / f"{name}_TEST.arff"

    X_train, y_train = _load_single_arff_file(train_path)
    X_test, y_test = _load_single_arff_file(test_path)

    encoder = LabelEncoder()
    y = np.concatenate([y_train, y_test])
    encoder.fit(y)

    original_to_encoded = {
      str(original): int(encoded)
      for original, encoded in zip(encoder.classes_, encoder.transform(encoder.classes_))
    }

    mapping_data = {
      "dataset": name,
      "original_to_encoded": original_to_encoded
    }

    mapping_file = data_dir / f"{name}_label_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
      json.dump(mapping_data, f, indent=4, ensure_ascii=False)

    y_train = encoder.transform(y_train).astype(np.int64)
    y_test = encoder.transform(y_test).astype(np.int64)

    return X_train, y_train, X_test, y_test


def load_dataset_merged(
    name: str, data_dir: str, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge the training and test datasets, then split them again into (X_train, y_train, X_test, y_test)based on test_size
    """
    X_train, y_train, X_test, y_test = load_dataset(name=name, data_dir=data_dir, test_size=test_size, seed=seed)
    X_all = np.concatenate([X_train, X_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    _, counts = np.unique(y_all, return_counts=True)
    can_stratify = (len(counts)) > 0 and (counts.min() >= 2)  # to manage classes that contain singletons

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all if can_stratify else None
    )
    return X_train2, y_train2, X_test2, y_test2


def train_valid_split(X: np.ndarray, y: np.ndarray, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """80/20 split, used only on the Deep Learning side"""
    _, counts = np.unique(y, return_counts=True)
    can_stratify = (len(counts) > 0) and (counts.min() >= 2)  # to manage classes that contain singletons
    return train_test_split(X, y, test_size=val_size, random_state=seed, stratify=y if can_stratify else None)
