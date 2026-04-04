"""
io.py
File I/O utilities for loading and saving datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")
    return pd.read_csv(path, **read_csv_kwargs)


def save_df(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    ensure_parent_dir(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except Exception:
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=index)
            return fallback

    if suffix == ".csv":
        df.to_csv(path, index=index)
        return path

    raise ValueError(f"Unsupported file extension: {suffix}")
