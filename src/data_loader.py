"""Load the NASA C-MAPSS text files (read-only; the raw files are never modified)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def raw_file(kind: str, subset: str):
    """Path of e.g. train_FD001.txt.  kind is 'train', 'test' or 'RUL'."""
    return cfg.RAW_DIR / f"{kind}_{subset}.txt"


def _check(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Copy the NASA C-MAPSS files (train_FDxxx.txt, test_FDxxx.txt, "
            f"RUL_FDxxx.txt) into: {cfg.RAW_DIR}"
        )
    return path


def load_train(subset: str) -> pd.DataFrame:
    """Training trajectories (run-to-failure). 26 space separated columns, no header."""
    return pd.read_csv(_check(raw_file("train", subset)), sep=r"\s+", header=None,
                       names=cfg.RAW_COLUMNS)


def load_test(subset: str) -> pd.DataFrame:
    """Test trajectories (stop some time *before* failure)."""
    return pd.read_csv(_check(raw_file("test", subset)), sep=r"\s+", header=None,
                       names=cfg.RAW_COLUMNS)


def load_rul_truth(subset: str) -> pd.Series:
    """Official true RUL at the last recorded cycle of every test engine.

    Row i of RUL_FDxxx.txt belongs to test engine i (1-based)."""
    values = pd.read_csv(_check(raw_file("RUL", subset)), header=None)[0].to_numpy()
    return pd.Series(values, index=pd.RangeIndex(1, len(values) + 1, name=cfg.ID_COL),
                     name="rul_true")


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """RUL of a training row = (last cycle of that engine) - (current cycle)."""
    out = df.copy()
    out["rul"] = out.groupby(cfg.ID_COL)[cfg.CYCLE_COL].transform("max") - out[cfg.CYCLE_COL]
    return out


def add_test_rul(test: pd.DataFrame, rul_truth: pd.Series) -> pd.DataFrame:
    """Ground-truth RUL for *every* test row.

    The official file gives the RUL after the last test cycle; because cycles are
    consecutive, RUL at an earlier cycle = RUL_last + (last_cycle - cycle).
    """
    out = test.copy()
    last = out.groupby(cfg.ID_COL)[cfg.CYCLE_COL].transform("max")
    out["rul"] = out[cfg.ID_COL].map(rul_truth) + (last - out[cfg.CYCLE_COL])
    if out["rul"].isna().any():
        raise ValueError("RUL file does not cover every test engine.")
    return out


def cap_rul(rul, cap: int = cfg.RUL_CAP):
    """Piece-wise linear RUL target: min(RUL, cap)."""
    return np.minimum(rul, cap)


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Only the last recorded cycle of every engine (what the NASA challenge scores)."""
    return df.loc[df.groupby(cfg.ID_COL)[cfg.CYCLE_COL].idxmax()]
