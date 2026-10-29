"""
data_loader.py — Load REFIT raw and cleaned CSV files.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from src.config import RAW_DIR, CLEANED_DIR

logger = logging.getLogger(__name__)


def load_raw_house(house_id: int, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate Part1 + Part2 raw CSVs for a given house.

    Parameters
    ----------
    house_id : int
        REFIT house number (1‒21, skipping 14).
    data_dir : Path, optional
        Override for the raw data directory.

    Returns
    -------
    pd.DataFrame with parsed datetime index and all power columns.
    """
    d = data_dir or RAW_DIR

    # Handle inconsistent naming (House1 vs House_13)
    patterns = [
        f"RAW_House{house_id}_Part*.csv",
        f"RAW_House_{house_id}_Part*.csv",
    ]

    files = []
    for pat in patterns:
        files.extend(sorted(d.glob(pat)))

    if not files:
        raise FileNotFoundError(
            f"No raw CSV files found for House {house_id} in {d}"
        )

    logger.info("Loading %d raw file(s) for House %d: %s",
                len(files), house_id, [f.name for f in files])

    dfs = []
    for fp in files:
        df = pd.read_csv(
            fp,
            parse_dates=["Time"],
            low_memory=False,
        )
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined.rename(columns={"Time": "timestamp"}, inplace=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined.set_index("timestamp", inplace=True)
    combined.sort_index(inplace=True)

    # Drop the Unix column — we use the parsed timestamp
    if "Unix" in combined.columns:
        combined.drop(columns=["Unix"], inplace=True)

    logger.info("House %d raw: %d rows, columns %s, range %s to %s",
                house_id, len(combined), list(combined.columns),
                combined.index.min(), combined.index.max())
    return combined


def load_cleaned_house(house_id: int, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the officially cleaned CSV for a given house.

    Parameters
    ----------
    house_id : int
    data_dir : Path, optional

    Returns
    -------
    pd.DataFrame with parsed datetime index.
    """
    d = data_dir or CLEANED_DIR
    fp = d / f"CLEAN_House{house_id}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Cleaned file not found: {fp}")

    logger.info("Loading cleaned data for House %d from %s", house_id, fp.name)

    df = pd.read_csv(
        fp,
        parse_dates=["Time"],
        low_memory=False,
    )
    df.rename(columns={"Time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if "Unix" in df.columns:
        df.drop(columns=["Unix"], inplace=True)

    logger.info("House %d cleaned: %d rows, range %s to %s",
                house_id, len(df), df.index.min(), df.index.max())
    return df
