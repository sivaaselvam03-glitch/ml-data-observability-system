"""
=============================================================
MODULE: ingestion/data_ingestion.py
PURPOSE: Load CSV, add ingestion timestamp, simulate daily
         snapshots, handle corrupt/missing files.
=============================================================
"""

import os
import shutil
import logging
import pandas as pd
from datetime import datetime, timedelta
import random

# ── Logger ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 1. LOAD CSV
# ─────────────────────────────────────────────────────────
def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file safely.
    Returns a pandas DataFrame or raises an informative error.

    Parameters
    ----------
    filepath : str
        Full or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        logger.error(f"File NOT found: {filepath}")
        raise FileNotFoundError(f"CSV file missing: {filepath}")

    if os.path.getsize(filepath) == 0:
        logger.error(f"File is empty: {filepath}")
        raise ValueError(f"CSV file is empty: {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded '{filepath}' → {df.shape[0]} rows × {df.shape[1]} cols")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"CSV parse error: {e}")
        raise ValueError(f"Corrupted CSV file: {filepath}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        raise


# ─────────────────────────────────────────────────────────
# 2. ADD INGESTION TIMESTAMP
# ─────────────────────────────────────────────────────────
def add_ingestion_timestamp(df: pd.DataFrame,
                             timestamp: datetime = None) -> pd.DataFrame:
    """
    Add an 'ingestion_timestamp' column to the DataFrame.

    Parameters
    ----------
    df        : pd.DataFrame
    timestamp : datetime (optional) – defaults to now()

    Returns
    -------
    pd.DataFrame with new column
    """
    ts = timestamp or datetime.now()
    df = df.copy()
    df["ingestion_timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Added ingestion_timestamp = {df['ingestion_timestamp'].iloc[0]}")
    return df


# ─────────────────────────────────────────────────────────
# 3. SIMULATE DAILY SNAPSHOTS
# ─────────────────────────────────────────────────────────
def simulate_daily_snapshots(
    source_filepath: str,
    snapshots_dir: str,
    num_days: int = 10,
    base_date: datetime = None,
    inject_anomalies: bool = True
) -> list:
    """
    Simulate 'num_days' daily CSV snapshots from the source file.
    Each day slightly modifies row count / values to mimic real drift.

    Parameters
    ----------
    source_filepath : str   – original CSV
    snapshots_dir   : str   – folder to save snapshot files
    num_days        : int   – how many daily snapshots to create
    base_date       : datetime – starting date (default: today - num_days)
    inject_anomalies: bool  – if True, inject bad data on day 7 & 9

    Returns
    -------
    list of snapshot file paths
    """
    os.makedirs(snapshots_dir, exist_ok=True)
    base_df = load_csv(source_filepath)
    base_date = base_date or (datetime.now() - timedelta(days=num_days))

    snapshot_paths = []

    for i in range(num_days):
        day = base_date + timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")

        # ── Simulate daily slice (random ~400-600 rows per day) ──
        sample_size = random.randint(400, 600)
        df_day = base_df.sample(n=min(sample_size, len(base_df)),
                                replace=True,
                                random_state=i).copy()
        df_day.reset_index(drop=True, inplace=True)

        # Re-assign Order IDs to be unique per snapshot
        df_day["Order ID"] = range(10001, 10001 + len(df_day))

        # ── Inject anomalies on specific days ──────────────────
        if inject_anomalies:
            # Day 7: large volume spike
            if i == 6:
                extra = base_df.sample(n=300, replace=True, random_state=99)
                df_day = pd.concat([df_day, extra], ignore_index=True)
                logger.info(f"[DAY {i+1}] Injected volume spike: {len(df_day)} rows")

            # Day 9: inject nulls + duplicate rows
            if i == 8:
                null_idx = df_day.sample(frac=0.05, random_state=42).index
                df_day.loc[null_idx, "Customer Name"] = None
                df_day.loc[null_idx, "Sales"] = None
                # Add 30 duplicate rows
                dupes = df_day.sample(n=30, replace=True, random_state=7)
                df_day = pd.concat([df_day, dupes], ignore_index=True)
                logger.info(f"[DAY {i+1}] Injected nulls + duplicates")

            # Day 10: inject outlier Sales values
            if i == 9:
                outlier_idx = df_day.sample(n=10, random_state=55).index
                df_day.loc[outlier_idx, "Sales"] = 9999999.0
                df_day.loc[outlier_idx, "Profit"] = -999999.0
                logger.info(f"[DAY {i+1}] Injected outlier values")

        # ── Add ingestion timestamp ─────────────────────────────
        df_day = add_ingestion_timestamp(df_day, timestamp=day)

        # ── Save snapshot ────────────────────────────────────────
        filename = f"snapshot_{day_str}.csv"
        filepath = os.path.join(snapshots_dir, filename)
        df_day.to_csv(filepath, index=False)
        snapshot_paths.append(filepath)
        logger.info(f"Saved snapshot: {filepath} ({len(df_day)} rows)")

    logger.info(f"Created {len(snapshot_paths)} snapshots in '{snapshots_dir}'")
    return snapshot_paths


# ─────────────────────────────────────────────────────────
# 4. LOAD LATEST SNAPSHOT
# ─────────────────────────────────────────────────────────
def load_latest_snapshot(snapshots_dir: str) -> pd.DataFrame:
    """
    Load the most recent snapshot CSV from the snapshots directory.

    Parameters
    ----------
    snapshots_dir : str

    Returns
    -------
    pd.DataFrame
    """
    files = sorted([
        f for f in os.listdir(snapshots_dir)
        if f.startswith("snapshot_") and f.endswith(".csv")
    ])
    if not files:
        raise FileNotFoundError(f"No snapshots found in: {snapshots_dir}")

    latest = os.path.join(snapshots_dir, files[-1])
    logger.info(f"Loading latest snapshot: {latest}")
    return load_csv(latest)


# ─────────────────────────────────────────────────────────
# 5. LOAD PREVIOUS SNAPSHOT (for schema/drift comparison)
# ─────────────────────────────────────────────────────────
def load_previous_snapshot(snapshots_dir: str,
                            steps_back: int = 1) -> pd.DataFrame:
    """
    Load the snapshot N steps before the latest one.

    Parameters
    ----------
    snapshots_dir : str
    steps_back    : int – 1 = one day ago, 2 = two days ago

    Returns
    -------
    pd.DataFrame or None if not enough snapshots
    """
    files = sorted([
        f for f in os.listdir(snapshots_dir)
        if f.startswith("snapshot_") and f.endswith(".csv")
    ])
    idx = -(steps_back + 1)
    if len(files) < steps_back + 1:
        logger.warning("Not enough snapshots for comparison.")
        return None

    prev = os.path.join(snapshots_dir, files[idx])
    logger.info(f"Loading previous snapshot: {prev}")
    return load_csv(prev)


# ─────────────────────────────────────────────────────────
# QUICK TEST  (run this file directly to verify)
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH   = os.path.join(BASE_DIR, "data", "Ecommerce_Sales_Data_2024_2025.csv")
    SNAP_DIR   = os.path.join(BASE_DIR, "data", "snapshots")

    paths = simulate_daily_snapshots(CSV_PATH, SNAP_DIR, num_days=10)
    df    = load_latest_snapshot(SNAP_DIR)
    print(f"\nLatest snapshot shape: {df.shape}")
    print(df.head(3))
