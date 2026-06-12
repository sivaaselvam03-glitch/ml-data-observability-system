# ingestion/__init__.py
from .data_ingestion import (
    load_csv,
    add_ingestion_timestamp,
    simulate_daily_snapshots,
    load_latest_snapshot,
    load_previous_snapshot,
)
