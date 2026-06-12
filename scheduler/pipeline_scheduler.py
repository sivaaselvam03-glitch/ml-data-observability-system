

import os
import sys
import logging
import traceback
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ingestion.data_ingestion import (
    load_latest_snapshot,
    load_previous_snapshot
)
from validation.quality_checks import run_all_checks
from metrics.metrics_generator import extract_metrics, print_metrics_summary
from database.db_manager import (
    setup_database, insert_metrics, fetch_all_metrics
)
from ml_anomaly.anomaly_detector import run_ml_pipeline
from utils.logger import setup_logger

# ════════════════════════════════════════════════════════
# PATHS  ← Adjust if your folder layout differs
# ════════════════════════════════════════════════════════
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "data", "snapshots")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "ml_anomaly", "saved_model.pkl")
LOG_DIR       = os.path.join(PROJECT_ROOT, "logs")

logger = setup_logger("scheduler", LOG_DIR)



def run_pipeline() -> dict:
    """
    Execute the full data observability pipeline.
    Called by the scheduler every day (or manually).

    Returns
    -------
    dict with execution results
    """
    run_start = datetime.now()
    logger.info("=" * 60)
    logger.info(f"PIPELINE START  →  {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    result = {
        "status":        "success",
        "started_at":    run_start.isoformat(),
        "finished_at":   None,
        "metrics":       None,
        "ml_result":     None,
        "db_row_id":     None,
        "error":         None,
    }

    try:
        # ─── STEP 1: Load latest snapshot ─────────────────
        logger.info("STEP 1 → Loading latest snapshot...")
        df      = load_latest_snapshot(SNAPSHOTS_DIR)
        prev_df = load_previous_snapshot(SNAPSHOTS_DIR)
        logger.info(f"  Current snapshot : {len(df):,} rows")
        logger.info(f"  Previous snapshot: {len(prev_df):,} rows" if prev_df is not None else "  No previous snapshot")

        # ─── STEP 2: Run all quality checks ───────────────
        logger.info("STEP 2 → Running 21 quality checks...")
        check_results = run_all_checks(df, prev_df)
        passed = sum(1 for r in check_results if r["passed"])
        failed = len(check_results) - passed
        logger.info(f"  Results: {passed} passed, {failed} failed")

        for r in check_results:
            status = "PASS" if r["passed"] else "FAIL"
            logger.info(f"  [{status}] {r['check']}")

        # ─── STEP 3: Generate metrics ──────────────────────
        logger.info("STEP 3 → Generating metrics...")
        metrics = extract_metrics(df, check_results, dataset_name="ecommerce_sales")
        result["metrics"] = metrics
        print_metrics_summary(metrics)

        # ─── STEP 4: Store in MySQL ────────────────────────
        logger.info("STEP 4 → Inserting metrics into MySQL...")
        db_row_id = insert_metrics(metrics)
        result["db_row_id"] = db_row_id
        logger.info(f"  Stored as row id = {db_row_id}")

        # ─── STEP 5: ML anomaly detection ─────────────────
        logger.info("STEP 5 → Running ML anomaly detection...")
        historical = fetch_all_metrics()
        ml_result  = run_ml_pipeline(metrics, historical, model_path=MODEL_PATH)
        result["ml_result"] = ml_result

        iso = ml_result.get("isolation_forest", {})
        logger.info(f"  IsolationForest: {iso.get('prediction_label','?')} "
                    f"| score={iso.get('anomaly_score', 0):.4f}")

        prophet = ml_result.get("prophet", {})
        if "forecasted_row_count" in prophet:
            logger.info(f"  Prophet forecast: {prophet['forecasted_row_count']:.0f} rows "
                        f"[{prophet['lower_bound']:.0f} – {prophet['upper_bound']:.0f}]")
            if prophet.get("anomaly_detected"):
                logger.warning("  ⚠️  Prophet anomaly detected!")

        # ─── STEP 6: Log summary ───────────────────────────
        logger.info("STEP 6 → Logging summary...")
        anomaly_combined = metrics["anomaly_flag"] or iso.get("anomaly_flag", 0)
        if anomaly_combined:
            logger.warning(
                f"🚨 ANOMALY DETECTED  | score={metrics['quality_score']} "
                f"| ml_score={iso.get('anomaly_score', 0):.4f}"
            )
        else:
            logger.info(
                f"✅ All clear  | quality_score={metrics['quality_score']} "
                f"| drift={metrics['drift_score']:.4f}"
            )

    except Exception as e:
        result["status"] = "failed"
        result["error"]  = str(e)
        logger.error(f"PIPELINE FAILED: {e}")
        logger.error(traceback.format_exc())

    finally:
        run_end = datetime.now()
        result["finished_at"] = run_end.isoformat()
        elapsed = (run_end - run_start).total_seconds()
        logger.info(f"PIPELINE END  →  elapsed={elapsed:.2f}s  status={result['status']}")
        logger.info("=" * 60)

    return result


# ════════════════════════════════════════════════════════
# SCHEDULER SETUP
# ════════════════════════════════════════════════════════
def start_scheduler(
    hour: int   = 8,
    minute: int = 0
) -> None:
    """
    Start the APScheduler to run the pipeline daily.

    Parameters
    ----------
    hour   : int – hour in 24h format  (default 8 = 8 AM)
    minute : int – minute              (default 0)
    """
    # Ensure DB is ready before first run
    logger.info("Ensuring MySQL database setup...")
    setup_database()

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    scheduler.add_job(
        func=run_pipeline,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="daily_data_quality_pipeline",
        name="Daily Data Observability Pipeline",
        replace_existing=True,
        misfire_grace_time=3600   # allow 1hr of misfire
    )

    logger.info(
        f"Scheduler started. Pipeline will run daily at "
        f"{hour:02d}:{minute:02d} IST. Press Ctrl+C to stop."
    )
    print(f"\n🕐  APScheduler running. Next job: daily at {hour:02d}:{minute:02d} IST")
    print("   Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")
        print("\nScheduler stopped.")


# ════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Data Observability Pipeline Scheduler"
    )
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run the pipeline once immediately and exit"
    )
    parser.add_argument(
        "--hour",   type=int, default=8,
        help="Hour to run daily job (default: 8)"
    )
    parser.add_argument(
        "--minute", type=int, default=0,
        help="Minute to run daily job (default: 0)"
    )
    args = parser.parse_args()

    if args.run_now:
        print("Running pipeline immediately (--run-now)...\n")
        setup_database()
        result = run_pipeline()
        print(f"\nPipeline finished. Status: {result['status']}")
    else:
        start_scheduler(hour=args.hour, minute=args.minute)
