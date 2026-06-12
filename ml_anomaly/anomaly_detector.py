
import os
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn")

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False
    logger.warning("Prophet not installed. Run: pip install prophet")



FEATURE_COLS = [
    "row_count",
    "null_percentage",
    "duplicate_count",
    "quality_score",
    "drift_score",
    "outlier_count",
    "volume_change_pct",
    "max_skewness",
    "entropy_delta",
    "correlation_drift",
]


def prepare_features(records: list) -> pd.DataFrame:
    """
    Convert list of metrics dicts (from MySQL) into a
    feature DataFrame suitable for Isolation Forest.

    Parameters
    ----------
    records : list of dicts from db_manager.fetch_all_metrics()

    Returns
    -------
    pd.DataFrame with FEATURE_COLS, NaNs filled with 0
    """
    df = pd.DataFrame(records)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLS].fillna(0.0)


def train_isolation_forest(
    records: list,
    contamination: float = 0.1,
    model_path: str = None
):
   
    if not SKLEARN_OK:
        raise ImportError("scikit-learn is required. Run: pip install scikit-learn")

    if len(records) < 5:
        logger.warning(f"Only {len(records)} records – need ≥ 5 for training.")
        return None, None, None

    X = prepare_features(records)
    logger.info(f"Training Isolation Forest on {len(X)} samples × {len(FEATURE_COLS)} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

  
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        logger.info(f"Model saved to {model_path}")

    logger.info("Isolation Forest training complete.")
    return model, scaler, X


def load_model(model_path: str):
    """Load a previously saved (model, scaler) bundle."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return bundle["model"], bundle["scaler"]



def predict_anomaly(
    metrics: dict,
    model,
    scaler,
) -> dict:
    """
    Use trained Isolation Forest to predict if the latest
    metrics row is anomalous.

    Parameters
    ----------
    metrics : dict from metrics_generator.extract_metrics()
    model   : trained IsolationForest
    scaler  : fitted StandardScaler

    Returns
    -------
    dict with keys: anomaly_flag, anomaly_score, prediction_label
    """
    if not SKLEARN_OK:
        return {"anomaly_flag": 0, "anomaly_score": 0.0, "prediction_label": "unknown"}

    row = {col: metrics.get(col, 0.0) for col in FEATURE_COLS}
    X   = pd.DataFrame([row])
    X_scaled = scaler.transform(X)

   
    prediction = model.predict(X_scaled)[0]
    
    raw_score  = float(model.score_samples(X_scaled)[0])

    
    anomaly_score = round(max(0.0, min(1.0, -raw_score)), 4)
    anomaly_flag  = 1 if prediction == -1 else 0
    label         = "ANOMALY" if anomaly_flag else "NORMAL"

    logger.info(f"ML prediction: {label} | score={anomaly_score:.4f}")
    return {
        "anomaly_flag":      anomaly_flag,
        "anomaly_score":     anomaly_score,
        "prediction_label":  label,
        "raw_score":         raw_score
    }



def generate_synthetic_history(n_days: int = 30) -> list:
    """
    Generate n_days of realistic synthetic metrics rows
    so we can train the Isolation Forest from day 1.

    About 10% of rows are injected as anomalies.
    """
    records = []
    base_date = datetime.now() - timedelta(days=n_days)

    for i in range(n_days):
        day = base_date + timedelta(days=i)
        is_anomaly = (i % 10 == 7)  # every 10th day is anomalous

        if is_anomaly:
            row_count      = np.random.randint(900, 1200)   # spike
            null_pct       = np.random.uniform(8, 15)        # high nulls
            duplicate      = np.random.randint(40, 80)
            quality_score  = np.random.uniform(40, 60)
            drift_score    = np.random.uniform(0.25, 0.50)
            outlier_count  = np.random.randint(30, 60)
            vol_change     = np.random.uniform(35, 60)
        else:
            row_count      = np.random.randint(450, 560)
            null_pct       = np.random.uniform(0, 1.5)
            duplicate      = np.random.randint(0, 5)
            quality_score  = np.random.uniform(80, 100)
            drift_score    = np.random.uniform(0.0, 0.08)
            outlier_count  = np.random.randint(0, 8)
            vol_change     = np.random.uniform(0, 12)

        records.append({
            "dataset_name":     "ecommerce_sales",
            "timestamp":        day.strftime("%Y-%m-%d %H:%M:%S"),
            "row_count":        int(row_count),
            "null_percentage":  round(null_pct, 4),
            "duplicate_count":  int(duplicate),
            "quality_score":    round(quality_score, 2),
            "anomaly_flag":     int(is_anomaly),
            "drift_score":      round(drift_score, 4),
            "outlier_count":    int(outlier_count),
            "volume_change_pct": round(vol_change, 2),
            "max_skewness":     round(np.random.uniform(0.1, 0.8), 4),
            "entropy_delta":    round(np.random.uniform(0.0, 0.15), 4),
            "correlation_drift": round(np.random.uniform(0.0, 0.1), 4),
            "checks_passed":    np.random.randint(16, 21),
            "checks_failed":    np.random.randint(0, 5),
            "total_checks":     21,
            "created_at":       day.strftime("%Y-%m-%d %H:%M:%S"),
        })

    logger.info(f"Generated {n_days} synthetic history records.")
    return records


def forecast_row_count(
    records: list,
    forecast_days: int = 7
) -> pd.DataFrame:
   
    if not PROPHET_OK:
        logger.error("Prophet not installed. Run: pip install prophet")
        return pd.DataFrame()

    df = pd.DataFrame(records)[["timestamp", "row_count"]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    if len(df) < 5:
        logger.warning("Need at least 5 data points for Prophet.")
        return pd.DataFrame()

    logger.info(f"Training Prophet on {len(df)} rows → forecasting {forecast_days} days ahead")

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05
    )
    m.fit(df)

    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)

    # Merge actual values back
    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result = result.merge(
        df.rename(columns={"y": "actual"}),
        on="ds", how="left"
    )

   
    result["anomaly"] = (
        result["actual"].notna() &
        (result["actual"] > result["yhat_upper"])
    )

    logger.info(f"Prophet forecast complete. Anomalies in history: {result['anomaly'].sum()}")
    return result



def run_ml_pipeline(
    current_metrics: dict,
    historical_records: list,
    model_path: str = None
) -> dict:
  t = {
        "isolation_forest": {},
        "prophet":          {}
    }

    # ── Isolation Forest ──────────────────────────────────
    model, scaler = None, None
    if model_path and os.path.exists(model_path):
        try:
            model, scaler = load_model(model_path)
            logger.info("Loaded existing Isolation Forest model.")
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Retraining...")

    if model is None:
        if len(historical_records) < 5:
            logger.info("Insufficient real history. Using synthetic data for training.")
            train_records = generate_synthetic_history(30) + historical_records
        else:
            train_records = historical_records

        model, scaler, _ = train_isolation_forest(
            train_records,
            contamination=0.1,
            model_path=model_path
        )

    if model is not None and scaler is not None:
        result["isolation_forest"] = predict_anomaly(current_metrics, model, scaler)
    else:
        result["isolation_forest"] = {"anomaly_flag": 0, "anomaly_score": 0.0,
                                       "prediction_label": "insufficient_data"}

    
    if PROPHET_OK and len(historical_records) >= 5:
        prophet_df = forecast_row_count(historical_records, forecast_days=7)
        if not prophet_df.empty:
            today_row = prophet_df[prophet_df["ds"] == pd.Timestamp.now().normalize()]
            if not today_row.empty:
                result["prophet"] = {
                    "forecasted_row_count": round(float(today_row["yhat"].values[0]), 0),
                    "lower_bound":          round(float(today_row["yhat_lower"].values[0]), 0),
                    "upper_bound":          round(float(today_row["yhat_upper"].values[0]), 0),
                    "anomaly_detected":     bool(today_row["anomaly"].values[0])
                }
                result["prophet"]["forecast_df"] = prophet_df.to_dict("records")
            else:
                result["prophet"] = {"message": "No forecast row for today."}
    else:
        result["prophet"] = {"message": "Prophet unavailable or insufficient data."}

    return result



if __name__ == "__main__":
    print("Generating synthetic history and testing ML pipeline...")
    synthetic = generate_synthetic_history(30)

    # Simulate "today's" metrics
    today_metrics = {
        "dataset_name":     "ecommerce_sales",
        "row_count":        1100,       # spike!
        "null_percentage":  0.5,
        "duplicate_count":  2,
        "quality_score":    72.0,
        "drift_score":      0.18,
        "outlier_count":    5,
        "volume_change_pct": 45.0,
        "max_skewness":     0.6,
        "entropy_delta":    0.05,
        "correlation_drift": 0.04,
    }

    result = run_ml_pipeline(today_metrics, synthetic)

    print("\n── Isolation Forest ─────────────────────")
    for k, v in result["isolation_forest"].items():
        print(f"  {k}: {v}")

    print("\n── Prophet ──────────────────────────────")
    for k, v in result["prophet"].items():
        if k != "forecast_df":
            print(f"  {k}: {v}")
