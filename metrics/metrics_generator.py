
import logging
from datetime import datetime

logger = logging.getLogger(__name__)



CHECK_WEIGHTS = {
    "null_value_detection":       10,
    "duplicate_detection":         8,
    "data_type_validation":        5,
    "column_existence":            7,
    "range_validation":            6,
    "regex_pattern_validation":    4,
    "outlier_detection_iqr":       5,
    "outlier_detection_zscore":    4,
    "schema_change_detection":     7,
    "data_freshness":              6,
    "volume_anomaly":              6,
    "missing_value_percentage":    5,
    "psi_distribution_drift":      5,
    "correlation_drift":           4,
    "data_skew_detection":         3,
    "entropy_change":              3,
    "time_gap_detection":          3,
    "business_rule_validation":    8,
    "uniqueness_constraints":      6,
    "referential_consistency":     4,
    "sales_calculation_check":     5,
}



def compute_quality_score(check_results: list) -> float:
    """
    Weighted quality score from 0–100.

    Passed checks earn their full weight.
    Failed checks earn 0.

    Parameters
    ----------
    check_results : list of dicts from run_all_checks()

    Returns
    -------
    float – quality score 0 to 100
    """
    total_weight  = 0
    earned_weight = 0

    for result in check_results:
        check_name = result["check"]
        weight     = CHECK_WEIGHTS.get(check_name, 3)
        total_weight  += weight
        if result["passed"]:
            earned_weight += weight

    score = (earned_weight / total_weight * 100) if total_weight > 0 else 0
    return round(score, 2)


def extract_metrics(df,
                    check_results: list,
                    dataset_name: str = "ecommerce_sales") -> dict:
 
    checks = {r["check"]: r for r in check_results}

    row_count     = len(df)
    quality_score = compute_quality_score(check_results)

  
    null_pct = checks.get("null_value_detection", {}) \
                     .get("details", {}) \
                     .get("overall_null_pct", 0.0)


    duplicate_count = checks.get("duplicate_detection", {}) \
                            .get("details", {}) \
                            .get("duplicate_rows", 0)

    critical_checks = [
        "null_value_detection",
        "duplicate_detection",
        "schema_change_detection",
        "volume_anomaly",
        "business_rule_validation",
        "outlier_detection_zscore",
    ]
    anomaly_flag = int(any(
        not checks[c]["passed"]
        for c in critical_checks
        if c in checks
    ))

    drift_score = checks.get("psi_distribution_drift", {}) \
                        .get("details", {}) \
                        .get("psi_score", 0.0)


    outlier_count_iqr = sum(
        v.get("outlier_count", 0)
        for v in checks.get("outlier_detection_iqr", {})
                       .get("details", {}).values()
        if isinstance(v, dict)
    )

    volume_change_pct = checks.get("volume_anomaly", {}) \
                              .get("details", {}) \
                              .get("change_pct", 0.0)

    max_skewness = max(
        (v.get("skewness", 0)
         for v in checks.get("data_skew_detection", {})
                        .get("details", {})
                        .get("per_column", {}).values()
         if isinstance(v, dict)),
        default=0.0
    )

    entropy_delta = checks.get("entropy_change", {}) \
                          .get("details", {}) \
                          .get("entropy_delta", 0.0)

    correlation_drift = checks.get("correlation_drift", {}) \
                              .get("details", {}) \
                              .get("max_correlation_drift", 0.0)

    checks_passed = sum(1 for r in check_results if r["passed"])
    checks_failed = len(check_results) - checks_passed

    metrics = {

        "dataset_name":      dataset_name,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count":         int(row_count),
        "null_percentage":   float(null_pct),
        "duplicate_count":   int(duplicate_count),
        "quality_score":     float(quality_score),
        "anomaly_flag":      int(anomaly_flag),
        "drift_score":       float(drift_score),
  
        "outlier_count":     int(outlier_count_iqr),
        "volume_change_pct": float(volume_change_pct),
        "max_skewness":      float(abs(max_skewness)),
        "entropy_delta":     float(entropy_delta),
        "correlation_drift": float(correlation_drift),
        "checks_passed":     int(checks_passed),
        "checks_failed":     int(checks_failed),
        "total_checks":      int(len(check_results)),
    }

    logger.info(
        f"Metrics generated | score={quality_score} | anomaly={anomaly_flag} "
        f"| drift={drift_score} | checks={checks_passed}/{len(check_results)}"
    )
    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """Print a human-readable summary of the metrics dict."""
    print("\n" + "=" * 55)
    print("   DATA QUALITY METRICS SUMMARY")
    print("=" * 55)
    print(f"  Dataset       : {metrics['dataset_name']}")
    print(f"  Timestamp     : {metrics['timestamp']}")
    print(f"  Row Count     : {metrics['row_count']:,}")
    print(f"  Null %        : {metrics['null_percentage']:.2f}%")
    print(f"  Duplicates    : {metrics['duplicate_count']}")
    print(f"  Outliers(IQR) : {metrics['outlier_count']}")
    print(f"  Volume Δ %    : {metrics['volume_change_pct']:.2f}%")
    print(f"  Drift Score   : {metrics['drift_score']:.4f}")
    print(f"  Correlation Δ : {metrics['correlation_drift']:.4f}")
    print(f"  Entropy Δ     : {metrics['entropy_delta']:.4f}")
    print(f"  Skewness(max) : {metrics['max_skewness']:.4f}")
    print(f"  Checks        : {metrics['checks_passed']}/{metrics['total_checks']} passed")
    print(f"  Anomaly Flag  : {'🚨 YES' if metrics['anomaly_flag'] else '✅ NO'}")
    score = metrics['quality_score']
    bar_len = int(score / 5)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
    print(f"  Quality Score : [{bar}] {score:.1f}/100  Grade: {grade}")
    print("=" * 55 + "\n")



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.data_ingestion import load_latest_snapshot, load_previous_snapshot
    from validation.quality_checks import run_all_checks

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SNAP_DIR = os.path.join(BASE_DIR, "data", "snapshots")

    df      = load_latest_snapshot(SNAP_DIR)
    prev_df = load_previous_snapshot(SNAP_DIR)
    results = run_all_checks(df, prev_df)
    metrics = extract_metrics(df, results)
    print_metrics_summary(metrics)
