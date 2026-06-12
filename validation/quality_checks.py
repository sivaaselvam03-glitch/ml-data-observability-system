"""
=============================================================
MODULE: validation/quality_checks.py
PURPOSE: 20+ data quality & observability checks, all modular.
         Every check takes a DataFrame and returns a dict.
=============================================================
"""

import re
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import entropy

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════
# DATASET-SPECIFIC CONFIG  (tuned for Ecommerce dataset)
# ═════════════════════════════════════════════════════════
EXPECTED_COLUMNS = [
    "Order ID", "Order Date", "Customer Name", "Region",
    "City", "Category", "Sub-Category", "Product Name",
    "Quantity", "Unit Price", "Discount", "Sales",
    "Profit", "Payment Mode"
]

EXPECTED_DTYPES = {
    "Order ID":       ["int64", "int32"],
    "Quantity":       ["int64", "int32"],
    "Unit Price":     ["int64", "int32", "float64"],
    "Discount":       ["int64", "int32", "float64"],
    "Sales":          ["float64", "float32"],
    "Profit":         ["float64", "float32"],
}

NUMERIC_RANGES = {
    "Quantity":   (1, 5),
    "Discount":   (0, 20),
    "Unit Price": (100, 100000),
    "Sales":      (0, 500000),
    "Profit":     (-50000, 100000),
}

VALID_REGIONS       = {"North", "South", "East", "West"}
VALID_PAYMENT_MODES = {"Debit Card", "Credit Card", "UPI", "Net Banking", "COD"}
VALID_CATEGORIES    = {
    "Books", "Groceries", "Kitchen", "Clothing", "Furniture",
    "Beauty", "Home Decor", "Electronics", "Sports", "Toys"
}

# ════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════
def _result(check_name: str, passed: bool, details: dict) -> dict:
    """Wrap every check result in a standard envelope."""
    return {
        "check": check_name,
        "passed": passed,
        "details": details
    }


# ════════════════════════════════════════════════════════
# ── BASIC CHECKS ──────────────────────────────────────
# ════════════════════════════════════════════════════════

# CHECK 1
def check_null_values(df: pd.DataFrame) -> dict:
    """Detect null / NaN values per column."""
    null_counts = df.isnull().sum().to_dict()
    null_pct    = (df.isnull().mean() * 100).round(2).to_dict()
    total_nulls = sum(null_counts.values())
    return _result(
        "null_value_detection",
        passed=total_nulls == 0,
        details={
            "null_counts":      null_counts,
            "null_percentages": null_pct,
            "total_nulls":      int(total_nulls),
            "overall_null_pct": round(df.isnull().mean().mean() * 100, 2)
        }
    )


# CHECK 2
def check_duplicates(df: pd.DataFrame) -> dict:
    """Detect fully duplicate rows and duplicate Order IDs."""
    dup_rows  = int(df.duplicated().sum())
    dup_ids   = int(df["Order ID"].duplicated().sum()) if "Order ID" in df.columns else 0
    return _result(
        "duplicate_detection",
        passed=(dup_rows == 0 and dup_ids == 0),
        details={
            "duplicate_rows":      dup_rows,
            "duplicate_order_ids": dup_ids
        }
    )


# CHECK 3
def check_data_types(df: pd.DataFrame) -> dict:
    """Validate that key columns have expected dtypes."""
    mismatches = {}
    for col, expected in EXPECTED_DTYPES.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual not in expected:
                mismatches[col] = {"expected": expected, "actual": actual}
    return _result(
        "data_type_validation",
        passed=len(mismatches) == 0,
        details={"mismatches": mismatches, "checked_columns": list(EXPECTED_DTYPES.keys())}
    )


# CHECK 4
def check_column_existence(df: pd.DataFrame) -> dict:
    """Verify all expected columns are present."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_COLUMNS and c != "ingestion_timestamp"]
    return _result(
        "column_existence",
        passed=len(missing) == 0,
        details={"missing_columns": missing, "extra_columns": extra}
    )


# ════════════════════════════════════════════════════════
# ── ADVANCED CHECKS ───────────────────────────────────
# ════════════════════════════════════════════════════════

# CHECK 5
def check_range_validation(df: pd.DataFrame) -> dict:
    """Ensure numeric columns stay within expected min/max bounds."""
    violations = {}
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        out_of_range = col_data[(col_data < lo) | (col_data > hi)]
        if len(out_of_range) > 0:
            violations[col] = {
                "expected_range":     [lo, hi],
                "violation_count":    int(len(out_of_range)),
                "min_found":          float(col_data.min()),
                "max_found":          float(col_data.max())
            }
    return _result(
        "range_validation",
        passed=len(violations) == 0,
        details={"violations": violations}
    )


# CHECK 6
def check_regex_patterns(df: pd.DataFrame) -> dict:
    """Validate string columns against expected regex patterns."""
    results = {}

    # Order Date must be YYYY-MM-DD
    if "Order Date" in df.columns:
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        bad = df["Order Date"].dropna().apply(
            lambda x: not bool(re.match(pattern, str(x)))
        )
        results["Order Date"] = {
            "pattern": pattern,
            "invalid_count": int(bad.sum())
        }

    # Customer Name: at least 2 words, letters only
    if "Customer Name" in df.columns:
        pattern = r"^[A-Za-z]+ [A-Za-z]+$"
        bad = df["Customer Name"].dropna().apply(
            lambda x: not bool(re.match(pattern, str(x)))
        )
        results["Customer Name"] = {
            "pattern": pattern,
            "invalid_count": int(bad.sum())
        }

    passed = all(v["invalid_count"] == 0 for v in results.values())
    return _result("regex_pattern_validation", passed=passed, details=results)


# CHECK 7
def check_outliers_iqr(df: pd.DataFrame) -> dict:
    """Detect outliers using the Interquartile Range (IQR) method."""
    outlier_info = {}
    numeric_cols = ["Sales", "Profit", "Unit Price", "Quantity"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR    = Q3 - Q1
        lower  = Q1 - 1.5 * IQR
        upper  = Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]
        outlier_info[col] = {
            "lower_bound":   round(float(lower), 2),
            "upper_bound":   round(float(upper), 2),
            "outlier_count": int(len(outliers)),
            "outlier_pct":   round(len(outliers) / len(data) * 100, 2)
        }
    total_outliers = sum(v["outlier_count"] for v in outlier_info.values())
    return _result(
        "outlier_detection_iqr",
        passed=total_outliers == 0,
        details=outlier_info
    )


# CHECK 8
def check_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> dict:
    """Detect outliers using Z-score method (|z| > threshold)."""
    outlier_info = {}
    numeric_cols = ["Sales", "Profit", "Unit Price"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > threshold]
        outlier_info[col] = {
            "threshold":     threshold,
            "outlier_count": int(len(outliers)),
            "max_zscore":    round(float(z_scores.max()), 2) if len(z_scores) > 0 else 0
        }
    total = sum(v["outlier_count"] for v in outlier_info.values())
    return _result(
        "outlier_detection_zscore",
        passed=total == 0,
        details=outlier_info
    )


# CHECK 9
def check_schema_change(df: pd.DataFrame,
                         prev_df: pd.DataFrame) -> dict:
    """Compare current schema against previous snapshot's schema."""
    if prev_df is None:
        return _result("schema_change_detection", passed=True,
                       details={"message": "No previous snapshot available."})

    curr_cols  = set(df.columns)
    prev_cols  = set(prev_df.columns)
    added      = list(curr_cols - prev_cols)
    removed    = list(prev_cols - curr_cols)
    type_changes = {}
    for col in curr_cols & prev_cols:
        if str(df[col].dtype) != str(prev_df[col].dtype):
            type_changes[col] = {
                "prev_dtype": str(prev_df[col].dtype),
                "curr_dtype": str(df[col].dtype)
            }
    changed = bool(added or removed or type_changes)
    return _result(
        "schema_change_detection",
        passed=not changed,
        details={
            "columns_added":    added,
            "columns_removed":  removed,
            "type_changes":     type_changes
        }
    )


# CHECK 10
def check_data_freshness(df: pd.DataFrame,
                          max_days_old: int = 30) -> dict:
    """
    Check if the most recent Order Date is within max_days_old days.
    """
    if "Order Date" not in df.columns:
        return _result("data_freshness", passed=False,
                       details={"error": "Order Date column missing"})
    try:
        dates      = pd.to_datetime(df["Order Date"], errors="coerce").dropna()
        latest_date = dates.max()
        today       = pd.Timestamp.now().normalize()
        age_days    = (today - latest_date).days
        return _result(
            "data_freshness",
            passed=age_days <= max_days_old,
            details={
                "latest_order_date": str(latest_date.date()),
                "age_days":          int(age_days),
                "max_allowed_days":  max_days_old
            }
        )
    except Exception as e:
        return _result("data_freshness", passed=False, details={"error": str(e)})


# CHECK 11
def check_volume_anomaly(df: pd.DataFrame,
                          prev_df: pd.DataFrame,
                          threshold_pct: float = 30.0) -> dict:
    """Flag if row count changed by more than threshold_pct % vs previous."""
    curr_count = len(df)
    if prev_df is None:
        return _result("volume_anomaly", passed=True,
                       details={"current_rows": curr_count,
                                "message": "No previous snapshot."})
    prev_count  = len(prev_df)
    change_pct  = abs(curr_count - prev_count) / max(prev_count, 1) * 100
    return _result(
        "volume_anomaly",
        passed=change_pct <= threshold_pct,
        details={
            "previous_rows": prev_count,
            "current_rows":  curr_count,
            "change_pct":    round(change_pct, 2),
            "threshold_pct": threshold_pct
        }
    )


# CHECK 12
def check_missing_value_percentage(df: pd.DataFrame,
                                    max_pct: float = 5.0) -> dict:
    """Track missing value percentage per column with a threshold."""
    results = {}
    for col in df.columns:
        pct = round(df[col].isnull().mean() * 100, 2)
        results[col] = {"missing_pct": pct, "exceeds_threshold": pct > max_pct}
    flagged = [c for c, v in results.items() if v["exceeds_threshold"]]
    return _result(
        "missing_value_percentage",
        passed=len(flagged) == 0,
        details={"per_column": results, "columns_exceeding_threshold": flagged}
    )


# CHECK 13
def check_distribution_drift_psi(df: pd.DataFrame,
                                   prev_df: pd.DataFrame,
                                   col: str = "Sales",
                                   bins: int = 10) -> dict:
    """
    Population Stability Index (PSI) for a numeric column.
    PSI < 0.1  → no drift
    PSI 0.1-0.2 → slight drift
    PSI > 0.2  → significant drift
    """
    if prev_df is None or col not in df.columns or col not in prev_df.columns:
        return _result("psi_distribution_drift", passed=True,
                       details={"message": "Insufficient data for PSI."})

    curr_data = df[col].dropna().values
    prev_data = prev_df[col].dropna().values
    all_data  = np.concatenate([curr_data, prev_data])
    breakpoints = np.linspace(all_data.min(), all_data.max(), bins + 1)

    curr_counts = np.histogram(curr_data, bins=breakpoints)[0]
    prev_counts = np.histogram(prev_data, bins=breakpoints)[0]

    curr_pct = curr_counts / (curr_counts.sum() + 1e-9)
    prev_pct = prev_counts / (prev_counts.sum() + 1e-9)

    # Avoid log(0)
    curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)
    prev_pct = np.where(prev_pct == 0, 0.0001, prev_pct)

    psi = float(np.sum((curr_pct - prev_pct) * np.log(curr_pct / prev_pct)))
    psi = round(psi, 4)

    if psi < 0.1:
        drift_level = "none"
    elif psi < 0.2:
        drift_level = "slight"
    else:
        drift_level = "significant"

    return _result(
        "psi_distribution_drift",
        passed=psi < 0.2,
        details={"column": col, "psi_score": psi, "drift_level": drift_level}
    )


# CHECK 14
def check_correlation_drift(df: pd.DataFrame,
                              prev_df: pd.DataFrame,
                              threshold: float = 0.2) -> dict:
    """
    Compare Pearson correlation between Sales and Profit across snapshots.
    """
    numeric_cols = ["Sales", "Profit", "Unit Price", "Quantity", "Discount"]
    available    = [c for c in numeric_cols if c in df.columns]

    if prev_df is None or len(available) < 2:
        return _result("correlation_drift", passed=True,
                       details={"message": "Insufficient data."})

    curr_corr = df[available].corr().fillna(0)
    prev_corr = prev_df[available].corr().fillna(0)
    diff      = (curr_corr - prev_corr).abs()
    max_drift = float(diff.max().max())
    worst_pair = diff.stack().idxmax()

    return _result(
        "correlation_drift",
        passed=max_drift <= threshold,
        details={
            "max_correlation_drift": round(max_drift, 4),
            "worst_pair":            str(worst_pair),
            "threshold":             threshold
        }
    )


# CHECK 15
def check_data_skew(df: pd.DataFrame, threshold: float = 2.0) -> dict:
    """Detect highly skewed numeric distributions (|skewness| > threshold)."""
    skew_info = {}
    numeric_cols = ["Sales", "Profit", "Unit Price", "Quantity"]
    for col in numeric_cols:
        if col not in df.columns:
            continue
        skewness = float(df[col].dropna().skew())
        skew_info[col] = {
            "skewness":        round(skewness, 4),
            "highly_skewed":   abs(skewness) > threshold
        }
    flagged = [c for c, v in skew_info.items() if v["highly_skewed"]]
    return _result(
        "data_skew_detection",
        passed=len(flagged) == 0,
        details={"per_column": skew_info, "highly_skewed_columns": flagged}
    )


# CHECK 16
def check_entropy_change(df: pd.DataFrame,
                          prev_df: pd.DataFrame,
                          col: str = "Category") -> dict:
    """
    Measure Shannon entropy change of a categorical column.
    Big change → distribution has shifted.
    """
    if col not in df.columns:
        return _result("entropy_change", passed=True,
                       details={"message": f"Column '{col}' not found."})

    curr_freq = df[col].value_counts(normalize=True).values
    curr_ent  = float(entropy(curr_freq, base=2))

    if prev_df is None or col not in prev_df.columns:
        return _result("entropy_change", passed=True,
                       details={"current_entropy": round(curr_ent, 4),
                                "message": "No previous snapshot."})

    prev_freq = prev_df[col].value_counts(normalize=True).values
    prev_ent  = float(entropy(prev_freq, base=2))
    delta     = abs(curr_ent - prev_ent)

    return _result(
        "entropy_change",
        passed=delta < 0.5,
        details={
            "column":           col,
            "previous_entropy": round(prev_ent, 4),
            "current_entropy":  round(curr_ent, 4),
            "entropy_delta":    round(delta, 4)
        }
    )


# CHECK 17
def check_time_gaps(df: pd.DataFrame,
                    date_col: str = "Order Date",
                    max_gap_days: int = 7) -> dict:
    """Detect large time gaps in the Order Date sequence."""
    if date_col not in df.columns:
        return _result("time_gap_detection", passed=True,
                       details={"message": f"Column '{date_col}' not found."})

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    gaps  = dates.diff().dt.days.dropna()
    large_gaps = gaps[gaps > max_gap_days]

    return _result(
        "time_gap_detection",
        passed=len(large_gaps) == 0,
        details={
            "max_gap_days_found": int(gaps.max()) if len(gaps) > 0 else 0,
            "large_gap_count":    int(len(large_gaps)),
            "max_allowed_gap":    max_gap_days
        }
    )


# CHECK 18
def check_business_rules(df: pd.DataFrame) -> dict:
    """
    Domain-specific business rule validation:
    1. Sales must be > 0
    2. Profit cannot exceed Sales
    3. Discount must be between 0 and 20
    4. Region must be one of the valid set
    5. Payment Mode must be valid
    6. Category must be valid
    """
    violations = {}

    if "Sales" in df.columns:
        neg_sales = int((df["Sales"].dropna() <= 0).sum())
        if neg_sales:
            violations["sales_positive"] = f"{neg_sales} rows with Sales <= 0"

    if "Sales" in df.columns and "Profit" in df.columns:
        both = df[["Sales", "Profit"]].dropna()
        impossible = int((both["Profit"] > both["Sales"]).sum())
        if impossible:
            violations["profit_le_sales"] = f"{impossible} rows where Profit > Sales"

    if "Discount" in df.columns:
        bad_disc = int(((df["Discount"].dropna() < 0) | (df["Discount"].dropna() > 20)).sum())
        if bad_disc:
            violations["discount_range"] = f"{bad_disc} rows with invalid Discount"

    if "Region" in df.columns:
        invalid_regions = df[~df["Region"].isin(VALID_REGIONS)]["Region"].unique().tolist()
        if invalid_regions:
            violations["valid_region"] = f"Invalid regions: {invalid_regions}"

    if "Payment Mode" in df.columns:
        invalid_pm = df[~df["Payment Mode"].isin(VALID_PAYMENT_MODES)]["Payment Mode"].unique().tolist()
        if invalid_pm:
            violations["valid_payment_mode"] = f"Invalid payment modes: {invalid_pm}"

    if "Category" in df.columns:
        invalid_cat = df[~df["Category"].isin(VALID_CATEGORIES)]["Category"].unique().tolist()
        if invalid_cat:
            violations["valid_category"] = f"Invalid categories: {invalid_cat}"

    return _result(
        "business_rule_validation",
        passed=len(violations) == 0,
        details={"violations": violations, "rules_checked": 6}
    )


# CHECK 19
def check_uniqueness_constraints(df: pd.DataFrame) -> dict:
    """Enforce uniqueness on Order ID column."""
    results = {}
    if "Order ID" in df.columns:
        total    = len(df)
        unique   = df["Order ID"].nunique()
        dup_count = total - unique
        results["Order ID"] = {
            "total_rows":    total,
            "unique_values": unique,
            "duplicates":    dup_count,
            "passed":        dup_count == 0
        }
    passed = all(v["passed"] for v in results.values())
    return _result("uniqueness_constraints", passed=passed, details=results)


# CHECK 20
def check_referential_consistency(df: pd.DataFrame) -> dict:
    """
    Check that Sub-Category values are consistent with their Category.
    For our dataset, Sub-Categories are free text so we verify they
    are non-empty and non-null when Category is present.
    """
    issues = []
    if "Category" in df.columns and "Sub-Category" in df.columns:
        both_present = df[["Category", "Sub-Category"]].dropna()
        empty_sub = (both_present["Sub-Category"].str.strip() == "").sum()
        if empty_sub > 0:
            issues.append(f"{empty_sub} rows with empty Sub-Category")

        # Ensure every row with Category also has Sub-Category
        cat_no_sub = df["Category"].notna() & df["Sub-Category"].isna()
        if cat_no_sub.sum() > 0:
            issues.append(f"{int(cat_no_sub.sum())} rows have Category but missing Sub-Category")

    return _result(
        "referential_consistency",
        passed=len(issues) == 0,
        details={"issues": issues}
    )


# CHECK 21  (bonus)
def check_sales_calculation(df: pd.DataFrame) -> dict:
    """
    Verify Sales ≈ Quantity × Unit Price × (1 - Discount/100).
    Allow 1% tolerance for rounding.
    """
    required = ["Sales", "Quantity", "Unit Price", "Discount"]
    if not all(c in df.columns for c in required):
        return _result("sales_calculation_check", passed=True,
                       details={"message": "Required columns missing."})

    df2 = df[required].dropna().copy()
    df2["expected_sales"] = df2["Quantity"] * df2["Unit Price"] * (1 - df2["Discount"] / 100)
    df2["pct_diff"] = ((df2["Sales"] - df2["expected_sales"]).abs() / (df2["expected_sales"] + 1e-9)) * 100
    bad_rows = int((df2["pct_diff"] > 1.0).sum())

    return _result(
        "sales_calculation_check",
        passed=bad_rows == 0,
        details={
            "rows_with_bad_calculation": bad_rows,
            "tolerance_pct": 1.0,
            "sample_max_diff_pct": round(float(df2["pct_diff"].max()), 2)
        }
    )


# ════════════════════════════════════════════════════════
# MASTER RUNNER
# ════════════════════════════════════════════════════════
def run_all_checks(df: pd.DataFrame,
                   prev_df: pd.DataFrame = None) -> list:
    """
    Execute all 21 quality checks and return a list of result dicts.

    Parameters
    ----------
    df      : current snapshot DataFrame
    prev_df : previous snapshot DataFrame (for drift/schema checks)

    Returns
    -------
    list of check result dicts
    """
    checks = [
        check_null_values(df),
        check_duplicates(df),
        check_data_types(df),
        check_column_existence(df),
        check_range_validation(df),
        check_regex_patterns(df),
        check_outliers_iqr(df),
        check_outliers_zscore(df),
        check_schema_change(df, prev_df),
        check_data_freshness(df),
        check_volume_anomaly(df, prev_df),
        check_missing_value_percentage(df),
        check_distribution_drift_psi(df, prev_df, col="Sales"),
        check_correlation_drift(df, prev_df),
        check_data_skew(df),
        check_entropy_change(df, prev_df, col="Category"),
        check_time_gaps(df),
        check_business_rules(df),
        check_uniqueness_constraints(df),
        check_referential_consistency(df),
        check_sales_calculation(df),
    ]
    passed_count = sum(1 for c in checks if c["passed"])
    logger.info(f"Quality checks complete: {passed_count}/{len(checks)} passed")
    return checks


# ════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.data_ingestion import load_latest_snapshot, load_previous_snapshot

    BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SNAP_DIR  = os.path.join(BASE_DIR, "data", "snapshots")

    df      = load_latest_snapshot(SNAP_DIR)
    prev_df = load_previous_snapshot(SNAP_DIR)
    results = run_all_checks(df, prev_df)

    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"{status} | {r['check']}")
