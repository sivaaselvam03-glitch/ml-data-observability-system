# 🔍 ML-Driven Data Observability System
### Final Year Project — Data Quality Monitoring with Anomaly Detection

---

## 📌 Project Overview

This system treats an **Ecommerce Sales CSV dataset as a live production data source**
with simulated daily updates. It automatically monitors data quality, detects anomalies
using Machine Learning, stores metrics in MySQL, and visualises everything in Power BI.

---

## 🗂️ Project Structure

```
data_observability/
│
├── data/
│   ├── Ecommerce_Sales_Data_2024_2025.csv   ← Source dataset
│   └── snapshots/                           ← Simulated daily CSV files
│       ├── snapshot_2024-10-01.csv
│       ├── snapshot_2024-10-02.csv
│       └── ...
│
├── ingestion/
│   └── data_ingestion.py        ← Load CSV, add timestamps, create snapshots
│
├── validation/
│   └── quality_checks.py        ← 21 modular quality checks
│
├── metrics/
│   └── metrics_generator.py     ← Convert checks → numerical metrics + score
│
├── database/
│   └── db_manager.py            ← MySQL: create table, insert, fetch
│
├── ml_anomaly/
│   ├── anomaly_detector.py      ← Isolation Forest + Prophet
│   └── saved_model.pkl          ← Saved model (auto-generated)
│
├── scheduler/
│   └── pipeline_scheduler.py    ← APScheduler daily automation
│
├── utils/
│   └── logger.py                ← Centralised logging
│
├── notebooks/
│   └── DataObservability_MainNotebook.ipynb   ← Full demo notebook
│
├── logs/
│   └── pipeline_YYYY-MM-DD.log  ← Daily log files
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions (Step by Step for Beginners)

### Step 1 — Install Python Libraries

Open Anaconda Prompt or Command Prompt and run:

```bash
pip install pandas numpy scipy scikit-learn mysql-connector-python APScheduler prophet matplotlib seaborn
```

### Step 2 — Set Up MySQL

1. Open **MySQL Workbench**
2. Connect to your local server (host: localhost, port: 3306)
3. Run this SQL:
```sql
CREATE DATABASE IF NOT EXISTS data_observability;
```
4. Note your MySQL **username** and **password**

### Step 3 — Configure the Project

Open `database/db_manager.py` and edit:
```python
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": "YOUR_MYSQL_PASSWORD",   # ← Change this
    "database": "data_observability"
}
```

### Step 4 — Place Your Dataset

Put the CSV file at:
```
data_observability/data/Ecommerce_Sales_Data_2024_2025.csv
```

### Step 5 — Run the Jupyter Notebook

```bash
cd data_observability
jupyter notebook
```

Open `notebooks/DataObservability_MainNotebook.ipynb` and run cells top to bottom.

---

## 🔄 Daily Automation (APScheduler)

Run the pipeline **once now** (for testing):
```bash
python scheduler/pipeline_scheduler.py --run-now
```

Schedule it **every day at 8 AM automatically**:
```bash
python scheduler/pipeline_scheduler.py --hour 8 --minute 0
```

---

## 📊 Quality Checks Implemented (21 Checks)

| # | Check | Category |
|---|-------|----------|
| 1 | Null value detection | Basic |
| 2 | Duplicate row detection | Basic |
| 3 | Data type validation | Basic |
| 4 | Column existence check | Basic |
| 5 | Range validation (Quantity, Discount, Sales, Profit) | Advanced |
| 6 | Regex pattern validation (dates, names) | Advanced |
| 7 | Outlier detection (IQR method) | Advanced |
| 8 | Outlier detection (Z-score method) | Advanced |
| 9 | Schema change detection | Advanced |
| 10 | Data freshness check | Advanced |
| 11 | Volume anomaly detection | Advanced |
| 12 | Missing value percentage tracking | Advanced |
| 13 | Distribution drift (PSI) | Advanced |
| 14 | Correlation drift | Advanced |
| 15 | Data skew detection | Advanced |
| 16 | Entropy change detection | Advanced |
| 17 | Time gap detection | Advanced |
| 18 | Business rule validation | Advanced |
| 19 | Uniqueness constraints | Advanced |
| 20 | Referential consistency | Advanced |
| 21 | Sales calculation check | Business |

---

## 🤖 ML Models Used

### Isolation Forest
- Trained on historical metrics (row_count, null_percentage, quality_score, etc.)
- Detects rows in metrics history that deviate from the normal pattern
- Outputs: anomaly_flag (0/1) and anomaly_score (0.0–1.0)

### Prophet (Facebook)
- Forecasts future `row_count` using time-series modelling
- Detects if actual row count falls outside the 95% confidence interval
- Outputs: forecasted value + anomaly flag

---

## 🗄️ MySQL Schema

```sql
CREATE TABLE metrics_history (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name       VARCHAR(100),
    timestamp          DATETIME,
    row_count          INT,
    null_percentage    DECIMAL(6,4),
    duplicate_count    INT,
    quality_score      DECIMAL(6,2),
    anomaly_flag       TINYINT(1),
    drift_score        DECIMAL(8,4),
    outlier_count      INT,
    volume_change_pct  DECIMAL(8,2),
    max_skewness       DECIMAL(8,4),
    entropy_delta      DECIMAL(8,4),
    correlation_drift  DECIMAL(8,4),
    checks_passed      INT,
    checks_failed      INT,
    total_checks       INT,
    created_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 📊 Power BI Connection Steps

1. Open Power BI Desktop
2. Click **Home → Get Data → MySQL database**
3. Enter:
   - Server: `localhost`
   - Database: `data_observability`
4. Enter your MySQL credentials
5. Select `metrics_history` table OR use the custom SQL queries from Cell 13 of the notebook
6. Create visuals:
   - **Line chart**: `report_date` vs `quality_score`
   - **Bar chart**: `report_date` vs `row_count` (color by anomaly_flag)
   - **Area chart**: `report_date` vs `null_percentage`
   - **Scatter**: `quality_score` vs `drift_score` (color by anomaly_flag)
   - **KPI card**: Latest `quality_score`

---

## 📄 Requirements

```
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
mysql-connector-python>=8.0.0
APScheduler>=3.9.0
prophet>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

## 🎓 Resume Description

> **ML-Driven Data Observability System** | Python, MySQL, Power BI, Scikit-learn, Prophet
>
> Built an end-to-end automated data quality monitoring system that ingests 5,000+ daily
> e-commerce records, executes 21 modular observability checks (null detection, schema drift,
> PSI distribution analysis, business rule validation), and computes a weighted quality score.
> Integrated Isolation Forest for multi-dimensional anomaly detection on time-series metrics
> and Facebook Prophet for row-count forecasting. Automated daily execution via APScheduler,
> persisted metrics in MySQL, and visualised trends in Power BI dashboards.

---

## 🗣️ Interview Explanation (What to Say)

**"Tell me about your final year project."**

> "I built a data observability platform — essentially a system that watches over a data
> pipeline and automatically flags quality problems before they affect downstream reports.
>
> The system ingests daily snapshots of an e-commerce dataset, runs 21 checks covering
> everything from basic nulls and duplicates to advanced things like PSI distribution drift
> and entropy change. It computes a weighted quality score from 0 to 100.
>
> On the ML side, I used Isolation Forest, which is an unsupervised algorithm, to detect
> anomalies in the metrics history — for example, sudden spikes in null percentage or
> unusual volume changes. I also integrated Prophet for time-series forecasting to predict
> expected row counts and flag deviations.
>
> Everything is automated with APScheduler, metrics are stored in MySQL, and I built a
> Power BI dashboard showing quality trends, drift indicators, and anomaly flags over time."
