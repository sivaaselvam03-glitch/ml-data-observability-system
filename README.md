# 🚀 ML-Driven Data Observability System

## 📌 Overview

The ML-Driven Data Observability System is an intelligent monitoring platform that automates data quality validation, anomaly detection, and data drift monitoring for modern data pipelines.

The system combines traditional data quality checks with machine learning techniques to ensure reliable, accurate, and trustworthy data for analytics and AI applications. It continuously monitors incoming datasets, detects anomalies, tracks quality metrics, and provides actionable insights through interactive dashboards.

---

## ✨ Features

### 🔍 Data Quality Monitoring

* Null Value Detection
* Duplicate Record Detection
* Data Type Validation
* Schema Validation
* Missing Column Detection
* Range Validation
* Pattern Validation
* Business Rule Validation

### 📊 Data Drift Detection

* Population Stability Index (PSI)
* Distribution Drift Analysis
* Correlation Drift Detection
* Statistical Change Monitoring

### 🤖 Machine Learning-Based Anomaly Detection

* Isolation Forest Anomaly Detection
* IQR-Based Outlier Detection
* Z-Score Outlier Detection
* Volume Change Monitoring
* Data Freshness Monitoring

### 📈 Visualization & Reporting

* Power BI Dashboard Integration
* Quality Score Tracking
* Drift Trend Analysis
* Historical Metrics Monitoring
* Automated Reports

### ⚙️ Automation

* APScheduler-Based Scheduling
* Automated Daily Monitoring
* Centralized Logging
* MySQL Metrics Storage

---

## 🏗️ System Architecture

```text
Data Source (CSV Files)
         │
         ▼
  Data Ingestion Layer
         │
         ▼
 Data Quality Checks
         │
         ▼
 Data Drift Detection
         │
         ▼
 ML Anomaly Detection
         │
         ▼
 Metrics Storage (MySQL)
         │
         ▼
 Power BI Dashboard
```

## 🛠️ Technology Stack

| Component            | Technology          |
| -------------------- | ------------------- |
| Programming Language | Python              |
| Database             | MySQL               |
| Visualization        | Power BI            |
| Machine Learning     | Scikit-Learn        |
| Scheduling           | APScheduler         |
| Data Processing      | Pandas, NumPy       |
| Statistical Analysis | SciPy               |
| Forecasting          | Prophet             |
| Plotting             | Matplotlib, Seaborn |

---

## 📂 Project Structure

```text
ml-data-observability-system/
│
├── data/
├── database/
├── ingestion/
├── metrics/
├── ml_anomaly/
├── scheduler/
├── utils/
├── validation/
├── notebooks/
│
├── app.py
├── README.md
└── requirements.txt
```

---

## 🤖 Machine Learning Model

### Isolation Forest

The project uses the Isolation Forest algorithm for anomaly detection.

#### Advantages

* Unsupervised Learning
* No Labeled Data Required
* Efficient for Large Datasets
* Effective at Detecting Rare Events
* Scalable and Fast

---

## 📊 Key Metrics Monitored

* Quality Score
* Null Percentage
* Duplicate Count
* Drift Score (PSI)
* Outlier Count
* Volume Change Percentage
* Data Freshness
* Correlation Change
* Anomaly Flag

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/sivaaselvam03-glitch/ml-data-observability-system.git
cd ml-data-observability-system
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python app.py
```

---

## 📈 Sample Capabilities

✅ 21 Automated Data Quality Checks

✅ Machine Learning-Based Anomaly Detection

✅ Data Drift Monitoring Using PSI

✅ Automated Daily Monitoring

✅ Historical Trend Analysis

✅ Dashboard-Based Visualization

---

## 🔮 Future Enhancements

* Real-Time Data Streaming
* Kafka Integration
* Cloud Deployment (AWS / Azure)
* AutoML-Based Anomaly Detection
* Email & Slack Alerts
* LLM-Based Root Cause Analysis
* Predictive Data Quality Monitoring

---

## 👨‍💻 Author

**Siva Selvam**

B.Tech Information Technology

Aspiring Data Scientist | AI Engineer | ML Engineer

GitHub: https://github.com/sivaaselvam03-glitch

---

⭐ If you found this project useful, consider giving it a star!
