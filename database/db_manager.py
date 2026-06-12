

import logging
import mysql.connector
from mysql.connector import Error
from datetime import datetime

logger = logging.getLogger(__name__)




DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",          
    "password": "",  # 
    "database": "data_observability"
}





SQL_CREATE_DATABASE = "CREATE DATABASE IF NOT EXISTS data_observability;"

SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS metrics_history (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    dataset_name       VARCHAR(100)    NOT NULL,
    timestamp          DATETIME        NOT NULL,
    row_count          INT             NOT NULL,
    null_percentage    DECIMAL(6,4)    NOT NULL DEFAULT 0,
    duplicate_count    INT             NOT NULL DEFAULT 0,
    quality_score      DECIMAL(6,2)    NOT NULL DEFAULT 0,
    anomaly_flag       TINYINT(1)      NOT NULL DEFAULT 0,
    drift_score        DECIMAL(8,4)    NOT NULL DEFAULT 0,
    outlier_count      INT             NOT NULL DEFAULT 0,
    volume_change_pct  DECIMAL(8,2)    NOT NULL DEFAULT 0,
    max_skewness       DECIMAL(8,4)    NOT NULL DEFAULT 0,
    entropy_delta      DECIMAL(8,4)    NOT NULL DEFAULT 0,
    correlation_drift  DECIMAL(8,4)    NOT NULL DEFAULT 0,
    checks_passed      INT             NOT NULL DEFAULT 0,
    checks_failed      INT             NOT NULL DEFAULT 0,
    total_checks       INT             NOT NULL DEFAULT 0,
    created_at         DATETIME        DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_dataset_ts (dataset_name, timestamp),
    INDEX idx_anomaly   (anomaly_flag),
    INDEX idx_created   (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""





def get_connection(config: dict = None):
    """
    Create and return a MySQL connection.
    Raises mysql.connector.Error on failure.
    """
    cfg = config or DB_CONFIG
    try:
        conn = mysql.connector.connect(**cfg)
        if conn.is_connected():
            logger.info(f"Connected to MySQL @ {cfg['host']}:{cfg['port']}")
            return conn
    except Error as e:
        logger.error(f"MySQL connection failed: {e}")
        raise




def setup_database(config: dict = None) -> None:
    """
    Create the database and table if they don't exist.
    Run this ONCE before any inserts.
    """
    cfg = config or DB_CONFIG

    # Connect WITHOUT specifying a database first
    init_cfg = {k: v for k, v in cfg.items() if k != "database"}
    conn = None
    try:
        conn = mysql.connector.connect(**init_cfg)
        cursor = conn.cursor()
        cursor.execute(SQL_CREATE_DATABASE)
        logger.info("Database 'data_observability' ensured.")

        cursor.execute(f"USE {cfg['database']}")
        cursor.execute(SQL_CREATE_TABLE)
        conn.commit()
        logger.info("Table 'metrics_history' ensured.")
        cursor.close()
    except Error as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()





INSERT_SQL = """
INSERT INTO metrics_history (
    dataset_name, timestamp, row_count, null_percentage,
    duplicate_count, quality_score, anomaly_flag, drift_score,
    outlier_count, volume_change_pct, max_skewness,
    entropy_delta, correlation_drift, checks_passed,
    checks_failed, total_checks
) VALUES (
    %(dataset_name)s, %(timestamp)s, %(row_count)s, %(null_percentage)s,
    %(duplicate_count)s, %(quality_score)s, %(anomaly_flag)s, %(drift_score)s,
    %(outlier_count)s, %(volume_change_pct)s, %(max_skewness)s,
    %(entropy_delta)s, %(correlation_drift)s, %(checks_passed)s,
    %(checks_failed)s, %(total_checks)s
)
"""

def insert_metrics(metrics: dict, config: dict = None) -> int:
    """
    Insert a metrics dict into metrics_history.

    Parameters
    ----------
    metrics : dict from metrics_generator.extract_metrics()
    config  : DB_CONFIG override (optional)

    Returns
    -------
    int – the new row's auto-increment ID
    """
    cfg  = config or DB_CONFIG
    conn = None
    try:
        conn   = get_connection(cfg)
        cursor = conn.cursor()
        cursor.execute(INSERT_SQL, metrics)
        conn.commit()
        new_id = cursor.lastrowid
        logger.info(f"Inserted metrics row id={new_id} | score={metrics['quality_score']}")
        cursor.close()
        return new_id
    except Error as e:
        logger.error(f"Insert failed: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()





def fetch_all_metrics(config: dict = None) -> list:
    """
    Fetch all rows from metrics_history ordered by timestamp.

    Returns
    -------
    list of dicts
    """
    cfg  = config or DB_CONFIG
    conn = None
    try:
        conn   = get_connection(cfg)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM metrics_history
            ORDER BY timestamp ASC
        """)
        rows = cursor.fetchall()
        logger.info(f"Fetched {len(rows)} metric rows.")
        cursor.close()
        return rows
    except Error as e:
        logger.error(f"Fetch failed: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()





def fetch_recent_metrics(n: int = 30, config: dict = None) -> list:
    """Fetch the most recent N metric rows."""
    cfg  = config or DB_CONFIG
    conn = None
    try:
        conn   = get_connection(cfg)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT * FROM metrics_history
            ORDER BY timestamp DESC
            LIMIT %s
        """, (n,))
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except Error as e:
        logger.error(f"Fetch recent failed: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()





def metrics_exist_today(dataset_name: str, config: dict = None) -> bool:
    """Return True if a row was already inserted today."""
    cfg  = config or DB_CONFIG
    conn = None
    try:
        conn   = get_connection(cfg)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM metrics_history
            WHERE dataset_name = %s
              AND DATE(timestamp) = CURDATE()
        """, (dataset_name,))
        count = cursor.fetchone()[0]
        cursor.close()
        return count > 0
    except Error as e:
        logger.error(f"Existence check failed: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()




if __name__ == "__main__":
    print("Setting up database...")
    setup_database()
    print("Database + table created successfully.")
    print("\nSQL for reference:")
    print(SQL_CREATE_TABLE)
