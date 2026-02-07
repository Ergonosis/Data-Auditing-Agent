"""Anomaly detection tools using ML and statistical methods"""

from crewai_tools import tool
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.tools.databricks_client import query_gold_tables
from src.utils.logging import get_logger
from sklearn.ensemble import IsolationForest
import pickle
from pathlib import Path

logger = get_logger(__name__)

# Load pre-trained model (if exists)
MODEL_PATH = Path("src/ml/isolation_forest_model.pkl")
if MODEL_PATH.exists():
    with open(MODEL_PATH, 'rb') as f:
        isolation_forest_model = pickle.load(f)
else:
    # Create default model
    isolation_forest_model = IsolationForest(contamination=0.05, random_state=42)
    logger.warning("No pre-trained model found, using default Isolation Forest")


@tool("run_isolation_forest")
def run_isolation_forest(transactions: list) -> dict:
    """
    Run Isolation Forest anomaly detection on transaction features

    Features used:
    - amount (log-scaled)
    - vendor_id (encoded)
    - day_of_week, day_of_month
    - time_since_last_transaction_from_vendor

    Args:
        transactions: List of transaction dicts with keys: txn_id, amount, vendor_id, date

    Returns:
        {
            'anomaly_scores': [
                {'txn_id': 'x', 'score': -0.8, 'is_anomaly': True},
                ...
            ],
            'anomaly_count': int
        }
    """
    logger.info(f"Running Isolation Forest on {len(transactions)} transactions")

    try:
        df = pd.DataFrame(transactions)

        if df.empty:
            return {'anomaly_scores': [], 'anomaly_count': 0}

        # Feature engineering
        df['date'] = pd.to_datetime(df['date'])
        df['log_amount'] = np.log1p(df['amount'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day

        # Vendor encoding (simple hash for now)
        df['vendor_encoded'] = df['vendor_id'].fillna('unknown').apply(lambda x: hash(x) % 1000)

        # Features for model
        features = ['log_amount', 'vendor_encoded', 'day_of_week', 'day_of_month']
        X = df[features].fillna(0)

        # Predict anomaly scores
        if hasattr(isolation_forest_model, 'decision_function'):
            scores = isolation_forest_model.decision_function(X)
        else:
            # Train model if not fitted
            isolation_forest_model.fit(X)
            scores = isolation_forest_model.decision_function(X)

        df['anomaly_score'] = scores
        df['is_anomaly'] = scores < -0.5  # Threshold for anomaly

        # Prepare output
        anomaly_scores = df[['txn_id', 'anomaly_score', 'is_anomaly']].to_dict('records')

        result = {
            'anomaly_scores': anomaly_scores[:100],  # Limit output
            'anomaly_count': int(df['is_anomaly'].sum())
        }

        logger.info(f"Isolation Forest complete: {result['anomaly_count']} anomalies detected")
        return result

    except Exception as e:
        logger.error(f"Isolation Forest failed: {e}")
        return {'anomaly_scores': [], 'anomaly_count': 0}


@tool("check_vendor_spending_profile")
def check_vendor_spending_profile(vendor_id: str, amount: float) -> dict:
    """
    Check if transaction amount is typical for vendor

    Args:
        vendor_id: Canonical vendor ID
        amount: Transaction amount

    Returns:
        {
            'is_outlier': bool,
            'z_score': float,
            'vendor_mean': float,
            'vendor_std': float,
            'explanation': str
        }
    """
    logger.info(f"Checking spending profile for {vendor_id}, amount ${amount}")

    try:
        # Query vendor profile from cached stats
        profile = query_gold_tables(f"""
            SELECT mean_amount, std_dev, frequency
            FROM gold.vendor_profiles
            WHERE vendor_id = '{vendor_id}'
        """)

        if profile.empty:
            logger.warning(f"No profile found for {vendor_id}")
            return {
                'is_outlier': False,
                'z_score': 0.0,
                'vendor_mean': amount,
                'vendor_std': 0.0,
                'explanation': 'No historical data for vendor'
            }

        mean = profile['mean_amount'].iloc[0]
        std = profile['std_dev'].iloc[0]

        # Calculate z-score
        z_score = (amount - mean) / std if std > 0 else 0

        is_outlier = abs(z_score) > 3.0

        explanation = f"Amount ${amount:.2f} is {z_score:.1f}σ from mean ${mean:.2f}"

        result = {
            'is_outlier': is_outlier,
            'z_score': round(z_score, 2),
            'vendor_mean': round(mean, 2),
            'vendor_std': round(std, 2),
            'explanation': explanation
        }

        logger.info(f"Vendor profile check: {'OUTLIER' if is_outlier else 'NORMAL'}")
        return result

    except Exception as e:
        logger.error(f"Vendor profile check failed: {e}")
        return {
            'is_outlier': False,
            'z_score': 0.0,
            'vendor_mean': 0.0,
            'vendor_std': 0.0,
            'explanation': f'Error: {e}'
        }


@tool("detect_amount_outliers")
def detect_amount_outliers(transactions: list) -> dict:
    """
    Simple statistical outlier detection using z-scores

    Args:
        transactions: List of transaction dicts

    Returns:
        {
            'outliers': [{'txn_id': 'x', 'amount': 5000, 'z_score': 4.2}, ...],
            'outlier_count': int
        }
    """
    logger.info(f"Detecting amount outliers in {len(transactions)} transactions")

    try:
        df = pd.DataFrame(transactions)

        if df.empty or 'amount' not in df.columns:
            return {'outliers': [], 'outlier_count': 0}

        # Calculate z-scores
        mean = df['amount'].mean()
        std = df['amount'].std()

        df['z_score'] = (df['amount'] - mean) / std if std > 0 else 0
        df['is_outlier'] = abs(df['z_score']) > 2.0  # 2 sigma threshold

        outliers = df[df['is_outlier']][['txn_id', 'amount', 'z_score']].to_dict('records')

        result = {
            'outliers': outliers[:50],
            'outlier_count': len(outliers)
        }

        logger.info(f"Found {result['outlier_count']} amount outliers")
        return result

    except Exception as e:
        logger.error(f"Amount outlier detection failed: {e}")
        return {'outliers': [], 'outlier_count': 0}


@tool("time_series_deviation_check")
def time_series_deviation_check(recurring_transactions: list) -> dict:
    """
    Check for deviations in recurring transactions (simplified version)

    Args:
        recurring_transactions: List of recurring transactions (same vendor, similar amounts)

    Returns:
        {
            'deviations': [{'txn_id': 'x', 'expected': 100, 'actual': 150, 'deviation_pct': 50}, ...],
            'deviation_count': int
        }
    """
    logger.info(f"Checking time series deviations for {len(recurring_transactions)} transactions")

    try:
        df = pd.DataFrame(recurring_transactions)

        if df.empty or len(df) < 3:
            return {'deviations': [], 'deviation_count': 0}

        # Calculate rolling mean
        df = df.sort_values('date')
        df['rolling_mean'] = df['amount'].rolling(window=3, min_periods=1).mean()
        df['deviation_pct'] = abs((df['amount'] - df['rolling_mean']) / df['rolling_mean']) * 100

        # Flag deviations >20%
        df['is_deviation'] = df['deviation_pct'] > 20

        deviations = df[df['is_deviation']][['txn_id', 'amount', 'rolling_mean', 'deviation_pct']].to_dict('records')

        result = {
            'deviations': deviations[:20],
            'deviation_count': len(deviations)
        }

        logger.info(f"Found {result['deviation_count']} time series deviations")
        return result

    except Exception as e:
        logger.error(f"Time series check failed: {e}")
        return {'deviations': [], 'deviation_count': 0}


@tool("batch_anomaly_scorer")
def batch_anomaly_scorer(transactions: list) -> dict:
    """
    Combine all anomaly signals into single score (0-100)

    Args:
        transactions: List of transactions with anomaly data attached

    Returns:
        {
            'scored_transactions': [
                {'txn_id': 'x', 'anomaly_score': 85, 'reasons': ['isolation_forest', 'vendor_outlier']},
                ...
            ],
            'high_risk_count': int  # score >70
        }
    """
    logger.info(f"Scoring {len(transactions)} transactions")

    try:
        scored = []

        for txn in transactions:
            score = 0
            reasons = []

            # Isolation Forest signal
            if txn.get('is_anomaly_if'):
                score += 30
                reasons.append('isolation_forest')

            # Vendor outlier signal
            if txn.get('is_vendor_outlier'):
                score += 30
                reasons.append('vendor_outlier')

            # Amount outlier signal
            if txn.get('is_amount_outlier'):
                score += 20
                reasons.append('amount_outlier')

            # Time series deviation
            if txn.get('is_time_series_deviation'):
                score += 20
                reasons.append('time_series_deviation')

            scored.append({
                'txn_id': txn['txn_id'],
                'anomaly_score': min(score, 100),
                'reasons': reasons
            })

        high_risk = [s for s in scored if s['anomaly_score'] > 70]

        result = {
            'scored_transactions': scored[:100],
            'high_risk_count': len(high_risk)
        }

        logger.info(f"Scoring complete: {result['high_risk_count']} high-risk transactions")
        return result

    except Exception as e:
        logger.error(f"Batch scoring failed: {e}")
        return {'scored_transactions': [], 'high_risk_count': 0}
