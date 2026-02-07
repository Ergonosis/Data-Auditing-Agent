# TASK 3: Reconciliation & Anomaly Detection Agents

## Objective
Implement the Reconciliation Agent (5 tools) and Anomaly Detection Agent (5 tools) - both are deterministic and run in PARALLEL with Data Quality Agent.

## Context
These are the **2nd and 3rd parallel agents**. They process ALL transactions simultaneously with Data Quality. Both are 100% deterministic (NO LLM calls) using SQL, embeddings, and pre-trained ML models.

**Critical Performance Targets**:
- Reconciliation Agent: <2 minutes for 1000 transactions
- Anomaly Detection Agent: <1.5 minutes for 1000 transactions

---

## Part A: Reconciliation Agent

### Purpose
Match transactions across multiple sources (credit card, bank, receipts, emails) using deterministic matching algorithms and Knowledge Graph entity resolution.

### Architecture
```
Reconciliation Agent
├─ Tool 1: cross_source_matcher (SQL JOIN on amount/date/vendor)
├─ Tool 2: entity_resolver_kg (KG lookup via Delta Lake tables)
├─ Tool 3: fuzzy_vendor_matcher (Levenshtein distance + embeddings)
├─ Tool 4: receipt_transaction_matcher (OCR matching)
└─ Tool 5: find_orphan_transactions (Single-source detection)
```

---

### Files to Create

#### 1. `/src/tools/reconciliation_tools.py` (~300 lines)

```python
"""Reconciliation tools for cross-source transaction matching"""

from crewai_tools import tool
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from src.tools.databricks_client import query_gold_tables
from src.utils.logging import get_logger
from difflib import SequenceMatcher
import numpy as np

logger = get_logger(__name__)

@tool("cross_source_matcher")
def cross_source_matcher(source_1: str, source_2: str, date_range: tuple) -> dict:
    """
    Match transactions from two sources based on amount, date, and vendor

    Matching criteria:
    - Amount: exact match or ±5% (for currency conversion)
    - Date: ±3 days window
    - Vendor: exact match or KG-resolved entity match

    Args:
        source_1: First source name (e.g., 'credit_card')
        source_2: Second source name (e.g., 'bank')
        date_range: (start_date, end_date) tuple

    Returns:
        {
            'matched_pairs': [
                {'source_1_id': 'cc_123', 'source_2_id': 'bank_456', 'confidence': 0.95, 'match_reason': 'exact_amount_date'},
                ...
            ],
            'unmatched_source_1': ['cc_789', ...],
            'unmatched_source_2': ['bank_999', ...],
            'match_rate': 0.88
        }
    """
    logger.info(f"Matching {source_1} vs {source_2} for date range {date_range}")

    try:
        # Query both sources
        start_date, end_date = date_range

        df1 = query_gold_tables(f"""
            SELECT txn_id, amount, vendor, vendor_id, date
            FROM gold.transactions
            WHERE source = '{source_1}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
        """)

        df2 = query_gold_tables(f"""
            SELECT txn_id, amount, vendor, vendor_id, date
            FROM gold.transactions
            WHERE source = '{source_2}'
              AND date BETWEEN '{start_date}' AND '{end_date}'
        """)

        if df1.empty or df2.empty:
            logger.warning(f"No data found for {source_1} or {source_2}")
            return {
                'matched_pairs': [],
                'unmatched_source_1': [],
                'unmatched_source_2': [],
                'match_rate': 0.0
            }

        matched_pairs = []
        matched_ids_1 = set()
        matched_ids_2 = set()

        # Convert dates to datetime if needed
        df1['date'] = pd.to_datetime(df1['date'])
        df2['date'] = pd.to_datetime(df2['date'])

        # Match on amount + date + vendor
        for idx1, row1 in df1.iterrows():
            for idx2, row2 in df2.iterrows():
                if row2['txn_id'] in matched_ids_2:
                    continue  # Already matched

                # Amount matching (±5%)
                amount_match = abs(row1['amount'] - row2['amount']) / row1['amount'] <= 0.05

                # Date matching (±3 days)
                date_diff = abs((row1['date'] - row2['date']).days)
                date_match = date_diff <= 3

                # Vendor matching (exact or KG entity)
                vendor_match = (
                    row1['vendor'] == row2['vendor'] or
                    (row1.get('vendor_id') and row1['vendor_id'] == row2.get('vendor_id'))
                )

                if amount_match and date_match and vendor_match:
                    confidence = 0.95 if date_diff == 0 and row1['amount'] == row2['amount'] else 0.85

                    matched_pairs.append({
                        'source_1_id': row1['txn_id'],
                        'source_2_id': row2['txn_id'],
                        'confidence': confidence,
                        'match_reason': f'amount_match={amount_match}, date_diff={date_diff}',
                        'amount': row1['amount'],
                        'vendor': row1['vendor']
                    })

                    matched_ids_1.add(row1['txn_id'])
                    matched_ids_2.add(row2['txn_id'])
                    break  # Move to next row1

        unmatched_1 = [txn for txn in df1['txn_id'].tolist() if txn not in matched_ids_1]
        unmatched_2 = [txn for txn in df2['txn_id'].tolist() if txn not in matched_ids_2]

        match_rate = len(matched_pairs) / len(df1) if len(df1) > 0 else 0

        result = {
            'matched_pairs': matched_pairs[:100],  # Limit output size
            'unmatched_source_1': unmatched_1[:50],
            'unmatched_source_2': unmatched_2[:50],
            'match_rate': round(match_rate, 3),
            'total_matched': len(matched_pairs),
            'total_unmatched_1': len(unmatched_1),
            'total_unmatched_2': len(unmatched_2)
        }

        logger.info(f"Matching complete: {len(matched_pairs)} matches, match rate {match_rate:.1%}")
        return result

    except Exception as e:
        logger.error(f"Cross-source matching failed: {e}")
        raise


@tool("entity_resolver_kg")
def entity_resolver_kg(vendor_name: str) -> dict:
    """
    Resolve vendor entity using Knowledge Graph (Delta Lake tables)

    Args:
        vendor_name: Raw vendor name (e.g., "AMZN MKTP US*1A2B3C4D5")

    Returns:
        {
            'canonical_entity_id': 'amazon_marketplace',
            'canonical_name': 'Amazon Marketplace',
            'aliases': ['AMZN MKTP', 'Amazon.com', 'AMAZON MKTPLACE'],
            'confidence': 0.95
        }
    """
    logger.info(f"Resolving entity for vendor: {vendor_name}")

    try:
        # Query KG entities table (Delta Lake)
        result = query_gold_tables(f"""
            SELECT entity_id, canonical_name, aliases
            FROM kg_entities
            WHERE '{vendor_name}' IN (canonical_name, aliases)
               OR aliases LIKE '%{vendor_name}%'
            LIMIT 1
        """)

        if not result.empty:
            row = result.iloc[0]
            return {
                'canonical_entity_id': row['entity_id'],
                'canonical_name': row['canonical_name'],
                'aliases': row.get('aliases', []),
                'confidence': 0.95
            }
        else:
            # No exact match - return unknown
            logger.warning(f"No KG entity found for {vendor_name}")
            return {
                'canonical_entity_id': f'unknown_{vendor_name.replace(" ", "_").lower()}',
                'canonical_name': vendor_name,
                'aliases': [],
                'confidence': 0.0
            }

    except Exception as e:
        logger.error(f"Entity resolution failed: {e}")
        return {
            'canonical_entity_id': f'unknown_{vendor_name}',
            'canonical_name': vendor_name,
            'aliases': [],
            'confidence': 0.0
        }


@tool("fuzzy_vendor_matcher")
def fuzzy_vendor_matcher(vendor_a: str, vendor_b: str) -> float:
    """
    Calculate fuzzy match similarity between two vendor names

    Uses Levenshtein distance (SequenceMatcher)

    Args:
        vendor_a: First vendor name
        vendor_b: Second vendor name

    Returns:
        Similarity score 0-1 (1 = exact match)
    """
    try:
        # Normalize (lowercase, strip whitespace)
        a = vendor_a.lower().strip()
        b = vendor_b.lower().strip()

        # Calculate similarity
        similarity = SequenceMatcher(None, a, b).ratio()

        logger.info(f"Fuzzy match '{vendor_a}' vs '{vendor_b}': {similarity:.2f}")
        return round(similarity, 3)

    except Exception as e:
        logger.error(f"Fuzzy matching failed: {e}")
        return 0.0


@tool("receipt_transaction_matcher")
def receipt_transaction_matcher(receipt_data: dict, transactions_table: str) -> dict:
    """
    Match OCR-extracted receipt to credit card transaction

    Args:
        receipt_data: {
            'vendor': str,
            'amount': float,
            'date': str (ISO format)
        }
        transactions_table: Name of transactions table

    Returns:
        {
            'matched_transaction_id': str or None,
            'confidence': float,
            'amount_delta': float,
            'date_delta_days': int
        }
    """
    logger.info(f"Matching receipt: {receipt_data}")

    try:
        vendor = receipt_data['vendor']
        amount = receipt_data['amount']
        receipt_date = pd.to_datetime(receipt_data['date'])

        # Query transactions ±7 days from receipt date
        start_date = receipt_date - timedelta(days=7)
        end_date = receipt_date + timedelta(days=7)

        df = query_gold_tables(f"""
            SELECT txn_id, vendor, amount, date
            FROM {transactions_table}
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
              AND amount BETWEEN {amount * 0.95} AND {amount * 1.05}
        """)

        if df.empty:
            logger.warning("No matching transactions found for receipt")
            return {
                'matched_transaction_id': None,
                'confidence': 0.0,
                'amount_delta': None,
                'date_delta_days': None
            }

        # Find best match
        df['date'] = pd.to_datetime(df['date'])
        df['amount_delta'] = abs(df['amount'] - amount)
        df['date_delta'] = abs((df['date'] - receipt_date).dt.days)
        df['vendor_similarity'] = df['vendor'].apply(lambda v: fuzzy_vendor_matcher.func(v, vendor))

        # Score = weighted combination
        df['score'] = (
            (1 - df['amount_delta'] / amount) * 0.4 +
            (1 - df['date_delta'] / 7) * 0.3 +
            df['vendor_similarity'] * 0.3
        )

        best_match = df.loc[df['score'].idxmax()]

        result = {
            'matched_transaction_id': best_match['txn_id'],
            'confidence': round(best_match['score'], 3),
            'amount_delta': round(best_match['amount_delta'], 2),
            'date_delta_days': int(best_match['date_delta'])
        }

        logger.info(f"Receipt matched to {result['matched_transaction_id']} (confidence: {result['confidence']})")
        return result

    except Exception as e:
        logger.error(f"Receipt matching failed: {e}")
        return {
            'matched_transaction_id': None,
            'confidence': 0.0,
            'amount_delta': None,
            'date_delta_days': None
        }


@tool("find_orphan_transactions")
def find_orphan_transactions(all_sources: list) -> dict:
    """
    Find transactions that appear in only one source (SUSPICIOUS)

    Args:
        all_sources: List of source names ['credit_card', 'bank', 'receipts']

    Returns:
        {
            'orphan_count': int,
            'orphans': [
                {'txn_id': 'cc_123', 'source': 'credit_card', 'amount': 500, 'vendor': 'Unknown Inc'},
                ...
            ]
        }
    """
    logger.info(f"Finding orphan transactions across sources: {all_sources}")

    try:
        # Query all transactions from all sources
        all_txns = []

        for source in all_sources:
            df = query_gold_tables(f"""
                SELECT txn_id, source, amount, vendor, date
                FROM gold.transactions
                WHERE source = '{source}'
            """)
            all_txns.append(df)

        # Concatenate all dataframes
        combined = pd.concat(all_txns, ignore_index=True)

        if combined.empty:
            return {'orphan_count': 0, 'orphans': []}

        # Group by (amount, vendor, date) to find transactions in only one source
        combined['key'] = combined['amount'].astype(str) + '_' + combined['vendor'] + '_' + combined['date'].astype(str)

        orphans_df = combined[combined.groupby('key')['source'].transform('nunique') == 1]

        orphans = orphans_df.to_dict('records')[:50]  # Limit to 50

        result = {
            'orphan_count': len(orphans_df),
            'orphans': orphans
        }

        logger.info(f"Found {result['orphan_count']} orphan transactions")
        return result

    except Exception as e:
        logger.error(f"Orphan detection failed: {e}")
        return {'orphan_count': 0, 'orphans': []}
```

#### 2. `/src/agents/reconciliation_agent.py` (~100 lines)

```python
"""Reconciliation Agent - matches transactions across sources"""

from crewai import Agent, Task
from src.tools.reconciliation_tools import (
    cross_source_matcher,
    entity_resolver_kg,
    fuzzy_vendor_matcher,
    receipt_transaction_matcher,
    find_orphan_transactions
)

reconciliation_agent = Agent(
    role="Transaction Reconciliation Specialist",
    goal="Match transactions across credit card, bank, email, and receipt sources with 95%+ accuracy",
    backstory="""You are an expert financial auditor with 15 years of experience in reconciliation.
    You have a keen eye for spotting discrepancies and can match transactions even when vendor names vary.""",

    tools=[
        cross_source_matcher,
        entity_resolver_kg,
        fuzzy_vendor_matcher,
        receipt_transaction_matcher,
        find_orphan_transactions
    ],

    verbose=True,
    allow_delegation=False,
    llm=None  # Pure tool-based, no LLM reasoning
)

reconciliation_task = Task(
    description="""
    Match transactions across multiple sources:
    1. Use cross_source_matcher to match credit_card vs bank transactions
    2. Use entity_resolver_kg to resolve vendor name variations
    3. Use fuzzy_vendor_matcher for typo matching
    4. Use receipt_transaction_matcher for receipt-to-transaction matching
    5. Use find_orphan_transactions to identify single-source transactions (SUSPICIOUS)

    **Output**: {
        'matched_pairs': [...],
        'unmatched_transactions': [...],  # SUSPICIOUS
        'low_confidence_matches': [...],  # confidence <0.7
        'orphan_transactions': [...]  # appear in only one source
    }
    """,

    agent=reconciliation_agent,
    expected_output="JSON with matched pairs and unmatched (suspicious) transactions"
)
```

---

## Part B: Anomaly Detection Agent

### Purpose
Detect statistical and ML-based anomalies using Isolation Forest, vendor profiles, and time-series models.

### Architecture
```
Anomaly Detection Agent
├─ Tool 1: run_isolation_forest (Pre-trained sklearn model)
├─ Tool 2: check_vendor_spending_profile (Cached stats lookup)
├─ Tool 3: detect_amount_outliers (Z-score calculation)
├─ Tool 4: time_series_deviation_check (Prophet model - optional)
└─ Tool 5: batch_anomaly_scorer (Combine all signals)
```

---

### Files to Create

#### 3. `/src/tools/anomaly_tools.py` (~250 lines)

```python
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
```

#### 4. `/src/agents/anomaly_detection_agent.py`

```python
"""Anomaly Detection Agent - detects statistical and ML anomalies"""

from crewai import Agent, Task
from src.tools.anomaly_tools import (
    run_isolation_forest,
    check_vendor_spending_profile,
    detect_amount_outliers,
    time_series_deviation_check,
    batch_anomaly_scorer
)

anomaly_agent = Agent(
    role="Anomaly Detection Specialist",
    goal="Detect statistical and ML-based anomalies in transaction patterns with 90%+ accuracy",
    backstory="""You are a data scientist specializing in fraud detection and anomaly analysis.
    You use machine learning and statistical methods to identify unusual patterns.""",

    tools=[
        run_isolation_forest,
        check_vendor_spending_profile,
        detect_amount_outliers,
        time_series_deviation_check,
        batch_anomaly_scorer
    ],

    verbose=True,
    allow_delegation=False,
    llm=None
)

anomaly_task = Task(
    description="""
    Detect anomalies in transactions using multiple methods:
    1. Run Isolation Forest on transaction features
    2. Check vendor spending profiles (z-scores)
    3. Detect amount outliers (statistical)
    4. Check time series deviations (for recurring transactions)
    5. Combine all signals into single anomaly score

    **Output**: {
        'anomaly_scores': [...],
        'flagged_transactions': [...]  # score >70
    }
    """,

    agent=anomaly_agent,
    expected_output="JSON with anomaly scores and flagged transactions"
)
```

---

## Testing Requirements

Create `/tests/test_agents/test_reconciliation_anomaly.py`:

```python
def test_cross_source_matcher():
    from src.tools.reconciliation_tools import cross_source_matcher
    result = cross_source_matcher.func('credit_card', 'bank', ('2025-02-01', '2025-02-28'))
    assert 'matched_pairs' in result
    assert 'match_rate' in result

def test_isolation_forest():
    from src.tools.anomaly_tools import run_isolation_forest
    transactions = [
        {'txn_id': 'cc_1', 'amount': 100, 'vendor_id': 'v1', 'date': '2025-02-01'},
        {'txn_id': 'cc_2', 'amount': 5000, 'vendor_id': 'v2', 'date': '2025-02-02'},
    ]
    result = run_isolation_forest.func(transactions)
    assert 'anomaly_count' in result
```

---

## Success Criteria

✅ Reconciliation agent matches transactions across sources
✅ Entity resolver queries KG tables
✅ Fuzzy matcher calculates similarity correctly
✅ Orphan detector finds single-source transactions
✅ Isolation Forest detects anomalies
✅ Vendor profile lookup works with cached stats
✅ Anomaly scorer combines signals correctly
✅ Both agents complete in target time (<2 min)
✅ All tests pass

---

## Important Notes

- **No LLM calls** - both agents are 100% deterministic
- **Performance critical** - optimize SQL queries, avoid nested loops where possible
- **Vendor profiles** - assume `gold.vendor_profiles` table exists (or create mock)
- **KG tables** - assume `kg_entities` Delta table exists (or use empty default)

---

## Dependencies
- `crewai`, `crewai-tools`, `pandas`, `numpy`, `scikit-learn`
- Existing modules: `src.tools.databricks_client`, `src.utils.*`

---

## Estimated Effort
~550 lines of code, 2-3 hours for both agents.
