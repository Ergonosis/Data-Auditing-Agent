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
