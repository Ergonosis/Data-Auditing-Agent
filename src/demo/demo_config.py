"""Demo-specific configuration helpers"""

import os
from typing import Dict, Any


def is_demo_mode() -> bool:
    """
    Check if system is running in demo mode

    Returns:
        True if demo mode is active
    """
    return (
        os.getenv("DEMO_MODE") == "true" or
        os.getenv("ENVIRONMENT") == "demo"
    )


def get_demo_config_overrides() -> Dict[str, Any]:
    """
    Get demo-specific configuration overrides

    These adjust thresholds and settings to work better with
    the smaller RIA demo dataset.

    Returns:
        Dictionary of config overrides
    """
    return {
        'rules': {
            'data_quality': {
                'completeness_threshold': 0.85,  # Slightly lower for demo (missing receipts expected)
            },
            'anomaly_detection': {
                'amount_outlier': {
                    'sigma': 2.0,  # More sensitive for small dataset
                    'min_transactions': 3,  # Lower minimum for demo
                },
                'isolation_forest': {
                    'contamination': 0.10,  # Higher expected anomaly rate in demo
                }
            },
            'escalation': {
                'severity_thresholds': {
                    'CRITICAL': {'min_score': 75},  # Slightly lower threshold
                    'WARNING': {'min_score': 45},
                    'INFO': {'min_score': 20}
                }
            }
        },
        'whitelisted_vendors': [
            # SaaS vendors that don't have paper receipts
            'AWS',
            'Slack',
            'Microsoft',
            'Salesforce',
            'DocuSign',
            'Zoom',
            'Google',
            'Adobe'
        ]
    }


def get_demo_data_dir() -> str:
    """
    Get demo data directory path

    Returns:
        Path to demo data directory
    """
    return os.getenv("DEMO_DATA_DIR", "ria_data")
