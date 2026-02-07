#!/usr/bin/env python3
"""
Weekly feedback analysis job - auto-tunes rules based on false positives

Runs every Sunday 2:00 AM
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.databricks_client import query_gold_tables
from src.utils.config_loader import load_config, save_config
from src.utils.logging import get_logger
from datetime import datetime

logger = get_logger(__name__)


def analyze_false_positives():
    """
    Main feedback analysis logic

    1. Query all reviewed flags from past 7 days
    2. Calculate false positive rates by vendor and rule
    3. Auto-whitelist vendors with >80% FP rate
    4. Adjust rule thresholds for >50% FP rate
    5. Save updated config
    """
    logger.info("=" * 60)
    logger.info("WEEKLY FEEDBACK ANALYSIS")
    logger.info("=" * 60)

    try:
        # Query false positives
        false_positives = query_gold_tables("""
            SELECT flag_id, txn_id, severity_level, vendor_id, amount,
                   explanation, human_decision
            FROM flags
            WHERE reviewed = true
              AND human_decision = 'false_positive'
              AND created_at > CURRENT_DATE - INTERVAL 7 DAYS
        """)

        all_flags = query_gold_tables("""
            SELECT flag_id, vendor_id, explanation
            FROM flags
            WHERE created_at > CURRENT_DATE - INTERVAL 7 DAYS
        """)

        logger.info(f"📊 Analyzing {len(false_positives)} false positives from {len(all_flags)} total flags")

        if false_positives.empty or all_flags.empty:
            logger.info("No data to analyze this week")
            return

        # Load config
        config = load_config()
        changes_made = []

        # Strategy 1: Auto-whitelist vendors with >80% FP rate
        vendor_fp_rates = (
            false_positives.groupby('vendor_id').size() /
            all_flags.groupby('vendor_id').size()
        )

        for vendor_id, fp_rate in vendor_fp_rates.items():
            if fp_rate > 0.80 and vendor_id not in config.get('whitelisted_vendors', []):
                config.setdefault('whitelisted_vendors', []).append(vendor_id)
                logger.info(f"✅ Auto-whitelisted vendor {vendor_id} (FP rate: {fp_rate:.1%})")
                changes_made.append(f"whitelisted_{vendor_id}")

        # Strategy 2: Adjust rule thresholds
        rule_fp_rates = (
            false_positives.groupby('explanation').size() /
            all_flags.groupby('explanation').size()
        )

        for rule_name, fp_rate in rule_fp_rates.items():
            if fp_rate > 0.50 and 'amount_outlier' in rule_name:
                # Increase sigma threshold
                old_sigma = config['rules']['anomaly_detection']['amount_outlier']['sigma']
                new_sigma = old_sigma + 0.5
                config['rules']['anomaly_detection']['amount_outlier']['sigma'] = new_sigma

                logger.info(f"✅ Increased amount_outlier sigma: {old_sigma} → {new_sigma} (FP rate: {fp_rate:.1%})")
                changes_made.append(f"sigma_{old_sigma}_to_{new_sigma}")

        # Save updated config
        if changes_made:
            config['last_updated'] = datetime.now().isoformat()
            save_config('config/rules.yaml', config)
            logger.info(f"✅ Config updated with {len(changes_made)} changes")
        else:
            logger.info("No rule changes needed this week")

        logger.info("=" * 60)
        logger.info("FEEDBACK ANALYSIS COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Feedback analysis failed: {e}")
        raise


if __name__ == "__main__":
    analyze_false_positives()
