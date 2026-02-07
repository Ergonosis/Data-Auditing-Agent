#!/usr/bin/env python
"""
Demo runner script for Ergonosis Auditing System

This script runs the full audit pipeline using RIA CSV demo data.
It validates data exists, sets up demo mode, and provides detailed output.

Usage:
    python scripts/run_demo.py              # Run full demo audit
    python scripts/run_demo.py --dry-run    # Preview data only (no audit)
    python scripts/run_demo.py --limit 100  # Process only first 100 transactions
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set demo mode environment variables
os.environ["DEMO_MODE"] = "true"
os.environ["ENVIRONMENT"] = "demo"
os.environ["STATE_BACKEND"] = "memory"
os.environ["LOG_LEVEL"] = "INFO"

from src.demo.csv_data_loader import DemoDataLoader
from src.orchestrator.orchestrator_agent import AuditOrchestrator
from src.utils.logging import get_logger

logger = get_logger(__name__)


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def validate_demo_data(data_dir: str = "ria_data") -> bool:
    """
    Validate that demo data files exist

    Args:
        data_dir: Path to demo data directory

    Returns:
        True if all required files exist
    """
    data_path = Path(data_dir)
    required_files = [
        "ria_clients.csv",
        "ria_bank_transactions.csv",
        "ria_credit_card_expenses_with_cardholders.csv",
        "ria_receipts_travel_and_business_dev.csv"
    ]

    print_header("Demo Data Validation")

    if not data_path.exists():
        print(f"❌ Demo data directory not found: {data_dir}")
        return False

    print(f"✅ Demo data directory found: {data_path.absolute()}")

    missing_files = []
    for filename in required_files:
        filepath = data_path / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✅ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {filename} (missing)")
            missing_files.append(filename)

    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False

    print("\n✅ All demo data files present")
    return True


def show_data_summary():
    """Display summary statistics of demo data"""
    print_header("Demo Data Summary")

    try:
        loader = DemoDataLoader()
        stats = loader.get_summary_stats()

        print(f"📁 Data Directory: {stats['data_dir']}")
        print(f"\n📊 Record Counts:")
        print(f"  • Clients: {stats['clients']:,}")
        print(f"  • Bank Transactions: {stats['bank_transactions']:,}")
        print(f"  • Credit Card Expenses: {stats['credit_card_expenses']:,}")
        print(f"  • Receipts: {stats['receipts']:,}")

        # Calculate receipt coverage
        if stats['credit_card_expenses'] > 0:
            receipt_coverage = (stats['receipts'] / stats['credit_card_expenses']) * 100
            print(f"\n📝 Receipt Coverage: {receipt_coverage:.1f}%")
            print(f"  ({stats['receipts']:,} receipts / {stats['credit_card_expenses']:,} expenses)")

        # Show date ranges
        print(f"\n📅 Date Ranges:")
        if not loader.credit_card_expenses.empty:
            cc_dates = loader.credit_card_expenses['expense_date']
            print(f"  • Credit Cards: {cc_dates.min().date()} to {cc_dates.max().date()}")

        if not loader.bank_transactions.empty:
            bank_dates = loader.bank_transactions['transaction_date']
            print(f"  • Bank: {bank_dates.min().date()} to {bank_dates.max().date()}")

        if not loader.receipts.empty:
            receipt_dates = loader.receipts['receipt_date']
            print(f"  • Receipts: {receipt_dates.min().date()} to {receipt_dates.max().date()}")

    except Exception as e:
        print(f"❌ Error loading demo data: {e}")
        sys.exit(1)


def run_demo_audit(limit: int = None):
    """
    Run full audit cycle in demo mode

    Args:
        limit: Optional limit on number of transactions to process
    """
    print_header("Running Demo Audit")

    # Set transaction limit if specified
    if limit:
        print(f"⚙️  Processing limit: {limit} transactions")
        os.environ["DEMO_TRANSACTION_LIMIT"] = str(limit)

    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🚀 Initializing orchestrator...")

    try:
        orchestrator = AuditOrchestrator()
        print(f"  Audit Run ID: {orchestrator.audit_run_id}")

        print("\n🔄 Running audit cycle...")
        print("  (This may take 2-5 minutes depending on LLM API speed)")

        start_time = datetime.now()
        results = orchestrator.run_audit_cycle()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Display results
        print_header("Audit Results")

        print(f"✅ Status: {results['status'].upper()}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        print(f"\n📊 Transactions:")
        print(f"  • Total Processed: {results['transaction_count']:,}")

        if 'suspicious_count' in results:
            print(f"  • Suspicious: {results['suspicious_count']:,}")

        print(f"  • Flags Created: {results['flags_created']:,}")

        if results['transaction_count'] > 0:
            flag_rate = (results['flags_created'] / results['transaction_count']) * 100
            print(f"  • Flag Rate: {flag_rate:.1f}%")

        # Show estimated costs
        print(f"\n💰 Estimated Costs:")
        print(f"  • LLM API calls: ~${(results['flags_created'] * 0.001):.3f}")
        print(f"  (Actual costs may vary based on LLM usage)")

        print_header("Demo Audit Complete")
        print(f"\n✅ Successfully completed demo audit!")
        print(f"📋 Created {results['flags_created']} audit flags")
        print(f"🆔 Audit Run ID: {results['audit_run_id']}")

        return results

    except Exception as e:
        print(f"\n❌ Audit failed: {e}")
        logger.error(f"Demo audit failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Ergonosis Auditing System in demo mode with RIA data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Preview demo data without running audit"
    )

    parser.add_argument(
        '--limit',
        type=int,
        help="Limit number of transactions to process (for faster testing)"
    )

    parser.add_argument(
        '--data-dir',
        default="ria_data",
        help="Path to demo data directory (default: ria_data)"
    )

    args = parser.parse_args()

    # Set data directory
    os.environ["DEMO_DATA_DIR"] = args.data_dir

    print_header("Ergonosis Auditing System - Demo Mode")
    print(f"🎯 Mode: DEMO (using CSV files)")
    print(f"📁 Data Directory: {args.data_dir}")

    # Validate data exists
    if not validate_demo_data(args.data_dir):
        print("\n❌ Demo data validation failed. Please ensure RIA CSV files are present.")
        sys.exit(1)

    # Show data summary
    show_data_summary()

    # Run audit or exit if dry-run
    if args.dry_run:
        print_header("Dry Run Complete")
        print("✅ Demo data validated successfully")
        print("💡 Run without --dry-run to execute full audit")
    else:
        run_demo_audit(limit=args.limit)


if __name__ == "__main__":
    main()
