#!/usr/bin/env python3
"""
Quick verification script to check demo setup without running full pipeline.
This script only tests data loading and configuration, not the full orchestrator.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set demo mode
os.environ["DEMO_MODE"] = "true"
os.environ["ENVIRONMENT"] = "demo"
os.environ["STATE_BACKEND"] = "memory"


def test_demo_data_loading():
    """Test that demo data can be loaded"""
    print("=" * 70)
    print("Testing Demo Data Loading")
    print("=" * 70)

    try:
        from src.demo.csv_data_loader import DemoDataLoader

        loader = DemoDataLoader()
        stats = loader.get_summary_stats()

        print(f"✅ Demo data loader initialized")
        print(f"📁 Data directory: {stats['data_dir']}")
        print(f"\n📊 Record counts:")
        print(f"  • Clients: {stats['clients']:,}")
        print(f"  • Credit Card Expenses: {stats['credit_card_expenses']:,}")
        print(f"  • Bank Transactions: {stats['bank_transactions']:,}")
        print(f"  • Receipts: {stats['receipts']:,}")

        # Test transaction loading
        transactions = loader.get_transactions_for_audit()
        print(f"\n✅ Transactions loaded: {len(transactions):,}")
        print(f"  Columns: {list(transactions.columns)}")

        return True

    except Exception as e:
        print(f"❌ Demo data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demo_databricks_client():
    """Test Databricks client in demo mode"""
    print("\n" + "=" * 70)
    print("Testing Databricks Client (Demo Mode)")
    print("=" * 70)

    try:
        from src.tools.databricks_client import get_databricks_connection, query_gold_tables

        conn = get_databricks_connection()
        print(f"✅ Connection mode: {conn}")

        if conn != "DEMO_MODE":
            print(f"⚠️  Expected DEMO_MODE, got: {conn}")
            return False

        # Test query routing
        df = query_gold_tables("SELECT * FROM gold.recent_transactions")
        print(f"✅ Query executed: returned {len(df):,} rows")
        print(f"  Columns: {list(df.columns)}")

        return True

    except Exception as e:
        print(f"❌ Databricks client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_demo_config():
    """Test demo configuration loading"""
    print("\n" + "=" * 70)
    print("Testing Demo Configuration")
    print("=" * 70)

    try:
        from src.utils.config_loader import load_config

        config = load_config()
        print(f"✅ Config loaded: version {config['version']}")

        if 'demo' not in config['version']:
            print(f"⚠️  Expected demo config version, got: {config['version']}")

        print(f"✅ Whitelisted vendors: {len(config.get('whitelisted_vendors', []))}")
        print(f"  Sample: {config.get('whitelisted_vendors', [])[:3]}")

        print(f"✅ Domain configs: {list(config['domain_configs'].keys())}")

        return True

    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_manager():
    """Test in-memory state manager"""
    print("\n" + "=" * 70)
    print("Testing State Manager (In-Memory)")
    print("=" * 70)

    try:
        from src.orchestrator.state_manager import save_workflow_state, restore_workflow_state

        test_state = {
            'status': 'test',
            'test_data': 'demo_verification'
        }

        save_workflow_state("test-123", test_state)
        print("✅ State saved (in-memory)")

        restored = restore_workflow_state("test-123")
        print(f"✅ State restored: {restored}")

        if restored == test_state:
            print("✅ State matches")
            return True
        else:
            print(f"⚠️  State mismatch: {restored} != {test_state}")
            return False

    except Exception as e:
        print(f"❌ State manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests"""
    print("\n🎯 Ergonosis Demo Setup Verification")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python: {sys.version.split()[0]}")

    results = {
        'Data Loading': test_demo_data_loading(),
        'Databricks Client': test_demo_databricks_client(),
        'Configuration': test_demo_config(),
        'State Manager': test_state_manager()
    }

    print("\n" + "=" * 70)
    print("Verification Summary")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 All verification tests passed!")
        print("\n💡 Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Set up .env file with API keys")
        print("  3. Run full demo: python scripts/run_demo.py --dry-run")
        print("  4. Run audit: python scripts/run_demo.py")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
