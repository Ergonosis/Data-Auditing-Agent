"""Isolated test for escalation agent with fixed tool schemas"""
import os
from dotenv import load_dotenv

# Load environment variables before importing agents
load_dotenv()
os.environ['TEST_MODE'] = 'true'

from src.agents.escalation_agent import escalation_agent, escalation_task
from src.tools.escalation_tools import get_test_mode_flags, clear_test_mode_flags
from crewai import Crew, Process


def main():
    print("\n" + "="*60)
    print("ISOLATED ESCALATION AGENT TEST")
    print("="*60)

    # Clear any existing flags
    clear_test_mode_flags()
    print("\n✅ Cleared existing test flags")

    # Minimal test data
    suspicious_txns = [
        {
            'txn_id': 'cc_001',
            'vendor': 'Unknown Inc',
            'amount': 5000.00,
            'date': '2024-01-15',
            'source': 'credit_card',
            'account': 'ACCT-001',
            'category': 'Unknown',
            'description': 'Large unknown transaction'
        }
    ]

    parallel_results = {
        'data_quality': {
            'quality_score': 0.85,
            'incomplete_records': [],
            'duplicate_records': []
        },
        'reconciliation': {
            'matched': False,
            'match_rate': 0.5,
            'suspicious_transactions': ['cc_001']
        }
    }

    print("\n📊 Test Data:")
    print(f"   - Suspicious transactions: {len(suspicious_txns)}")
    print(f"   - Transaction ID: {suspicious_txns[0]['txn_id']}")
    print(f"   - Amount: ${suspicious_txns[0]['amount']}")
    print(f"   - Reconciliation matched: {parallel_results['reconciliation']['matched']}")

    # Create crew with only escalation agent
    crew = Crew(
        agents=[escalation_agent],
        tasks=[escalation_task],
        process=Process.sequential,
        verbose=True
    )

    print("\n" + "="*60)
    print("🚀 Starting isolated escalation agent test...")
    print("="*60 + "\n")

    try:
        result = crew.kickoff(inputs={
            'suspicious_transactions': suspicious_txns,
            'audit_run_id': 'test-run-isolated-001',
            'parallel_results': parallel_results
        })

        print("\n" + "="*60)
        print("✅ Agent execution completed")
        print("="*60)
        print(f"\nResult: {result}")

        # Verify flags were created
        flags = get_test_mode_flags()
        print("\n" + "="*60)
        print("📋 FLAG CREATION RESULTS")
        print("="*60)
        print(f"\nFlags created: {len(flags)}")

        if len(flags) > 0:
            print("\n✅ SUCCESS - Flag creation working!")
            print("\nFlag details:")
            for i, flag in enumerate(flags, 1):
                print(f"\n  Flag {i}:")
                print(f"    - Flag ID: {flag['flag_id']}")
                print(f"    - Transaction ID: {flag['txn_id']}")
                print(f"    - Severity: {flag['severity']}")
                print(f"    - Explanation: {flag['explanation'][:100]}...")

            print("\n" + "="*60)
            print("✅ TEST PASSED - Escalation agent successfully created flags")
            print("="*60 + "\n")
            return 0
        else:
            print("\n❌ FAILURE - No flags created")
            print("   This indicates the escalation agent did not call create_audit_flag")
            print("\n" + "="*60)
            print("❌ TEST FAILED")
            print("="*60 + "\n")
            return 1

    except Exception as e:
        print("\n" + "="*60)
        print("❌ ERROR - Test execution failed")
        print("="*60)
        print(f"\nError: {e}")
        print(f"\nError type: {type(e).__name__}")

        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

        print("\n" + "="*60)
        print("❌ TEST FAILED WITH EXCEPTION")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
