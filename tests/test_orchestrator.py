"""Tests for orchestrator and main workflow"""

import pytest
from src.orchestrator.orchestrator_agent import AuditOrchestrator
from src.orchestrator.retry_handler import retry_with_exponential_backoff
from src.utils.errors import AuditSystemError


def test_orchestrator_initialization():
    """Test that orchestrator initializes correctly"""
    orchestrator = AuditOrchestrator()
    assert orchestrator.audit_run_id is not None
    assert orchestrator.config is not None
    assert orchestrator.start_time is None


def test_full_audit_cycle():
    """Integration test - full audit cycle with mock data"""
    orchestrator = AuditOrchestrator()

    # This will use mock data adapter
    results = orchestrator.run_audit_cycle()

    assert 'audit_run_id' in results
    assert 'status' in results
    assert results['status'] in ['completed', 'failed']
    assert 'transaction_count' in results
    assert 'flags_created' in results


def test_retry_handler():
    """Test retry logic with exponential backoff"""
    attempts = []

    def failing_func():
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception("Test failure")
        return "success"

    result = retry_with_exponential_backoff(failing_func, max_retries=5, base_delay=0)
    assert result == "success"
    assert len(attempts) == 3


def test_retry_handler_exhaustion():
    """Test that retry handler raises error after max attempts"""
    def always_fail():
        raise Exception("Always fails")

    with pytest.raises(AuditSystemError):
        retry_with_exponential_backoff(always_fail, max_retries=3, base_delay=0)


def test_merge_suspicious_results():
    """Test merging results from parallel agents"""
    import pandas as pd

    orchestrator = AuditOrchestrator()

    # Mock parallel results
    parallel_results = {
        'reconciliation': {
            'unmatched_transactions': [
                {'txn_id': 'txn_001'},
                {'txn_id': 'txn_002'}
            ]
        },
        'anomaly': {
            'flagged_transactions': [
                {'txn_id': 'txn_003'},
                {'txn_id': 'txn_004'}
            ]
        },
        'data_quality': {
            'incomplete_records': ['txn_005']
        }
    }

    # Mock transactions dataframe
    transactions = pd.DataFrame({
        'txn_id': ['txn_001', 'txn_002', 'txn_003', 'txn_004', 'txn_005', 'txn_006'],
        'amount': [100, 200, 300, 400, 500, 600]
    })

    suspicious = orchestrator._merge_suspicious_results(parallel_results, transactions)

    assert len(suspicious) == 5
    suspicious_ids = [t['txn_id'] for t in suspicious]
    assert 'txn_001' in suspicious_ids
    assert 'txn_002' in suspicious_ids
    assert 'txn_003' in suspicious_ids
    assert 'txn_004' in suspicious_ids
    assert 'txn_005' in suspicious_ids
    assert 'txn_006' not in suspicious_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
