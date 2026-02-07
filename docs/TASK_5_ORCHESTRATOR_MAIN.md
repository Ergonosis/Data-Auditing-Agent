# TASK 5: Orchestrator, Logging Agent & Main Entry Point

## Objective
Implement the Orchestrator Agent (CrewAI Manager), Logging Agent, main entry point, and feedback analyzer script to complete the full audit system.

## Context
This is the **final integration task** that brings all agents together. The Orchestrator coordinates the workflow, manages state, handles failures, and dispatches agents in the correct order (3 parallel → 2 sequential → logging throughout).

---

## Part A: Logging Agent

### Purpose
Continuous background logging of all agent decisions and tool calls to append-only audit trail.

### Files to Create

#### 1. `/src/tools/logging_tools.py` (~150 lines)

```python
"""Audit logging tools for compliance and transparency"""

from crewai_tools import tool
from typing import Dict, Any, List
from datetime import datetime
from src.utils.logging import get_logger
import json

logger = get_logger(__name__)

# In-memory audit trail (in production, this would write to database)
AUDIT_TRAIL = []

@tool("log_agent_decision")
def log_agent_decision(
    agent_name: str,
    action: str,
    input_data: dict,
    output_data: dict,
    metadata: dict
) -> None:
    """
    Log agent decision with full context

    Args:
        agent_name: Name of agent
        action: Action/tool called
        input_data: Input parameters
        output_data: Output results
        metadata: Additional metadata (execution time, tokens, cost)
    """
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'agent_name': agent_name,
        'action': action,
        'input_data': input_data,
        'output_data': output_data,
        'metadata': metadata
    }

    AUDIT_TRAIL.append(log_entry)
    logger.info(f"Logged {agent_name} decision", action=action)


@tool("create_audit_trail_entry")
def create_audit_trail_entry(flag_id: str, decision_chain: list) -> None:
    """
    Link flag to decision chain

    Args:
        flag_id: Flag UUID
        decision_chain: List of decisions leading to flag
    """
    entry = {
        'flag_id': flag_id,
        'decision_chain': decision_chain,
        'timestamp': datetime.now().isoformat()
    }

    AUDIT_TRAIL.append(entry)
    logger.info(f"Created audit trail for flag {flag_id}")


@tool("get_audit_trail")
def get_audit_trail(audit_run_id: str) -> list:
    """
    Retrieve audit trail for specific run

    Args:
        audit_run_id: Audit run ID

    Returns:
        List of audit trail entries
    """
    # In production, query from database
    # For now, return in-memory trail
    return AUDIT_TRAIL


@tool("generate_lineage_trace")
def generate_lineage_trace(transaction_id: str) -> dict:
    """
    Generate data lineage graph for transaction

    Args:
        transaction_id: Transaction ID

    Returns:
        Lineage graph structure
    """
    # Simplified lineage
    return {
        'transaction_id': transaction_id,
        'lineage': [
            {'stage': 'Bronze.credit_cards', 'row_id': 'row_456'},
            {'stage': 'Silver.transactions_cleaned', 'row_id': 'row_789'},
            {'stage': 'Gold.transactions_enriched', 'row_id': 'row_123'},
            {'stage': 'Audit.flagged', 'flag_id': 'flag_xxx'}
        ]
    }
```

#### 2. `/src/agents/logging_agent.py`

```python
"""Logging Agent - records all decisions for transparency"""

from crewai import Agent, Task
from src.tools.logging_tools import (
    log_agent_decision,
    create_audit_trail_entry,
    get_audit_trail,
    generate_lineage_trace
)

logging_agent = Agent(
    role="Audit Trail Recorder",
    goal="Record every decision and action for compliance and debugging",
    backstory="""You are a compliance officer ensuring full transparency and auditability.
    Every action must be logged for regulatory requirements.""",

    tools=[
        log_agent_decision,
        create_audit_trail_entry,
        get_audit_trail,
        generate_lineage_trace
    ],

    verbose=True,
    allow_delegation=False,
    llm=None
)

logging_task = Task(
    description="Continuously log all agent decisions throughout audit cycle",
    agent=logging_agent,
    expected_output="Confirmation of logging completion"
)
```

---

## Part B: Orchestrator Agent

### Purpose
Master coordinator that manages the entire audit workflow, dispatches agents, handles failures, and tracks state.

### Files to Create

#### 3. `/src/orchestrator/__init__.py`
```python
"""Orchestrator module"""
```

#### 4. `/src/orchestrator/retry_handler.py` (~80 lines)

```python
"""Retry logic with exponential backoff"""

import time
from typing import Callable, Any
from src.utils.logging import get_logger
from src.utils.errors import AuditSystemError

logger = get_logger(__name__)

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 5,
    base_delay: int = 30,
    max_delay: int = 480,
    *args,
    **kwargs
) -> Any:
    """
    Retry function with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Base delay in seconds (30s)
        max_delay: Max delay cap (480s = 8 min)
        *args, **kwargs: Arguments to pass to func

    Returns:
        Function result

    Raises:
        AuditSystemError: If all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} retry attempts exhausted")
                raise AuditSystemError(f"Failed after {max_retries} attempts: {e}")

            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
            time.sleep(delay)
```

#### 5. `/src/orchestrator/orchestrator_agent.py` (~300 lines)

```python
"""Orchestrator Agent - master coordinator for audit workflow"""

from crewai import Crew, Process
import uuid
from datetime import datetime
from typing import Dict, Any, List
from src.tools.databricks_client import query_gold_tables, get_last_audit_timestamp, check_databricks_health
from src.orchestrator.state_manager import save_workflow_state, restore_workflow_state, mark_audit_complete
from src.orchestrator.retry_handler import retry_with_exponential_backoff
from src.utils.logging import get_logger
from src.utils.errors import AuditSystemError, DatabricksConnectionError
from src.utils.config_loader import load_config
from src.utils.metrics import (
    audit_completion_time,
    agent_success_rate,
    transactions_processed,
    flags_created
)
import time

# Import all agents
from src.agents.data_quality_agent import data_quality_agent, data_quality_task
from src.agents.reconciliation_agent import reconciliation_agent, reconciliation_task
from src.agents.anomaly_detection_agent import anomaly_agent, anomaly_task
from src.agents.context_enrichment_agent import context_agent, context_task
from src.agents.escalation_agent import escalation_agent, escalation_task
from src.agents.logging_agent import logging_agent, logging_task

logger = get_logger(__name__)


class AuditOrchestrator:
    """Master orchestrator for audit workflow"""

    def __init__(self):
        self.audit_run_id = str(uuid.uuid4())
        self.config = load_config()
        self.start_time = None

    def run_audit_cycle(self) -> Dict[str, Any]:
        """
        Execute full audit cycle

        Returns:
            Summary dictionary with results
        """
        self.start_time = time.time()
        logger.info(f"🚀 Starting audit run: {self.audit_run_id}")

        try:
            # Step 1: Check Databricks health
            if not check_databricks_health():
                raise DatabricksConnectionError("Databricks connection unhealthy")

            # Step 2: Query new transactions
            last_audit = get_last_audit_timestamp()
            logger.info(f"Last audit timestamp: {last_audit}")

            transactions = retry_with_exponential_backoff(
                query_gold_tables,
                sql_query=f"""
                    SELECT * FROM gold.recent_transactions
                    WHERE created_at > '{last_audit}'
                    LIMIT 1000
                """
            )

            logger.info(f"📊 Processing {len(transactions)} transactions")

            if transactions.empty:
                logger.info("No new transactions to audit")
                return {
                    'audit_run_id': self.audit_run_id,
                    'status': 'completed',
                    'transaction_count': 0,
                    'flags_created': 0
                }

            # Step 3: Save initial state
            save_workflow_state(self.audit_run_id, {
                'status': 'in_progress',
                'transaction_count': len(transactions),
                'started_at': datetime.now().isoformat(),
                'completed_agents': [],
                'pending_agents': ['DataQuality', 'Reconciliation', 'Anomaly', 'Context', 'Escalation', 'Logging']
            })

            # Step 4: Execute PARALLEL agents (Data Quality, Reconciliation, Anomaly)
            logger.info("🔄 Executing parallel agents...")
            parallel_results = self._run_parallel_agents(transactions)

            # Update state
            save_workflow_state(self.audit_run_id, {
                'status': 'in_progress',
                'completed_agents': ['DataQuality', 'Reconciliation', 'Anomaly'],
                'pending_agents': ['Context', 'Escalation', 'Logging'],
                'parallel_results': parallel_results
            })

            # Step 5: Identify suspicious transactions
            suspicious_txns = self._merge_suspicious_results(parallel_results, transactions)
            logger.info(f"🚨 {len(suspicious_txns)} suspicious transactions identified")

            if len(suspicious_txns) == 0:
                logger.info("No suspicious transactions found - audit complete")
                mark_audit_complete(self.audit_run_id, {
                    'transaction_count': len(transactions),
                    'flags_created': 0,
                    'duration_seconds': time.time() - self.start_time
                })
                return {
                    'audit_run_id': self.audit_run_id,
                    'status': 'completed',
                    'transaction_count': len(transactions),
                    'flags_created': 0
                }

            # Step 6: Execute SEQUENTIAL agents (Context, Escalation)
            logger.info("➡️ Executing sequential agents...")
            final_results = self._run_sequential_agents(suspicious_txns, parallel_results)

            # Step 7: Mark complete
            duration = time.time() - self.start_time
            summary = {
                'audit_run_id': self.audit_run_id,
                'status': 'completed',
                'transaction_count': len(transactions),
                'suspicious_count': len(suspicious_txns),
                'flags_created': len(final_results.get('flags', [])),
                'duration_seconds': duration
            }

            mark_audit_complete(self.audit_run_id, summary)

            # Update metrics
            audit_completion_time.observe(duration)
            transactions_processed.labels(domain='default').inc(len(transactions))
            flags_created.labels(severity='CRITICAL').inc(
                sum(1 for f in final_results.get('flags', []) if f.get('severity') == 'CRITICAL')
            )

            logger.info(f"✅ Audit complete: {self.audit_run_id} ({duration:.1f}s)")
            return summary

        except Exception as e:
            logger.error(f"Audit failed: {e}")
            save_workflow_state(self.audit_run_id, {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise AuditSystemError(f"Audit {self.audit_run_id} failed: {e}")

    def _run_parallel_agents(self, transactions) -> Dict[str, Any]:
        """Run Data Quality, Reconciliation, Anomaly agents in parallel"""

        try:
            parallel_crew = Crew(
                agents=[data_quality_agent, reconciliation_agent, anomaly_agent],
                tasks=[data_quality_task, reconciliation_task, anomaly_task],
                process=Process.parallel,
                verbose=True
            )

            # Execute parallel agents
            inputs = {'transactions': transactions.to_dict('records')}
            results = parallel_crew.kickoff(inputs=inputs)

            logger.info("Parallel agents completed successfully")
            return results

        except Exception as e:
            logger.error(f"Parallel agents failed: {e}")
            raise

    def _run_sequential_agents(self, suspicious_txns: List[dict], parallel_results: dict) -> Dict[str, Any]:
        """Run Context Enrichment and Escalation agents sequentially"""

        try:
            sequential_crew = Crew(
                agents=[context_agent, escalation_agent],
                tasks=[context_task, escalation_task],
                process=Process.sequential,
                verbose=True
            )

            inputs = {
                'suspicious_transactions': suspicious_txns,
                'audit_run_id': self.audit_run_id,
                'parallel_results': parallel_results
            }

            results = sequential_crew.kickoff(inputs=inputs)

            logger.info("Sequential agents completed successfully")
            return results

        except Exception as e:
            logger.error(f"Sequential agents failed: {e}")
            raise

    def _merge_suspicious_results(self, parallel_results: dict, all_transactions) -> List[dict]:
        """
        Merge results from parallel agents to identify suspicious transactions

        Args:
            parallel_results: Results from Data Quality, Reconciliation, Anomaly
            all_transactions: All transactions DataFrame

        Returns:
            List of suspicious transaction dicts
        """
        suspicious_ids = set()

        # Add unmatched transactions from Reconciliation
        unmatched = parallel_results.get('reconciliation', {}).get('unmatched_transactions', [])
        suspicious_ids.update([t['txn_id'] for t in unmatched])

        # Add high anomaly scores
        anomalies = parallel_results.get('anomaly', {}).get('flagged_transactions', [])
        suspicious_ids.update([a['txn_id'] for a in anomalies])

        # Add incomplete/bad quality records
        incomplete = parallel_results.get('data_quality', {}).get('incomplete_records', [])
        suspicious_ids.update(incomplete)

        # Filter transactions
        suspicious_txns = all_transactions[all_transactions['txn_id'].isin(suspicious_ids)]

        return suspicious_txns.to_dict('records')
```

---

## Part C: Main Entry Point

#### 6. `/src/main.py` (~50 lines)

```python
"""Main entry point for audit system"""

import os
from src.orchestrator.orchestrator_agent import AuditOrchestrator
from src.utils.logging import get_logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("ERGONOSIS AUDITING - Data Auditing Agent Ecosystem")
    logger.info("=" * 60)

    try:
        # Create orchestrator
        orchestrator = AuditOrchestrator()

        # Run audit cycle
        results = orchestrator.run_audit_cycle()

        # Print summary
        logger.info("=" * 60)
        logger.info("AUDIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Audit Run ID: {results['audit_run_id']}")
        logger.info(f"Status: {results['status']}")
        logger.info(f"Transactions Processed: {results['transaction_count']}")
        logger.info(f"Flags Created: {results['flags_created']}")
        logger.info(f"Duration: {results['duration_seconds']:.1f}s")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
```

---

## Part D: Feedback Analyzer Script

#### 7. `/scripts/feedback_analyzer.py` (~200 lines)

```python
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
```

---

## Testing Requirements

Create `/tests/test_orchestrator.py`:

```python
def test_orchestrator_initialization():
    from src.orchestrator.orchestrator_agent import AuditOrchestrator
    orchestrator = AuditOrchestrator()
    assert orchestrator.audit_run_id is not None

def test_full_audit_cycle():
    """Integration test - full audit cycle with mock data"""
    from src.orchestrator.orchestrator_agent import AuditOrchestrator
    orchestrator = AuditOrchestrator()

    # This will use mock data adapter
    results = orchestrator.run_audit_cycle()

    assert 'audit_run_id' in results
    assert 'status' in results
    assert results['status'] in ['completed', 'failed']

def test_retry_handler():
    from src.orchestrator.retry_handler import retry_with_exponential_backoff

    attempts = []

    def failing_func():
        attempts.append(1)
        if len(attempts) < 3:
            raise Exception("Test failure")
        return "success"

    result = retry_with_exponential_backoff(failing_func, max_retries=5, base_delay=0)
    assert result == "success"
    assert len(attempts) == 3
```

---

## Success Criteria

✅ Orchestrator initializes successfully
✅ Parallel agents execute correctly (Data Quality, Reconciliation, Anomaly)
✅ Sequential agents execute correctly (Context, Escalation)
✅ State is saved and restored properly
✅ Retry logic works with exponential backoff
✅ Main entry point runs full audit cycle
✅ Feedback analyzer updates config based on false positives
✅ All metrics are tracked (Prometheus)
✅ Integration test passes end-to-end

---

## Important Notes

- **Graceful Degradation**: If any agent fails, save state and log error (don't crash entire audit)
- **Performance**: Target <10 minutes for 1000 transactions
- **State Management**: Save state after each major step
- **Error Handling**: Comprehensive try/except with logging
- **Metrics**: Track completion time, agent success rates, flag counts

---

## Final Integration Checklist

After completing this task, verify:

1. ✅ All 6 agents are working
2. ✅ Orchestrator coordinates agents correctly
3. ✅ Parallel execution works (Data Quality, Reconciliation, Anomaly)
4. ✅ Sequential execution works (Context, Escalation)
5. ✅ State management persists to Redis
6. ✅ Retry logic handles failures
7. ✅ Main entry point executes full cycle
8. ✅ Feedback analyzer updates rules
9. ✅ All metrics are tracked
10. ✅ Integration test passes

---

## Dependencies
- `crewai`
- All previously implemented modules

---

## Estimated Effort
~550 lines of code, 3-4 hours for orchestrator + testing.

---

## Final Note

After this task, the **ENTIRE AUDIT SYSTEM** will be complete and functional! 🎉

You can then run:
```bash
python src/main.py
```

To execute a full audit cycle with mock data.
