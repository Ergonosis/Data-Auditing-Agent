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

# Import all agents (4-agent simplified pipeline for demo)
from src.agents.data_quality_agent import data_quality_agent, data_quality_task
from src.agents.reconciliation_agent import reconciliation_agent, reconciliation_task
# Temporarily disabled: anomaly_detection_agent (not in initial scope)
# Temporarily disabled: context_enrichment_agent (all tools broken in demo mode)
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

            # Step 3: Save initial state (4-agent pipeline)
            save_workflow_state(self.audit_run_id, {
                'status': 'in_progress',
                'transaction_count': len(transactions),
                'started_at': datetime.now().isoformat(),
                'completed_agents': [],
                'pending_agents': ['DataQuality', 'Reconciliation', 'Escalation', 'Logging']
            })

            # Step 4: Execute PARALLEL agents (Data Quality, Reconciliation)
            logger.info("🔄 Executing parallel agents...")
            parallel_results = self._run_parallel_agents(transactions)

            # Update state
            save_workflow_state(self.audit_run_id, {
                'status': 'in_progress',
                'completed_agents': ['DataQuality', 'Reconciliation'],
                'pending_agents': ['Escalation', 'Logging'],
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
        """Run Data Quality and Reconciliation agents (sequential in CrewAI)"""

        try:
            # Simplified 4-agent pipeline: Data Quality + Reconciliation only
            parallel_crew = Crew(
                agents=[data_quality_agent, reconciliation_agent],
                tasks=[data_quality_task, reconciliation_task],
                process=Process.sequential,
                verbose=True
            )

            # Execute parallel agents
            # Convert DataFrame to JSON-serializable format
            import json
            transactions_json = json.loads(transactions.to_json(orient='records', date_format='iso'))
            inputs = {'transactions': transactions_json}
            crew_output = parallel_crew.kickoff(inputs=inputs)

            # CrewOutput object - extract the actual result
            # The final task output is in crew_output.raw which is a string containing JSON
            logger.info("Parallel agents completed successfully")

            # Parse the crew output - it's the final agent's output as a JSON string
            if hasattr(crew_output, 'raw'):
                results = json.loads(crew_output.raw) if isinstance(crew_output.raw, str) else crew_output.raw
            elif hasattr(crew_output, 'json_dict'):
                results = crew_output.json_dict
            else:
                # Fallback - return empty results structure (simplified for 4-agent pipeline)
                results = {
                    'data_quality': {},
                    'reconciliation': {}
                }

            return results

        except Exception as e:
            logger.error(f"Parallel agents failed: {e}")
            raise

    def _run_sequential_agents(self, suspicious_txns: List[dict], parallel_results: dict) -> Dict[str, Any]:
        """Run Escalation and Logging agents sequentially"""

        try:
            # Simplified 4-agent pipeline: Escalation + Logging
            sequential_crew = Crew(
                agents=[escalation_agent, logging_agent],
                tasks=[escalation_task, logging_task],
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
            parallel_results: Results from Data Quality and Reconciliation (simplified pipeline)
            all_transactions: All transactions DataFrame

        Returns:
            List of suspicious transaction dicts
        """
        suspicious_ids = set()

        # Add unmatched transactions from Reconciliation
        unmatched = parallel_results.get('reconciliation', {}).get('unmatched_transactions', [])
        suspicious_ids.update([t['txn_id'] for t in unmatched])

        # Anomaly detection removed (not in 4-agent scope)
        # anomalies = parallel_results.get('anomaly', {}).get('flagged_transactions', [])
        # suspicious_ids.update([a['txn_id'] for a in anomalies])

        # Add incomplete/bad quality records
        incomplete = parallel_results.get('data_quality', {}).get('incomplete_records', [])
        suspicious_ids.update(incomplete)

        # Filter transactions
        suspicious_txns = all_transactions[all_transactions['txn_id'].isin(suspicious_ids)]

        return suspicious_txns.to_dict('records')
