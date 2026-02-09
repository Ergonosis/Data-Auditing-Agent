"""Escalation and severity classification tools"""

from crewai.tools import tool
from typing import Dict, Any, List
import uuid
import os
from datetime import datetime
from src.tools.llm_client import call_llm
from src.utils.logging import get_logger
from src.utils.config_loader import load_config
from src.constants import SeverityLevel
import json

logger = get_logger(__name__)

# Global list for collecting flags in test mode
_test_mode_flags = []


def get_test_mode_flags():
    """Retrieve flags collected during test mode"""
    return _test_mode_flags.copy()


def clear_test_mode_flags():
    """Clear test mode flags (call at start of each test run)"""
    global _test_mode_flags
    _test_mode_flags = []

@tool("calculate_severity_score")
def calculate_severity_score(transaction: dict[str, Any], agent_results: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate severity score based on Data Quality and Reconciliation outputs (rule-based)

    SIMPLIFIED 4-AGENT PIPELINE SCORING:
    - Not matched in reconciliation: +50 points (increased weight - primary indicator)
    - Incomplete data quality: +30 points
    - High transaction amount (>$10k): +20 points (heuristic)

    Args:
        transaction: Transaction data
        agent_results: Results from Data Quality and Reconciliation agents {
            'data_quality': {...},
            'reconciliation': {...}
        }

    Returns:
        {
            'severity_score': int (0-100),
            'level': 'CRITICAL' | 'WARNING' | 'INFO',
            'confidence': float,
            'contributing_factors': list
        }
    """
    logger.info(f"Calculating severity for txn {transaction.get('txn_id', 'unknown')}")

    try:
        score = 0
        factors = []

        # Factor 1: Reconciliation match (highest weight - now primary indicator)
        if not agent_results.get('reconciliation', {}).get('matched', False):
            score += 50
            factors.append('no_reconciliation_match')

        # Factor 2: Data quality (incomplete records)
        if agent_results.get('data_quality', {}).get('incomplete', False):
            score += 30
            factors.append('incomplete_data')

        # Factor 3: Amount-based heuristic (no ML/anomaly detection)
        amount = transaction.get('amount', 0)
        if amount > 10000:
            score += 20
            factors.append('high_amount')

        # Removed: Anomaly detection (not in 4-agent scope)
        # Removed: Context enrichment (email approval, receipts - not in 4-agent scope)

        # Determine severity level (adjusted thresholds for simplified scoring)
        if score >= 70:
            level = SeverityLevel.CRITICAL
        elif score >= 50:
            level = SeverityLevel.WARNING
        else:
            level = SeverityLevel.INFO

        # Confidence based on number of factors (max 3 now instead of 4)
        confidence = min(len(factors) / 3, 1.0)

        result = {
            'severity_score': min(score, 100),
            'level': level.value,
            'confidence': round(confidence, 2),
            'contributing_factors': factors
        }

        logger.info(f"Severity calculated: {level.value} (score: {score}, factors: {len(factors)})")
        return result

    except Exception as e:
        logger.error(f"Severity calculation failed: {e}")
        return {
            'severity_score': 0,
            'level': SeverityLevel.INFO.value,
            'confidence': 0.0,
            'contributing_factors': []
        }


@tool("generate_root_cause_analysis")
def generate_root_cause_analysis(transaction: dict[str, Any], agent_results: dict[str, Any]) -> str:
    """
    Generate human-readable explanation for why transaction was flagged

    Uses template for simple cases, LLM for complex cases (confidence <0.7)

    Args:
        transaction: Transaction data
        agent_results: All agent results

    Returns:
        Human-readable explanation string
    """
    logger.info(f"Generating root cause analysis for {transaction['txn_id']}")

    try:
        severity_info = agent_results.get('escalation', {})
        confidence = severity_info.get('confidence', 0)

        # Simple case: use template
        if confidence >= 0.7:
            factors = severity_info.get('contributing_factors', [])

            explanations = {
                'no_reconciliation_match': f"No matching bank transaction found for ${transaction['amount']} to {transaction['vendor']}",
                'high_anomaly_score': f"Transaction amount ${transaction['amount']} is statistically unusual for this vendor",
                'no_email_approval': "No email approval or authorization found",
                'no_receipt': "Receipt not found in system"
            }

            reasons = [explanations.get(f, f) for f in factors]

            explanation = f"Flagged because: " + "; ".join(reasons) + "."

            logger.info("Used template for explanation")
            return explanation

        # Complex case: use LLM
        else:
            logger.info("Using LLM for complex explanation")

            prompt = f"""
Generate a concise 2-sentence explanation for why this transaction was flagged as suspicious:

Transaction:
- ID: {transaction['txn_id']}
- Vendor: {transaction['vendor']}
- Amount: ${transaction['amount']}
- Date: {transaction['date']}

Findings:
- Matched to bank: {agent_results.get('reconciliation', {}).get('matched', False)}
- Anomaly score: {agent_results.get('anomaly', {}).get('anomaly_score', 0)}/100
- Email approval: {agent_results.get('context', {}).get('email_approval', False)}
- Receipt found: {agent_results.get('context', {}).get('receipt_found', False)}

Respond with 2 sentences explaining the red flags:
"""

            explanation = call_llm(prompt, agent_name="Escalation")
            return explanation.strip()

    except Exception as e:
        logger.error(f"Root cause generation failed: {e}")
        return f"Transaction flagged due to multiple suspicious indicators."


@tool("batch_classify_with_llm")
def batch_classify_with_llm(transactions_json: str = "[]", agent_results_list_json: str = "[]") -> list[dict[str, str]]:
    """
    Batch classify edge-case transactions using LLM

    Args:
        transactions_json: JSON array string of transactions like '[{"txn_id": "x", "vendor": "AWS", "amount": 100}, ...]'
        agent_results_list_json: JSON array string of agent results like '[{"reconciliation": {"matched": false}, ...}, ...]'

    Returns:
        List of classifications [
            {'txn_id': 'x', 'severity': 'WARNING', 'explanation': '...'},
            ...
        ]
    """
    # Parse JSON strings to lists
    import json
    transactions = json.loads(transactions_json) if transactions_json else []
    agent_results_list = json.loads(agent_results_list_json) if agent_results_list_json else []

    logger.info(f"Batch classifying {len(transactions)} edge cases with LLM")

    try:
        # Build batch prompt
        batch_context = []
        for txn, results in zip(transactions, agent_results_list):
            batch_context.append(f"""
Transaction {txn['txn_id']}:
- Vendor: {txn['vendor']}
- Amount: ${txn['amount']}
- Matched: {results.get('reconciliation', {}).get('matched', False)}
- Anomaly: {results.get('anomaly', {}).get('anomaly_score', 0)}
- Approval: {results.get('context', {}).get('email_approval', False)}
- Receipt: {results.get('context', {}).get('receipt_found', False)}
""")

        prompt = f"""
Classify these {len(transactions)} transactions as CRITICAL, WARNING, or INFO. For each, provide a brief explanation.

{chr(10).join(batch_context)}

Respond with ONLY a JSON array (no markdown):
[
  {{"txn_id": "...", "severity": "CRITICAL|WARNING|INFO", "explanation": "..."}},
  ...
]
"""

        response = call_llm(prompt, model="anthropic/claude-3-5-haiku", agent_name="Escalation")

        # Parse JSON response
        try:
            classifications = json.loads(response.strip())
            logger.info(f"LLM classified {len(classifications)} transactions")
            return classifications
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response, using fallback")
            return [{'txn_id': t['txn_id'], 'severity': 'WARNING', 'explanation': 'Edge case'} for t in transactions]

    except Exception as e:
        logger.error(f"Batch classification failed: {e}")
        return []


@tool("create_audit_flag")
def create_audit_flag(transaction_id: str, audit_run_id: str, severity: str, explanation: str, evidence: dict[str, Any]) -> str:
    """
    Create audit flag entry (would write to database in production)

    Args:
        transaction_id: Transaction ID
        audit_run_id: Audit run ID
        severity: Severity level
        explanation: Human-readable explanation
        evidence: Supporting evidence dict

    Returns:
        Flag ID (UUID)
    """
    flag_id = str(uuid.uuid4())

    logger.info(f"Created flag {flag_id} for txn {transaction_id} (severity: {severity})")

    # In production, this would write to Flag DB
    # For now, just log it
    flag_data = {
        'flag_id': flag_id,
        'transaction_id': transaction_id,
        'audit_run_id': audit_run_id,
        'severity_level': severity,
        'explanation': explanation,
        'supporting_evidence_links': evidence,
        'created_at': datetime.now().isoformat()
    }

    logger.info(f"Flag created", **flag_data)

    # Collect flag in test mode for benchmarking
    if os.getenv('TEST_MODE') == 'true':
        global _test_mode_flags
        _test_mode_flags.append({
            'flag_id': flag_id,
            'txn_id': transaction_id,
            'severity': severity,
            'explanation': explanation
        })

    return flag_id


@tool("check_escalation_rules")
def check_escalation_rules(severity: str, amount: float, vendor: str) -> str:
    """
    Check domain-specific escalation rules and adjust severity

    Args:
        severity: Current severity level
        amount: Transaction amount
        vendor: Vendor name

    Returns:
        Updated severity level
    """
    logger.info(f"Checking escalation rules for {vendor}, ${amount}, severity {severity}")

    try:
        config = load_config()

        # Rule 1: High-value CRITICAL transactions
        if severity == SeverityLevel.CRITICAL.value and amount > 1000:
            logger.info(f"CRITICAL transaction over $1000 - escalating to CFO")
            return severity  # Keep CRITICAL

        # Rule 2: Whitelisted vendors
        whitelisted = config.get('whitelisted_vendors', [])
        if vendor in whitelisted and severity == SeverityLevel.WARNING.value:
            logger.info(f"Vendor {vendor} is whitelisted - downgrading to INFO")
            return SeverityLevel.INFO.value

        # Rule 3: Low-value INFO flags
        if severity == SeverityLevel.INFO.value and amount < 50:
            logger.info(f"Low-value INFO flag - auto-approving")
            return "AUTO_APPROVED"

        return severity

    except Exception as e:
        logger.error(f"Escalation rules check failed: {e}")
        return severity
