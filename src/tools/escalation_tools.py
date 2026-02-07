"""Escalation and severity classification tools"""

from crewai_tools import tool
from typing import Dict, Any, List
import uuid
from datetime import datetime
from src.tools.llm_client import call_llm
from src.utils.logging import get_logger
from src.utils.config_loader import load_config
from src.constants import SeverityLevel
import json

logger = get_logger(__name__)

@tool("calculate_severity_score")
def calculate_severity_score(transaction: dict, agent_results: dict) -> dict:
    """
    Calculate severity score based on all agent outputs (rule-based)

    Scoring rules:
    - Not matched: +40 points
    - Anomaly score >70: +30 points
    - No email approval: +20 points
    - No receipt found: +10 points

    Args:
        transaction: Transaction data
        agent_results: Results from all previous agents {
            'data_quality': {...},
            'reconciliation': {...},
            'anomaly': {...},
            'context': {...}
        }

    Returns:
        {
            'severity_score': int (0-100),
            'level': 'CRITICAL' | 'WARNING' | 'INFO',
            'confidence': float,
            'contributing_factors': list
        }
    """
    logger.info(f"Calculating severity for txn {transaction['txn_id']}")

    try:
        score = 0
        factors = []

        # Factor 1: Reconciliation match
        if not agent_results.get('reconciliation', {}).get('matched', False):
            score += 40
            factors.append('no_reconciliation_match')

        # Factor 2: Anomaly score
        anomaly_score = agent_results.get('anomaly', {}).get('anomaly_score', 0)
        if anomaly_score > 70:
            score += 30
            factors.append('high_anomaly_score')

        # Factor 3: Email approval
        if not agent_results.get('context', {}).get('email_approval', False):
            score += 20
            factors.append('no_email_approval')

        # Factor 4: Receipt
        if not agent_results.get('context', {}).get('receipt_found', False):
            score += 10
            factors.append('no_receipt')

        # Determine severity level
        if score >= 80:
            level = SeverityLevel.CRITICAL
        elif score >= 50:
            level = SeverityLevel.WARNING
        else:
            level = SeverityLevel.INFO

        # Confidence based on number of factors
        confidence = min(len(factors) / 4, 1.0)

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
def generate_root_cause_analysis(transaction: dict, agent_results: dict) -> str:
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
def batch_classify_with_llm(transactions: list, agent_results_list: list) -> list:
    """
    Batch classify edge-case transactions using LLM

    Args:
        transactions: List of transactions
        agent_results_list: List of agent results (parallel to transactions)

    Returns:
        List of classifications [
            {'txn_id': 'x', 'severity': 'WARNING', 'explanation': '...'},
            ...
        ]
    """
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
def create_audit_flag(transaction_id: str, audit_run_id: str, severity: str, explanation: str, evidence: dict) -> str:
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
