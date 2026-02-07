# TASK 4: Context Enrichment & Escalation Agents

## Objective
Implement the Context Enrichment Agent (5 tools, runs SEQUENTIALLY after parallel agents) and Escalation Agent (5 tools, final classification step).

## Context
These are **sequential agents** that process only the suspicious transactions (1-3% of total) identified by the parallel agents. Context Enrichment searches for supporting documentation, while Escalation classifies severity and generates explanations.

**Critical**: Context uses LLM sparingly (<5% of suspicious transactions), Escalation uses LLM for edge cases only (~10%).

---

## Part A: Context Enrichment Agent

### Purpose
Search emails, calendar, receipts for supporting documentation. Uses batch processing and semantic search (LLM-powered) only for high-priority unmatched transactions.

### Architecture
```
Context Enrichment Agent (runs on ~82 suspicious txns)
├─ Tool 1: search_emails_batch (SQL query, NO LLM)
├─ Tool 2: search_calendar_events (SQL query, NO LLM)
├─ Tool 3: extract_approval_chains (Keyword parsing, NO LLM)
├─ Tool 4: find_receipt_images (File search, NO LLM)
└─ Tool 5: semantic_search_documents (LLM embeddings, <5% usage)
```

---

### Files to Create

#### 1. `/src/tools/context_tools.py` (~300 lines)

```python
"""Context enrichment tools for finding supporting documentation"""

from crewai_tools import tool
import pandas as pd
from typing import Dict, Any, List
from datetime import timedelta
from src.tools.databricks_client import query_gold_tables
from src.tools.llm_client import call_llm
from src.utils.logging import get_logger
import re

logger = get_logger(__name__)

@tool("search_emails_batch")
def search_emails_batch(transactions: list) -> dict:
    """
    Batch search emails for mentions of vendors/amounts/dates

    Args:
        transactions: List of suspicious transactions [
            {'txn_id': 'x', 'vendor': 'AWS', 'amount': 500, 'date': '2025-02-01'},
            ...
        ]

    Returns:
        {
            'txn_x': {
                'email_matches': [
                    {'email_id': 'e1', 'subject': 'AWS Invoice', 'confidence': 0.9},
                    ...
                ]
            },
            ...
        }
    """
    logger.info(f"Searching emails for {len(transactions)} transactions")

    try:
        results = {}

        for txn in transactions:
            vendor = txn['vendor']
            amount = txn['amount']
            txn_date = pd.to_datetime(txn['date'])

            # Query emails ±3 days from transaction
            start_date = txn_date - timedelta(days=3)
            end_date = txn_date + timedelta(days=3)

            emails = query_gold_tables(f"""
                SELECT email_id, subject, sender, email_date
                FROM gold.emails
                WHERE email_date BETWEEN '{start_date}' AND '{end_date}'
                  AND (subject LIKE '%{vendor}%' OR body LIKE '%{vendor}%')
                LIMIT 5
            """)

            email_matches = []
            for _, email in emails.iterrows():
                # Calculate confidence based on match quality
                confidence = 0.9 if vendor.lower() in email['subject'].lower() else 0.7

                email_matches.append({
                    'email_id': email['email_id'],
                    'subject': email['subject'],
                    'sender': email['sender'],
                    'confidence': confidence
                })

            results[txn['txn_id']] = {
                'email_matches': email_matches,
                'match_count': len(email_matches)
            }

        logger.info(f"Email search complete: found matches for {sum(1 for r in results.values() if r['match_count'] > 0)} transactions")
        return results

    except Exception as e:
        logger.error(f"Email search failed: {e}")
        return {}


@tool("search_calendar_events")
def search_calendar_events(transaction_date: str, vendor: str) -> list:
    """
    Search calendar for events matching transaction date

    Args:
        transaction_date: Transaction date (ISO format)
        vendor: Vendor name

    Returns:
        List of matching events [
            {'event_id': 'cal_123', 'title': 'Client dinner', 'date': '2025-02-01'},
            ...
        ]
    """
    logger.info(f"Searching calendar for {vendor} on {transaction_date}")

    try:
        txn_date = pd.to_datetime(transaction_date)
        start_date = txn_date - timedelta(days=3)
        end_date = txn_date + timedelta(days=3)

        events = query_gold_tables(f"""
            SELECT event_id, title, event_date, description
            FROM gold.calendar_events
            WHERE event_date BETWEEN '{start_date}' AND '{end_date}'
              AND (title LIKE '%{vendor}%' OR description LIKE '%{vendor}%')
            LIMIT 5
        """)

        if events.empty:
            logger.info(f"No calendar events found for {vendor}")
            return []

        return events[['event_id', 'title', 'event_date']].to_dict('records')

    except Exception as e:
        logger.error(f"Calendar search failed: {e}")
        return []


@tool("extract_approval_chains")
def extract_approval_chains(email_thread_id: str) -> dict:
    """
    Extract approval information from email thread

    Args:
        email_thread_id: Email thread ID

    Returns:
        {
            'approved': bool,
            'approver': str,
            'timestamp': str,
            'approval_keywords': list
        }
    """
    logger.info(f"Extracting approval chain from thread {email_thread_id}")

    try:
        # Query email thread
        emails = query_gold_tables(f"""
            SELECT email_id, sender, body, email_date
            FROM gold.emails
            WHERE thread_id = '{email_thread_id}'
            ORDER BY email_date ASC
        """)

        if emails.empty:
            return {
                'approved': False,
                'approver': None,
                'timestamp': None,
                'approval_keywords': []
            }

        # Approval keywords
        approval_keywords = [
            'approved', 'authorize', 'authorized', 'go ahead', 'proceed',
            'looks good', 'lgtm', 'approved for payment', 'please pay'
        ]

        # Search for approval keywords in email bodies
        for _, email in emails.iterrows():
            body_lower = email['body'].lower()

            for keyword in approval_keywords:
                if keyword in body_lower:
                    return {
                        'approved': True,
                        'approver': email['sender'],
                        'timestamp': str(email['email_date']),
                        'approval_keywords': [keyword]
                    }

        # No approval found
        return {
            'approved': False,
            'approver': None,
            'timestamp': None,
            'approval_keywords': []
        }

    except Exception as e:
        logger.error(f"Approval extraction failed: {e}")
        return {
            'approved': False,
            'approver': None,
            'timestamp': None,
            'approval_keywords': []
        }


@tool("find_receipt_images")
def find_receipt_images(vendor: str, amount: float, date_range: tuple) -> list:
    """
    Find receipt images matching vendor/amount/date

    Args:
        vendor: Vendor name
        amount: Transaction amount
        date_range: (start_date, end_date)

    Returns:
        List of receipt file paths ['s3://bucket/receipt1.jpg', ...]
    """
    logger.info(f"Finding receipts for {vendor}, ${amount}")

    try:
        start_date, end_date = date_range

        receipts = query_gold_tables(f"""
            SELECT receipt_id, file_path, ocr_vendor, ocr_amount, ocr_date
            FROM gold.receipts_ocr
            WHERE ocr_date BETWEEN '{start_date}' AND '{end_date}'
              AND ocr_vendor LIKE '%{vendor}%'
              AND ocr_amount BETWEEN {amount * 0.95} AND {amount * 1.05}
            LIMIT 5
        """)

        if receipts.empty:
            logger.info(f"No receipts found for {vendor}")
            return []

        return receipts['file_path'].tolist()

    except Exception as e:
        logger.error(f"Receipt search failed: {e}")
        return []


@tool("semantic_search_documents")
def semantic_search_documents(query: str, top_k: int = 5) -> list:
    """
    Semantic search across documents using LLM embeddings (EXPENSIVE - use sparingly!)

    Args:
        query: Search query (e.g., "AWS infrastructure approval")
        top_k: Number of results to return

    Returns:
        List of relevant documents [
            {'doc_id': 'd1', 'title': '...', 'snippet': '...', 'relevance': 0.85},
            ...
        ]
    """
    logger.info(f"Semantic search for: {query}")

    try:
        # Use LLM to reformulate query and search
        prompt = f"""
You are searching a document database for information about: "{query}"

Given this context, generate 3 specific search keywords (comma-separated) that would find relevant documents.

Example:
Query: "AWS infrastructure approval"
Keywords: AWS invoice, infrastructure purchase, cloud services approval

Respond with ONLY the keywords, no explanation:
"""

        keywords_response = call_llm(prompt, agent_name="ContextEnrichment")
        keywords = [k.strip() for k in keywords_response.split(',')]

        logger.info(f"Expanded query to keywords: {keywords}")

        # Search documents using keywords (SQL)
        results = []
        for keyword in keywords[:3]:  # Limit to 3 keywords
            docs = query_gold_tables(f"""
                SELECT doc_id, title, snippet
                FROM gold.documents
                WHERE title LIKE '%{keyword}%' OR content LIKE '%{keyword}%'
                LIMIT 2
            """)

            for _, doc in docs.iterrows():
                results.append({
                    'doc_id': doc['doc_id'],
                    'title': doc['title'],
                    'snippet': doc['snippet'][:200],
                    'relevance': 0.8  # Simplified relevance score
                })

        return results[:top_k]

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []
```

#### 2. `/src/agents/context_enrichment_agent.py`

```python
"""Context Enrichment Agent - finds supporting documentation"""

from crewai import Agent, Task
from src.tools.context_tools import (
    search_emails_batch,
    search_calendar_events,
    extract_approval_chains,
    find_receipt_images,
    semantic_search_documents
)

context_agent = Agent(
    role="Context Enrichment Specialist",
    goal="Find supporting documentation (emails, receipts, approvals) for suspicious transactions",
    backstory="""You are a forensic investigator specializing in document analysis.
    You can find evidence across email archives, calendar systems, and receipt databases.""",

    tools=[
        search_emails_batch,
        search_calendar_events,
        extract_approval_chains,
        find_receipt_images,
        semantic_search_documents
    ],

    verbose=True,
    allow_delegation=False,
    llm=None
)

context_task = Task(
    description="""
    For suspicious transactions, find supporting documentation:
    1. Batch search emails for vendor/amount mentions
    2. Search calendar for related events
    3. Extract approval chains from email threads
    4. Find receipt images matching transaction
    5. Use semantic search ONLY for high-priority unmatched transactions (<5%)

    **Input**: List of suspicious transactions
    **Output**: {
        'enriched_transactions': [
            {
                'txn_id': 'x',
                'email_approval': True,
                'calendar_event': {...},
                'receipt_found': True,
                'confidence': 0.95
            },
            ...
        ]
    }
    """,

    agent=context_agent,
    expected_output="JSON with enriched transaction data"
)
```

---

## Part B: Escalation & Classification Agent

### Purpose
Classify transaction severity (CRITICAL/WARNING/INFO) and generate human-readable explanations. Uses rule-based scoring with LLM for edge cases only.

### Architecture
```
Escalation Agent (final step)
├─ Tool 1: calculate_severity_score (Rule-based, NO LLM)
├─ Tool 2: generate_root_cause_analysis (Template or LLM)
├─ Tool 3: batch_classify_with_llm (LLM, only for edge cases ~10%)
├─ Tool 4: create_audit_flag (Database write)
└─ Tool 5: check_escalation_rules (Domain-specific rules)
```

---

### Files to Create

#### 3. `/src/tools/escalation_tools.py` (~300 lines)

```python
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
```

#### 4. `/src/agents/escalation_agent.py`

```python
"""Escalation Agent - classifies severity and creates flags"""

from crewai import Agent, Task
from src.tools.escalation_tools import (
    calculate_severity_score,
    generate_root_cause_analysis,
    batch_classify_with_llm,
    create_audit_flag,
    check_escalation_rules
)

escalation_agent = Agent(
    role="Escalation & Classification Specialist",
    goal="Classify transaction severity and generate clear explanations for finance team",
    backstory="""You are a senior auditor with expertise in risk assessment and communication.
    You translate technical findings into clear, actionable insights for non-technical stakeholders.""",

    tools=[
        calculate_severity_score,
        generate_root_cause_analysis,
        batch_classify_with_llm,
        create_audit_flag,
        check_escalation_rules
    ],

    verbose=True,
    allow_delegation=False,
    llm=None
)

escalation_task = Task(
    description="""
    Classify suspicious transactions and create audit flags:
    1. Calculate severity score (rule-based)
    2. Generate root cause explanation (template or LLM)
    3. For edge cases (confidence <0.7), use batch_classify_with_llm
    4. Create audit flags in database
    5. Check escalation rules (e.g., whitelist adjustments)

    **Output**: {
        'flags': [
            {
                'flag_id': 'uuid',
                'txn_id': 'x',
                'severity': 'CRITICAL',
                'confidence': 0.92,
                'explanation': '...',
                'evidence': {...}
            },
            ...
        ]
    }
    """,

    agent=escalation_agent,
    expected_output="JSON with created audit flags"
)
```

---

## Testing Requirements

Create `/tests/test_agents/test_context_escalation.py`:

```python
def test_email_search():
    from src.tools.context_tools import search_emails_batch
    txns = [{'txn_id': 'cc_1', 'vendor': 'AWS', 'amount': 500, 'date': '2025-02-01'}]
    result = search_emails_batch.func(txns)
    assert isinstance(result, dict)

def test_severity_calculation():
    from src.tools.escalation_tools import calculate_severity_score
    txn = {'txn_id': 'cc_1', 'vendor': 'Unknown', 'amount': 1000}
    results = {
        'reconciliation': {'matched': False},
        'anomaly': {'anomaly_score': 85},
        'context': {'email_approval': False, 'receipt_found': False}
    }
    result = calculate_severity_score.func(txn, results)
    assert result['level'] == 'CRITICAL'

def test_flag_creation():
    from src.tools.escalation_tools import create_audit_flag
    flag_id = create_audit_flag.func('cc_1', 'run_123', 'CRITICAL', 'Test', {})
    assert isinstance(flag_id, str)
```

---

## Success Criteria

✅ Context agent searches emails/calendar/receipts
✅ Semantic search uses LLM sparingly (<5%)
✅ Severity calculator applies rules correctly
✅ Root cause uses templates first, LLM for complex cases
✅ Batch LLM classification works for edge cases
✅ Escalation rules adjust severity appropriately
✅ Flags are created with full context
✅ All tests pass

---

## Important Notes

- **Sequential Execution**: Context runs AFTER parallel agents, Escalation runs AFTER Context
- **LLM Budget**: Context uses <5% LLM, Escalation uses ~10% (batch only)
- **Batch Processing**: Always batch LLM calls (max 100 txns/call)
- **Error Handling**: Graceful fallbacks for all LLM failures

---

## Dependencies
- `crewai`, `crewai-tools`
- Existing modules: `src.tools.databricks_client`, `src.tools.llm_client`, `src.utils.*`

---

## Estimated Effort
~600 lines of code, 2-3 hours for both agents.
