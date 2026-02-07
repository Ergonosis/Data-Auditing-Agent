"""Reconciliation Agent - matches transactions across sources"""

from crewai import Agent, Task
from src.tools.reconciliation_tools import (
    cross_source_matcher,
    entity_resolver_kg,
    fuzzy_vendor_matcher,
    receipt_transaction_matcher,
    find_orphan_transactions
)

reconciliation_agent = Agent(
    role="Transaction Reconciliation Specialist",
    goal="Match transactions across credit card, bank, email, and receipt sources with 95%+ accuracy",
    backstory="""You are an expert financial auditor with 15 years of experience in reconciliation.
    You have a keen eye for spotting discrepancies and can match transactions even when vendor names vary.""",

    tools=[
        cross_source_matcher,
        entity_resolver_kg,
        fuzzy_vendor_matcher,
        receipt_transaction_matcher,
        find_orphan_transactions
    ],

    verbose=True,
    allow_delegation=False,
    llm=None  # Pure tool-based, no LLM reasoning
)

reconciliation_task = Task(
    description="""
    Match transactions across multiple sources:
    1. Use cross_source_matcher to match credit_card vs bank transactions
    2. Use entity_resolver_kg to resolve vendor name variations
    3. Use fuzzy_vendor_matcher for typo matching
    4. Use receipt_transaction_matcher for receipt-to-transaction matching
    5. Use find_orphan_transactions to identify single-source transactions (SUSPICIOUS)

    **Output**: {
        'matched_pairs': [...],
        'unmatched_transactions': [...],  # SUSPICIOUS
        'low_confidence_matches': [...],  # confidence <0.7
        'orphan_transactions': [...]  # appear in only one source
    }
    """,

    agent=reconciliation_agent,
    expected_output="JSON with matched pairs and unmatched (suspicious) transactions"
)
