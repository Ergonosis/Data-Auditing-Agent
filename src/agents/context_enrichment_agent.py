"""Context Enrichment Agent - finds supporting documentation"""

from crewai import Agent, Task
from src.tools.context_tools import (
    search_emails_batch,
    search_calendar_events,
    extract_approval_chains,
    find_receipt_images,
    semantic_search_documents
)
from src.agents.llm_config import get_shared_agent_llm

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
    llm=get_shared_agent_llm()
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
