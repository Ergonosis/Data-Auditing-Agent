"""Escalation Agent - classifies severity and creates flags"""

from crewai import Agent, Task
from src.tools.escalation_tools import (
    calculate_severity_score,
    generate_root_cause_analysis,
    batch_classify_with_llm,
    create_audit_flag,
    check_escalation_rules
)
from src.agents.llm_config import get_shared_agent_llm

escalation_agent = Agent(
    role="Escalation & Classification Specialist",
    goal="Use available tools to classify each transaction and create audit flags",
    backstory="""You are a systematic auditor who follows procedures precisely.
    You always use the provided tools in sequence and never skip steps or generate mock data.""",

    tools=[
        calculate_severity_score,
        generate_root_cause_analysis,
        batch_classify_with_llm,
        create_audit_flag,
        check_escalation_rules
    ],

    verbose=True,
    allow_delegation=False,
    llm=get_shared_agent_llm(),
    function_calling_llm=get_shared_agent_llm()  # Explicitly set function calling LLM
)

escalation_task = Task(
    description="""
    Process each suspicious transaction through the escalation workflow by calling tools.

    Input data:
    - Suspicious transactions: {suspicious_transactions}
    - Parallel agent results: {parallel_results}
    - Audit run ID: {audit_run_id}

    **MANDATORY WORKFLOW - Execute for EACH transaction in {suspicious_transactions}:**

    Step 1: Calculate severity
    - Tool: calculate_severity_score(transaction, agent_results)
    - Extract: severity_score, level, confidence, contributing_factors

    Step 2: Generate explanation
    - Tool: generate_root_cause_analysis(transaction, agent_results)
    - Extract: explanation string

    Step 3: **CRITICAL** - Create the audit flag
    - Tool: create_audit_flag(transaction_id, audit_run_id, severity, explanation, evidence)
    - Build evidence object: {matched: bool, high_amount: bool, vendor: str, contributing_factors: list}
    - This tool call is MANDATORY - it persists the flag to the database
    - Returns: flag_id (UUID)

    Step 4: Apply escalation rules
    - Tool: check_escalation_rules(severity, amount, vendor)
    - May modify severity based on business rules

    **IMPORTANT:**
    - You MUST call create_audit_flag for every transaction - no exceptions
    - Do NOT skip tool calls and generate mock output
    - The flag_id returned by create_audit_flag must be included in your final output
    - Use the ACTUAL flag_ids returned by the tools, not made-up UUIDs

    After processing all transactions, return a summary listing all flag_ids created.
    """,

    agent=escalation_agent,
    expected_output="Summary of all audit flags created, including the actual flag_id returned by create_audit_flag for each transaction"
)
