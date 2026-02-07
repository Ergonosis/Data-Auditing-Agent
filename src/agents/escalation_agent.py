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
