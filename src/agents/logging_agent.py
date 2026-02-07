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
    allow_delegation=False
)

logging_task = Task(
    description="Continuously log all agent decisions throughout audit cycle",
    agent=logging_agent,
    expected_output="Confirmation of logging completion"
)
