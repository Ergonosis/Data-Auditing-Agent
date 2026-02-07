"""SQL schemas for audit database tables."""

# Flags table - stores audit flags for human review
FLAGS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS flags (
    flag_id VARCHAR(36) PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    audit_run_id VARCHAR(36) NOT NULL,
    severity_level VARCHAR(20) CHECK (severity_level IN ('CRITICAL', 'WARNING', 'INFO')),
    confidence_score FLOAT CHECK (confidence_score BETWEEN 0 AND 1),
    explanation TEXT NOT NULL,
    supporting_evidence_links TEXT,  -- JSON stored as text
    reviewed BOOLEAN DEFAULT FALSE,
    human_decision VARCHAR(30),
    reviewer_id VARCHAR(255),
    review_timestamp TIMESTAMP,
    reviewer_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_flags_txn_id ON flags(transaction_id);
CREATE INDEX IF NOT EXISTS idx_flags_audit_run ON flags(audit_run_id);
CREATE INDEX IF NOT EXISTS idx_flags_reviewed ON flags(reviewed, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_flags_severity ON flags(severity_level, created_at DESC);
"""

# Audit trail table - immutable append-only log
AUDIT_TRAIL_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_run_id VARCHAR(36) NOT NULL,
    log_sequence_number BIGINT NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    tool_called VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    execution_time_ms BIGINT,
    input_data TEXT,  -- JSON stored as text
    output_summary TEXT,  -- JSON stored as text
    llm_model VARCHAR(100),
    llm_tokens_used BIGINT,
    llm_cost_dollars DECIMAL(10, 4),
    error_message TEXT,
    error_stack_trace TEXT,
    decision_chain TEXT,  -- JSON array stored as text
    PRIMARY KEY (audit_run_id, log_sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_audit_trail_agent ON audit_trail(agent_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_trail_timestamp ON audit_trail(timestamp DESC);
"""

# Workflow state table - tracks audit run state
WORKFLOW_STATE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflow_state (
    audit_run_id VARCHAR(36) PRIMARY KEY,
    workflow_status VARCHAR(20) CHECK (workflow_status IN ('pending', 'in_progress', 'completed', 'failed', 'paused')),
    current_agent VARCHAR(100),
    completed_agents TEXT,  -- JSON array
    pending_agents TEXT,  -- JSON array
    intermediate_results TEXT,  -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_workflow_status ON workflow_state(workflow_status, created_at DESC);
"""


def create_all_tables(cursor):
    """
    Execute all CREATE TABLE statements.

    Args:
        cursor: Database cursor object
    """
    cursor.execute(FLAGS_TABLE_SCHEMA)
    cursor.execute(AUDIT_TRAIL_TABLE_SCHEMA)
    cursor.execute(WORKFLOW_STATE_TABLE_SCHEMA)
