"""Audit log entry data model"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any


class AuditLogEntry(BaseModel):
    """Audit trail entry"""

    audit_run_id: str = Field(..., description="ID of audit run")
    log_sequence_number: int = Field(..., description="Sequence number within audit run")
    agent_name: str = Field(..., description="Name of agent that performed action")
    tool_called: str = Field(..., description="Name of tool that was called")
    timestamp: datetime = Field(default_factory=datetime.now, description="Action timestamp")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Sampled input data")
    output_summary: Dict[str, Any] = Field(default_factory=dict, description="Output summary")
    llm_model: Optional[str] = Field(None, description="LLM model if used")
    llm_tokens_used: Optional[int] = Field(None, description="Tokens consumed")
    llm_cost_dollars: Optional[float] = Field(None, description="Cost in USD")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_stack_trace: Optional[str] = Field(None, description="Stack trace if failed")
    decision_chain: Optional[list] = Field(None, description="Chain of decisions leading to this action")

    class Config:
        json_schema_extra = {
            "example": {
                "audit_run_id": "run_5432",
                "log_sequence_number": 1,
                "agent_name": "DataQuality",
                "tool_called": "check_completeness",
                "timestamp": "2025-02-06T10:00:05Z",
                "execution_time_ms": 250,
                "input_data": {"table": "recent_txns", "count": 1000},
                "output_summary": {"complete": 995, "score": 0.995},
                "llm_tokens_used": 0,
                "llm_cost_dollars": 0.0
            }
        }
