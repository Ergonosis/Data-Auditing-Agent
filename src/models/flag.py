"""Flag data model"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any
from src.constants import SeverityLevel, HumanDecision


class Flag(BaseModel):
    """Audit flag entity"""

    flag_id: str = Field(..., description="Unique flag ID (UUID)")
    transaction_id: str = Field(..., description="ID of flagged transaction")
    audit_run_id: str = Field(..., description="ID of audit run that created flag")
    severity_level: SeverityLevel = Field(..., description="Flag severity")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in flag (0-1)")
    explanation: str = Field(..., description="Human-readable explanation")
    supporting_evidence_links: Dict[str, Any] = Field(
        default_factory=dict,
        description="Links to supporting evidence (emails, receipts, etc.)"
    )
    reviewed: bool = Field(default=False, description="Whether human has reviewed")
    human_decision: Optional[HumanDecision] = Field(None, description="Human review decision")
    reviewer_id: Optional[str] = Field(None, description="ID of reviewer")
    review_timestamp: Optional[datetime] = Field(None, description="When reviewed")
    reviewer_notes: Optional[str] = Field(None, description="Reviewer comments")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "flag_id": "f12345-6789-abcd",
                "transaction_id": "cc_12345",
                "audit_run_id": "run_5432",
                "severity_level": "CRITICAL",
                "confidence_score": 0.92,
                "explanation": "Unauthorized charge: No matching bank transaction, amount $847 is 3.2σ above vendor average",
                "supporting_evidence_links": {
                    "reconciliation_match": False,
                    "anomaly_score": 0.85,
                    "email_approval": False,
                    "receipt_found": False
                },
                "reviewed": False
            }
        }
