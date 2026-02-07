"""Constants and enums for the audit system"""

from enum import Enum


class SeverityLevel(str, Enum):
    """Flag severity levels"""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class TransactionSource(str, Enum):
    """Transaction data sources"""
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank"
    EMAIL = "email"
    RECEIPT = "receipt"
    VENDOR_INVOICE = "vendor_invoice"


class AuditStatus(str, Enum):
    """Audit workflow status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class HumanDecision(str, Enum):
    """Human review decisions"""
    APPROVED = "approved"
    REJECTED = "rejected"
    FALSE_POSITIVE = "false_positive"
    NEEDS_MORE_INFO = "needs_more_info"


# Default configuration values
DEFAULT_COMPLETENESS_THRESHOLD = 0.90
DEFAULT_AMOUNT_MATCH_THRESHOLD = 0.05
DEFAULT_DATE_WINDOW_DAYS = 3
DEFAULT_ANOMALY_SIGMA = 2.5
DEFAULT_CRITICAL_SCORE = 80
DEFAULT_WARNING_SCORE = 50
DEFAULT_INFO_SCORE = 20

# Audit SLA
AUDIT_TIMEOUT_SECONDS = 1800  # 30 minutes
AGENT_TIMEOUT_SECONDS = 300    # 5 minutes per agent

# Cost limits
DAILY_LLM_COST_LIMIT = 10.0  # dollars
