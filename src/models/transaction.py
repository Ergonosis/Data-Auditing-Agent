"""Transaction data model"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any


class Transaction(BaseModel):
    """Transaction entity"""

    txn_id: str = Field(..., description="Unique transaction ID")
    source: str = Field(..., description="Transaction source (credit_card, bank, email, receipt)")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    vendor: str = Field(..., description="Vendor name (raw)")
    vendor_id: Optional[str] = Field(None, description="Canonical vendor ID from KG")
    date: datetime = Field(..., description="Transaction date")
    domain: Optional[str] = Field(None, description="Business domain (inferred or manual)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "txn_id": "cc_12345",
                "source": "credit_card",
                "amount": 500.00,
                "vendor": "AWS",
                "vendor_id": "aws_infrastructure",
                "date": "2025-02-03T10:00:00Z",
                "domain": "business_operations",
                "metadata": {"card_last4": "1234", "merchant_category": "cloud_services"}
            }
        }
