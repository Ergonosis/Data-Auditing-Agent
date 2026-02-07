"""Vendor profile data model"""

from pydantic import BaseModel, Field
from typing import Optional


class VendorProfile(BaseModel):
    """Statistical profile of vendor spending patterns"""

    vendor_id: str = Field(..., description="Canonical vendor ID")
    vendor_name: str = Field(..., description="Canonical vendor name")
    mean_amount: float = Field(..., description="Average transaction amount")
    std_dev: float = Field(..., description="Standard deviation of amount")
    frequency: int = Field(..., description="Number of transactions per month")
    typical_day_of_month: Optional[int] = Field(None, description="Typical payment day (for recurring)")
    last_transaction_date: Optional[str] = Field(None, description="Date of last transaction")

    class Config:
        json_schema_extra = {
            "example": {
                "vendor_id": "amazon_marketplace",
                "vendor_name": "Amazon Marketplace",
                "mean_amount": 500.00,
                "std_dev": 200.00,
                "frequency": 12,
                "typical_day_of_month": 15,
                "last_transaction_date": "2025-02-01"
            }
        }
