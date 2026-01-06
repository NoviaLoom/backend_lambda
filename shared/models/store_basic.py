"""
Basic store information.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class StoreBasic(BaseModel):
    """Basic store information."""

    store_id: str = Field(description="Unique store identifier")
    name: str = Field(description="Store name")
    city: Optional[str] = Field(default=None, description="City")
    country: Optional[str] = Field(default=None, description="Country")
    network_id: str = Field(description="Network identifier")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "store_id": "store-123",
                "name": "Decathlon Paris Wagram",
                "city": "Paris",
                "country": "France",
                "network_id": "decathlon-france"
            }
        }
    )

