"""
Pydantic schema models for validating and serializing video control data in the GameGen-X API.
Implements strict validation, security controls and performance optimizations for real-time
control signals with <50ms response time target.

Pydantic version: ^1.10.0
"""

from datetime import datetime
from typing import Dict, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator, root_validator, Json

from db.models.control import ControlType

# Constants for validation
ALLOWED_KEYS = {'W', 'A', 'S', 'D', 'SPACE', 'SHIFT', 'CTRL', 'ALT'}
ALLOWED_ACTIONS = {'press', 'release', 'hold'}
ALLOWED_ENV_PARAMS = {'weather', 'lighting', 'effects'}
ALLOWED_WEATHER = {'clear', 'rain', 'snow', 'fog'}
ALLOWED_LIGHTING = {'day', 'night', 'dawn', 'dusk'}
ALLOWED_EFFECTS = {'none', 'blur', 'glow', 'distortion'}
MAX_INSTRUCTION_LENGTH = 1000

class ControlBase(BaseModel):
    """Base Pydantic model for control data validation with enhanced security and performance."""
    
    type: str = Field(
        ...,
        description="Control type (keyboard, environment, instruction)",
        example="keyboard"
    )
    data: Json = Field(
        ...,
        description="Type-specific control data",
        example={"key": "W", "action": "press", "timestamp": 1234.56}
    )
    schema_version: str = Field(
        default="1.0",
        description="Schema version for validation",
        regex="^[0-9]+\\.[0-9]+$"
    )

    @validator("type")
    def validate_type(cls, value: str) -> str:
        """Validate control type with strict enum checking."""
        value = value.lower().strip()
        if value not in {t.name for t in ControlType}:
            raise ValueError(f"Invalid control type: {value}")
        return value

    @validator("data")
    def validate_data(cls, value: Json, values: Dict[str, Any]) -> Json:
        """Type-specific data validation with security checks."""
        if not isinstance(value, dict):
            raise ValueError("Data must be a JSON object")

        control_type = values.get("type")
        if control_type == "keyboard":
            if not all(k in value for k in ("key", "action", "timestamp")):
                raise ValueError("Missing required keyboard control fields")
            if value["key"].upper() not in ALLOWED_KEYS:
                raise ValueError(f"Invalid key: {value['key']}")
            if value["action"] not in ALLOWED_ACTIONS:
                raise ValueError(f"Invalid action: {value['action']}")

        elif control_type == "environment":
            if not all(k in value for k in ("parameter", "value", "timestamp")):
                raise ValueError("Missing required environment control fields")
            if value["parameter"] not in ALLOWED_ENV_PARAMS:
                raise ValueError(f"Invalid parameter: {value['parameter']}")
            
            param = value["parameter"]
            allowed_values = {
                "weather": ALLOWED_WEATHER,
                "lighting": ALLOWED_LIGHTING,
                "effects": ALLOWED_EFFECTS
            }
            if value["value"] not in allowed_values[param]:
                raise ValueError(f"Invalid value for {param}: {value['value']}")

        elif control_type == "instruction":
            if not all(k in value for k in ("instruction", "parameters", "timestamp")):
                raise ValueError("Missing required instruction control fields")
            if len(value["instruction"]) > MAX_INSTRUCTION_LENGTH:
                raise ValueError("Instruction too long")
            if not isinstance(value["parameters"], dict):
                raise ValueError("Parameters must be a JSON object")

        # Validate timestamp
        if not isinstance(value["timestamp"], (int, float)):
            raise ValueError("Timestamp must be numeric")
        if value["timestamp"] < 0:
            raise ValueError("Timestamp cannot be negative")

        return value

    @root_validator
    def validate_all(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-field validation and security checks."""
        if "type" not in values or "data" not in values:
            raise ValueError("Missing required fields")

        # Verify type-data consistency
        control_type = values["type"]
        data = values["data"]
        
        # Rate limiting check
        if "timestamp" in data:
            current_time = datetime.now().timestamp()
            if abs(current_time - data["timestamp"]) > 5.0:
                raise ValueError("Control signal timestamp too old")

        return values

    class Config:
        """Pydantic configuration for performance optimization."""
        validate_assignment = True
        validate_all = True
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class ControlCreate(ControlBase):
    """Schema for creating new control records with enhanced validation."""
    
    generation_id: UUID = Field(
        ...,
        description="Associated generation ID",
        example="123e4567-e89b-12d3-a456-426614174000"
    )
    video_id: UUID = Field(
        ...,
        description="Associated video ID",
        example="123e4567-e89b-12d3-a456-426614174001"
    )
    timestamp: float = Field(
        ...,
        description="Control signal timestamp",
        example=1234.56,
        ge=0
    )

class ControlResponse(ControlCreate):
    """Schema for control record responses with performance optimizations."""
    
    id: UUID = Field(
        ...,
        description="Control record ID",
        example="123e4567-e89b-12d3-a456-426614174002"
    )
    created_at: datetime = Field(
        ...,
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        ...,
        description="Record last update timestamp"
    )
    status: str = Field(
        default="active",
        description="Control record status",
        regex="^(active|inactive|failed)$"
    )
    metadata: Optional[Json] = Field(
        default=None,
        description="Additional metadata",
        example={"latency_ms": 45, "processed": True}
    )

    class Config:
        """Response schema configuration."""
        orm_mode = True
        validate_assignment = True
        json_encoders = {
            **ControlBase.Config.json_encoders,
            datetime: lambda v: v.isoformat()
        }