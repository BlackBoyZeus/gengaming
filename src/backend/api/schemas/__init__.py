"""
Centralized exports of Pydantic schema models for the GameGen-X API.
Provides comprehensive type validation and security controls for all API data.

Pydantic version: ^1.10.0
"""

__version__ = "1.0.0"

# Import control schemas
from api.schemas.control import (
    ControlBase,
    ControlCreate, 
    ControlResponse
)

# Import generation schemas
from api.schemas.generation import (
    GenerationParameters,
    GenerationBase,
    GenerationCreate,
    GenerationResponse
)

# Import status schemas
from api.schemas.status import (
    SystemStatus,
    ResourceMetrics,
    PerformanceMetrics,
    StatusResponse
)

# Export all schema models
__all__ = [
    # Control schemas
    "ControlBase",
    "ControlCreate",
    "ControlResponse",
    
    # Generation schemas
    "GenerationParameters",
    "GenerationBase", 
    "GenerationCreate",
    "GenerationResponse",
    
    # Status schemas
    "SystemStatus",
    "ResourceMetrics",
    "PerformanceMetrics",
    "StatusResponse"
]