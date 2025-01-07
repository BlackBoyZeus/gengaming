"""
Pydantic schema models for video generation request validation and response formatting.
Handles text prompts, generation parameters, quality metrics validation, and secure response formatting.

Pydantic version: ^1.10.0
"""

from datetime import datetime
from typing import Dict, Optional
from uuid import UUID
from pydantic import BaseModel, Field, constr, validator
from db.models.generation import GenerationStatus

class GenerationParameters(BaseModel):
    """
    Pydantic model for video generation parameters validation with enhanced resolution 
    and frame validation based on system requirements.
    """
    resolution_width: int = Field(
        ..., 
        description="Video width in pixels",
        example=1280
    )
    resolution_height: int = Field(
        ..., 
        description="Video height in pixels",
        example=720
    )
    frame_count: int = Field(
        ..., 
        description="Number of frames to generate",
        example=102
    )
    perspective: str = Field(
        ...,
        description="Camera perspective for generation",
        example="third_person"
    )

    @validator('resolution_width', 'resolution_height')
    def validate_resolution(cls, value: int, field: Field) -> int:
        """Validates resolution parameters against supported values."""
        valid_widths = [320, 848, 1280]
        valid_heights = [256, 480, 720]
        
        if field.name == 'resolution_width' and value not in valid_widths:
            raise ValueError(f"Width must be one of {valid_widths}")
        if field.name == 'resolution_height' and value not in valid_heights:
            raise ValueError(f"Height must be one of {valid_heights}")
            
        # Validate 720p requirement from technical specs
        if field.name == 'resolution_width' and value == 1280:
            if not hasattr(cls, 'resolution_height') or cls.resolution_height != 720:
                raise ValueError("720p resolution requires 1280x720")
                
        return value

    @validator('frame_count')
    def validate_frame_count(cls, value: int) -> int:
        """Validates frame count for video generation."""
        if value != 102:  # Standard frame count from technical specs
            raise ValueError("Frame count must be 102 for standard generation")
            
        # Validate 24 FPS compatibility
        duration = value / 24  # Technical spec requires 24 FPS
        if not (4 <= duration <= 5):  # ~4-5 seconds of content
            raise ValueError("Frame count must support 4-5 seconds at 24 FPS")
            
        return value

    @validator('perspective')
    def validate_perspective(cls, value: str) -> str:
        """Validates perspective setting."""
        valid_perspectives = ['first_person', 'third_person']
        if value not in valid_perspectives:
            raise ValueError(f"Perspective must be one of {valid_perspectives}")
        return value

class GenerationBase(BaseModel):
    """Base Pydantic model for generation data with enhanced prompt validation."""
    prompt: constr(min_length=1, max_length=1000) = Field(
        ...,
        description="Text prompt for video generation",
        example="A medieval castle with knights patrolling the walls"
    )
    parameters: GenerationParameters = Field(
        ...,
        description="Video generation parameters"
    )

class GenerationCreate(GenerationBase):
    """Pydantic model for generation request validation."""
    class Config:
        schema_extra = {
            "example": {
                "prompt": "A medieval castle with knights patrolling the walls",
                "parameters": {
                    "resolution_width": 1280,
                    "resolution_height": 720,
                    "frame_count": 102,
                    "perspective": "third_person"
                }
            }
        }

class GenerationResponse(GenerationBase):
    """Pydantic model for generation response formatting with quality metrics validation."""
    id: UUID = Field(..., description="Unique generation identifier")
    status: GenerationStatus = Field(..., description="Current generation status")
    metrics: Dict = Field(
        default_factory=dict,
        description="Generation quality metrics"
    )
    created_at: datetime = Field(..., description="Generation request timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    @validator('metrics')
    def validate_metrics(cls, value: Dict) -> Dict:
        """Validates generation quality metrics against required thresholds."""
        if not value:
            return value
            
        # Validate FID score
        if 'fid' in value and value['fid'] >= 300:  # From technical specs
            raise ValueError(f"FID score {value['fid']} exceeds threshold 300")
            
        # Validate FVD score
        if 'fvd' in value and value['fvd'] >= 1000:  # From technical specs
            raise ValueError(f"FVD score {value['fvd']} exceeds threshold 1000")
            
        # Ensure all required metrics are present when completed
        if 'status' in value and value['status'] == GenerationStatus.COMPLETED:
            required_metrics = {'fid', 'fvd', 'success_rate'}
            if not all(metric in value for metric in required_metrics):
                raise ValueError(f"Completed generation requires all metrics: {required_metrics}")
                
        return value

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "prompt": "A medieval castle with knights patrolling the walls",
                "parameters": {
                    "resolution_width": 1280,
                    "resolution_height": 720,
                    "frame_count": 102,
                    "perspective": "third_person"
                },
                "status": "completed",
                "metrics": {
                    "fid": 250,
                    "fvd": 850,
                    "success_rate": 0.85
                },
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:01:00Z"
            }
        }