"""
Pydantic schema models for system status, resource metrics, and performance monitoring.
Provides data validation and serialization for monitoring endpoints.
"""

from datetime import datetime, timezone
from typing import Literal
from functools import wraps

from pydantic import BaseModel, Field, validator, constr, confloat, conint

# Custom validation decorators
def validate_version_format(cls):
    @validator('version', pre=True)
    def check_version(v):
        if not isinstance(v, str) or not v.count('.') == 2:
            raise ValueError('Version must be in format X.Y.Z')
        return v
    cls.validate_version = check_version
    return cls

def validate_percentages(cls):
    @validator('*_percent', pre=True)
    def check_percentage(v):
        if not 0 <= float(v) <= 100:
            raise ValueError('Percentage must be between 0 and 100')
        return round(float(v), 2)
    cls.validate_percentage = check_percentage
    return cls

def validate_performance_thresholds(cls):
    @validator('generation_latency_ms')
    def check_latency(v):
        if v > 1000:
            raise ValueError('Generation latency exceeds maximum threshold of 1000ms')
        return v
    
    @validator('frame_rate_fps')
    def check_fps(v):
        if not 0 <= v <= 60:
            raise ValueError('Frame rate must be between 0 and 60 FPS')
        return v
    
    @validator('control_response_ms')
    def check_response(v):
        if v > 200:
            raise ValueError('Control response exceeds maximum threshold of 200ms')
        return v
    
    cls.validate_latency = check_latency
    cls.validate_fps = check_fps
    cls.validate_response = check_response
    return cls

@validate_version_format
class SystemStatus(BaseModel):
    """System health status and uptime information schema."""
    status: Literal['healthy', 'degraded', 'unhealthy'] = Field(
        ...,
        description="Current system health status"
    )
    uptime_seconds: confloat(gt=0) = Field(
        ...,
        description="System uptime in seconds"
    )
    last_restart: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last system restart timestamp in UTC"
    )
    version: constr(regex=r'^\d+\.\d+\.\d+$') = Field(
        ...,
        description="System version in semantic versioning format"
    )
    environment: Literal['development', 'staging', 'production'] = Field(
        ...,
        description="Deployment environment"
    )

@validate_percentages
class ResourceMetrics(BaseModel):
    """System resource utilization metrics schema."""
    cpu_usage_percent: confloat(ge=0, le=100) = Field(
        default=0.0,
        description="CPU utilization percentage"
    )
    memory_usage_percent: confloat(ge=0, le=100) = Field(
        default=0.0,
        description="Memory utilization percentage"
    )
    gpu_usage_percent: confloat(ge=0, le=100) = Field(
        default=0.0,
        description="GPU utilization percentage"
    )
    gpu_memory_percent: confloat(ge=0, le=100) = Field(
        default=0.0,
        description="GPU memory utilization percentage"
    )
    disk_usage_percent: confloat(ge=0, le=100) = Field(
        default=0.0,
        description="Disk space utilization percentage"
    )

@validate_performance_thresholds
class PerformanceMetrics(BaseModel):
    """System performance metrics schema."""
    generation_latency_ms: confloat(ge=0, le=1000) = Field(
        ...,
        description="Video generation latency in milliseconds"
    )
    frame_rate_fps: confloat(ge=0, le=60) = Field(
        ...,
        description="Video frame rate in frames per second"
    )
    control_response_ms: confloat(ge=0, le=200) = Field(
        ...,
        description="Control input response time in milliseconds"
    )
    concurrent_users: conint(ge=0) = Field(
        default=0,
        description="Number of concurrent system users"
    )

class StatusResponse(BaseModel):
    """Combined system status response schema."""
    system_status: SystemStatus = Field(
        ...,
        description="System health status information"
    )
    resource_metrics: ResourceMetrics = Field(
        ...,
        description="Resource utilization metrics"
    )
    performance_metrics: PerformanceMetrics = Field(
        ...,
        description="System performance metrics"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp in UTC"
    )

    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Ensure timestamp is in UTC timezone."""
        if v.tzinfo != timezone.utc:
            raise ValueError('Timestamp must be in UTC timezone')
        return v

    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }