"""
SQLAlchemy model for managing video generation requests and metadata.
Handles text prompts, generation parameters, and relationships with videos and control signals.

SQLAlchemy version: ^1.4.41
"""

from datetime import datetime, UTC
from enum import Enum
from uuid import uuid4
from sqlalchemy import Column, String, Enum as SQLEnum, event
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship, validates
from db.base import BaseModel

# Generation status enum
class GenerationStatus(str, Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

# System constants
VALID_RESOLUTIONS = ['720p']
VALID_FRAME_RATES = [24]
METRIC_THRESHOLDS = {
    'fid': 300,  # Fréchet Inception Distance threshold
    'fvd': 1000  # Fréchet Video Distance threshold
}

# Valid status transitions
STATUS_TRANSITIONS = {
    GenerationStatus.PENDING: [GenerationStatus.PROCESSING, GenerationStatus.FAILED],
    GenerationStatus.PROCESSING: [GenerationStatus.COMPLETED, GenerationStatus.FAILED],
    GenerationStatus.COMPLETED: [],
    GenerationStatus.FAILED: []
}

class Generation(BaseModel):
    """
    Enhanced SQLAlchemy model for managing video generation requests with quality metrics validation.
    Implements comprehensive tracking of generation parameters, metrics, and audit trail.
    """
    
    __tablename__ = 'generation'

    # Primary key using secure UUID
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Core fields
    prompt = Column(String(1000), nullable=False)
    status = Column(SQLEnum(GenerationStatus), nullable=False, default=GenerationStatus.PENDING)
    failure_reason = Column(String(500))
    
    # JSON fields for flexible storage
    parameters = Column(JSON, nullable=False)
    metrics = Column(JSON, default=lambda: {'thresholds': METRIC_THRESHOLDS})
    audit_log = Column(JSON, default=list)

    # Relationships
    videos = relationship("Video", back_populates="generation", cascade="all, delete-orphan")
    controls = relationship("Control", back_populates="generation", cascade="all, delete-orphan")

    def __init__(self, prompt: str, parameters: dict):
        """
        Initialize generation model with enhanced validation.
        
        Args:
            prompt (str): Text prompt for video generation
            parameters (dict): Generation parameters including resolution and FPS
        """
        super().__init__()
        
        self.id = uuid4()
        self.prompt = prompt
        
        # Validate parameters
        if not self.validate_parameters(parameters):
            raise ValueError("Invalid generation parameters")
        self.parameters = parameters
        
        # Initialize metrics with thresholds
        self.metrics = {
            'thresholds': METRIC_THRESHOLDS,
            'current': {},
            'success_rate': None
        }
        
        # Initialize audit log
        self.audit_log = [{
            'timestamp': datetime.now(UTC).isoformat(),
            'action': 'created',
            'details': {'prompt': prompt, 'parameters': parameters}
        }]

    @validates('status')
    def validate_status(self, key, value):
        """Validate status transitions."""
        if hasattr(self, 'status'):
            current_status = self.status
            if value not in STATUS_TRANSITIONS[current_status]:
                raise ValueError(f"Invalid status transition from {current_status} to {value}")
        return value

    def update_status(self, new_status: GenerationStatus, reason: str = None) -> None:
        """
        Update generation status with enhanced state transition validation.
        
        Args:
            new_status (GenerationStatus): New status to set
            reason (str, optional): Reason for status change, required for failures
        """
        if new_status == GenerationStatus.FAILED and not reason:
            raise ValueError("Failure reason is required when setting failed status")
            
        self.status = new_status
        if new_status == GenerationStatus.FAILED:
            self.failure_reason = reason
            
        # Log status change
        self.audit_log.append({
            'timestamp': datetime.now(UTC).isoformat(),
            'action': 'status_change',
            'details': {
                'from': self.status,
                'to': new_status,
                'reason': reason
            }
        })
        
        self.update_timestamp()

    def update_metrics(self, new_metrics: dict) -> None:
        """
        Update generation quality metrics with enhanced validation.
        
        Args:
            new_metrics (dict): New metrics including FID and FVD scores
        """
        # Validate metrics against thresholds
        if 'fid' in new_metrics and new_metrics['fid'] > METRIC_THRESHOLDS['fid']:
            raise ValueError(f"FID score {new_metrics['fid']} exceeds threshold {METRIC_THRESHOLDS['fid']}")
            
        if 'fvd' in new_metrics and new_metrics['fvd'] > METRIC_THRESHOLDS['fvd']:
            raise ValueError(f"FVD score {new_metrics['fvd']} exceeds threshold {METRIC_THRESHOLDS['fvd']}")
            
        # Update metrics
        self.metrics['current'] = new_metrics
        self.metrics['current']['timestamp'] = datetime.now(UTC).isoformat()
        
        # Calculate success rate
        if 'fid' in new_metrics and 'fvd' in new_metrics:
            fid_success = new_metrics['fid'] < METRIC_THRESHOLDS['fid']
            fvd_success = new_metrics['fvd'] < METRIC_THRESHOLDS['fvd']
            self.metrics['success_rate'] = (fid_success + fvd_success) / 2
            
        # Log metrics update
        self.audit_log.append({
            'timestamp': datetime.now(UTC).isoformat(),
            'action': 'metrics_update',
            'details': new_metrics
        })
        
        self.update_timestamp()

    @staticmethod
    def validate_parameters(parameters: dict) -> bool:
        """
        Validate generation parameters against system requirements.
        
        Args:
            parameters (dict): Parameters to validate
            
        Returns:
            bool: True if parameters are valid, False otherwise
        """
        if not parameters:
            return False
            
        # Validate resolution
        if 'resolution' not in parameters or parameters['resolution'] not in VALID_RESOLUTIONS:
            return False
            
        # Validate frame rate
        if 'fps' not in parameters or parameters['fps'] not in VALID_FRAME_RATES:
            return False
            
        return True

    def to_dict(self, include_audit_log: bool = False) -> dict:
        """
        Convert generation model to dictionary with enhanced metadata.
        
        Args:
            include_audit_log (bool): Whether to include audit log in output
            
        Returns:
            dict: Dictionary representation of generation data
        """
        data = super().to_dict()
        
        # Include relationships count
        data['video_count'] = len(self.videos)
        data['control_count'] = len(self.controls)
        
        # Include audit log if requested
        if include_audit_log:
            data['audit_log'] = self.audit_log
            
        return data

@event.listens_for(Generation, 'before_update')
def generation_before_update(mapper, connection, target):
    """Event listener to update timestamp before any updates."""
    target.update_timestamp()