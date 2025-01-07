"""
SQLAlchemy model for secure video data storage with enhanced validation and audit capabilities.
Implements data classification, quality metrics tracking, and relationship management.

SQLAlchemy version: ^1.4.41
"""

import enum
from datetime import datetime, UTC
from uuid import uuid4
from sqlalchemy import Column, ForeignKey, String, Integer, Enum, event
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship, validates
from sqlalchemy.schema import CheckConstraint
from db.base import BaseModel

class VideoStatus(enum.Enum):
    """Enum defining valid video status values with transition validation."""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

# Define valid status transitions for enhanced security
VALID_STATUS_TRANSITIONS = {
    (VideoStatus.PENDING, VideoStatus.PROCESSING),
    (VideoStatus.PROCESSING, VideoStatus.COMPLETED),
    (VideoStatus.PROCESSING, VideoStatus.FAILED)
}

# Schema for validating video metadata
METADATA_SCHEMA = {
    'type': 'object',
    'required': [
        'security_classification',
        'frame_rate',
        'resolution',
        'version',
        'quality_metrics'
    ],
    'properties': {
        'security_classification': {'type': 'string', 'enum': ['public', 'internal', 'confidential']},
        'frame_rate': {'type': 'number', 'minimum': 24},
        'resolution': {
            'type': 'object',
            'properties': {
                'width': {'type': 'integer', 'maximum': 1280},
                'height': {'type': 'integer', 'maximum': 720}
            }
        },
        'version': {'type': 'integer', 'minimum': 1},
        'quality_metrics': {
            'type': 'object',
            'properties': {
                'fid_score': {'type': 'number'},
                'fvd_score': {'type': 'number'}
            }
        }
    }
}

class Video(BaseModel):
    """Enhanced SQLAlchemy model for secure video data storage with audit trail and validation."""
    
    __tablename__ = 'videos'

    # Primary and foreign keys with UUID for enhanced security
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    generation_id = Column(UUID(as_uuid=True), ForeignKey('generations.id', ondelete='CASCADE'), nullable=False)

    # Core video attributes with validation
    status = Column(
        Enum(VideoStatus, name='video_status'),
        nullable=False,
        default=VideoStatus.PENDING
    )
    frame_count = Column(Integer, nullable=False)
    format = Column(String(10), nullable=False)
    metadata = Column(JSON, nullable=False)

    # Relationships with cascade rules
    generation = relationship("Generation", back_populates="videos")
    controls = relationship("Control", back_populates="video", cascade="all, delete-orphan")

    # Audit trail columns
    created_by = Column(String(255), nullable=False)
    updated_by = Column(String(255), nullable=False)

    # Quality constraints
    __table_args__ = (
        CheckConstraint('frame_count > 0', name='ck_frame_count_positive'),
        CheckConstraint("format IN ('mp4', 'webm')", name='ck_valid_format')
    )

    def __init__(self, generation_id: UUID, frame_count: int, format: str, metadata: dict, created_by: str):
        """Initialize video model with enhanced security and validation."""
        super().__init__()
        self.id = uuid4()
        self.generation_id = generation_id
        self.status = VideoStatus.PENDING
        self.frame_count = frame_count
        self.format = format
        self.metadata = self._initialize_metadata(metadata)
        self.created_by = created_by
        self.updated_by = created_by

    @validates('status')
    def validate_status(self, key, value):
        """Validate status transitions."""
        if hasattr(self, 'status') and self.status is not None:
            current = self.status
            new = value if isinstance(value, VideoStatus) else VideoStatus(value)
            if (current, new) not in VALID_STATUS_TRANSITIONS:
                raise ValueError(f"Invalid status transition: {current.value} -> {new.value}")
        return value if isinstance(value, VideoStatus) else VideoStatus(value)

    @validates('frame_count')
    def validate_frame_count(self, key, value):
        """Validate frame count meets quality requirements."""
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Frame count must be a positive integer")
        return value

    @validates('format')
    def validate_format(self, key, value):
        """Validate video format."""
        if value not in ('mp4', 'webm'):
            raise ValueError("Invalid video format")
        return value

    @validates('metadata')
    def validate_metadata(self, key, value):
        """Validate metadata schema compliance."""
        from jsonschema import validate
        validate(instance=value, schema=METADATA_SCHEMA)
        return value

    def update_status(self, new_status: VideoStatus, updated_by: str) -> None:
        """Update video status with transition validation and audit trail."""
        self.status = new_status
        self.updated_by = updated_by
        self.update_timestamp()

    def update_metadata(self, new_metadata: dict, updated_by: str) -> None:
        """Update video metadata with schema validation and versioning."""
        new_metadata['version'] = self.metadata.get('version', 0) + 1
        self.validate_metadata('metadata', new_metadata)
        self.metadata = new_metadata
        self.updated_by = updated_by
        self.update_timestamp()

    def to_dict(self, include_relationships: bool = False, security_level: str = 'public') -> dict:
        """Convert video model to dictionary with security filtering."""
        data = super().to_dict()
        
        # Filter metadata based on security level
        if security_level != 'confidential':
            data['metadata'] = {
                k: v for k, v in self.metadata.items()
                if k not in ('security_classification', 'quality_metrics')
            }
            
        if include_relationships:
            data['generation'] = self.generation.to_dict() if self.generation else None
            data['controls'] = [control.to_dict() for control in self.controls]
            
        return data

    def _initialize_metadata(self, metadata: dict) -> dict:
        """Initialize metadata with security classification and version."""
        metadata.setdefault('version', 1)
        metadata.setdefault('security_classification', 'public')
        self.validate_metadata('metadata', metadata)
        return metadata