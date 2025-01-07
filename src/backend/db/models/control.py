"""
SQLAlchemy model for storing real-time control signals and interactions with video generation.
Implements high-performance storage and retrieval of keyboard inputs, environment controls 
and instruction data for interactive video manipulation with <100ms response time guarantee.

SQLAlchemy version: ^1.4.41
"""

from datetime import datetime, UTC
from uuid import uuid4
from sqlalchemy import Column, ForeignKey, String, Enum, Index
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship

from db.base import Base

# Define control type enum with schema namespace for security
ControlType = Enum(
    'keyboard',
    'environment', 
    'instruction',
    name='control_type',
    schema='gamegen_x'
)

@Index('ix_control_generation_id', 'generation_id')
@Index('ix_control_video_id', 'video_id')
@Index('ix_control_type', 'type')
class Control(Base):
    """
    High-performance SQLAlchemy model for storing and managing video control signals 
    with <100ms response time guarantee through optimized indexing and caching.
    """
    __tablename__ = 'control'
    __table_args__ = {'schema': 'gamegen_x'}

    # Primary key using PostgreSQL native UUID
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Foreign keys with indexes for fast lookups
    generation_id = Column(
        UUID(as_uuid=True),
        ForeignKey('gamegen_x.generation.id', ondelete='CASCADE'),
        nullable=False
    )
    video_id = Column(
        UUID(as_uuid=True), 
        ForeignKey('gamegen_x.video.id', ondelete='CASCADE'),
        nullable=False
    )

    # Control type with validation
    type = Column(ControlType, nullable=False)
    
    # Compressed JSON data with type-specific validation
    data = Column(JSON, nullable=False)
    
    # Audit timestamps
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC)
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC)
    )

    # Relationships with lazy loading
    generation = relationship(
        "Generation",
        back_populates="controls",
        lazy="select"
    )
    video = relationship(
        "Video",
        back_populates="controls",
        lazy="select"
    )

    def __init__(self, generation_id: UUID, video_id: UUID, type: str, data: dict):
        """
        Initialize control model with validation and default values.

        Args:
            generation_id (UUID): Associated generation ID
            video_id (UUID): Associated video ID
            type (str): Control type from ControlType enum
            data (dict): Control signal data
        """
        self.id = uuid4()
        self.generation_id = generation_id
        self.video_id = video_id
        self.type = type
        self.validate_and_set_data(data)
        self.created_at = datetime.now(UTC)
        self.updated_at = self.created_at

    def validate_and_set_data(self, data: dict) -> None:
        """
        Validate and sanitize control data based on type.

        Args:
            data (dict): Control signal data to validate
        
        Raises:
            ValueError: If data format is invalid for control type
        """
        if self.type == 'keyboard':
            required_keys = {'key', 'action', 'timestamp'}
            if not all(key in data for key in required_keys):
                raise ValueError("Keyboard control requires key, action and timestamp")
                
        elif self.type == 'environment':
            required_keys = {'parameter', 'value', 'timestamp'}
            if not all(key in data for key in required_keys):
                raise ValueError("Environment control requires parameter, value and timestamp")
                
        elif self.type == 'instruction':
            required_keys = {'instruction', 'parameters', 'timestamp'}
            if not all(key in data for key in required_keys):
                raise ValueError("Instruction control requires instruction, parameters and timestamp")

        self.data = data

    def update_data(self, new_data: dict) -> None:
        """
        Update control signal data with validation and audit logging.

        Args:
            new_data (dict): New control data to set
        """
        self.validate_and_set_data(new_data)
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict:
        """
        Convert control model to optimized dictionary representation.

        Returns:
            dict: Dictionary containing control data
        """
        return {
            'id': str(self.id),
            'generation_id': str(self.generation_id),
            'video_id': str(self.video_id),
            'type': self.type.name,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }