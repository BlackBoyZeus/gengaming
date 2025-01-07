"""
Core SQLAlchemy declarative base module for GameGen-X database models.
Provides secure database configuration, metadata handling, and audit trail capabilities.

SQLAlchemy version: ^1.4.41
"""

from datetime import datetime, UTC
from sqlalchemy import MetaData, Column, DateTime, func
from sqlalchemy.ext.declarative import declarative_base, declared_attr

# Configure secure naming conventions for database constraints
metadata = MetaData(
    naming_convention={
        'ix': 'ix_%(column_0_label)s',
        'uq': 'uq_%(table_name)s_%(column_0_name)s',
        'ck': 'ck_%(table_name)s_%(constraint_name)s',
        'fk': 'fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s',
        'pk': 'pk_%(table_name)s'
    },
    schema='gamegen_x'  # Isolate tables in dedicated schema
)

# Create declarative base with configured metadata
Base = declarative_base(metadata=metadata)

class BaseModel(Base):
    """
    Abstract base model providing common functionality for all database models.
    Implements secure audit trails with UTC timestamps.
    """
    
    __abstract__ = True

    # Audit trail timestamps using UTC timezone
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now()
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        onupdate=func.now()
    )

    def __init__(self, *args, **kwargs):
        """Initialize model with secure UTC timestamps."""
        super().__init__(*args, **kwargs)
        now = datetime.now(UTC)
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    @declared_attr
    def __tablename__(cls) -> str:
        """
        Generate secure table name from class name.
        
        Returns:
            str: Lowercase sanitized table name
        """
        # Convert CamelCase to snake_case securely
        name = cls.__name__
        return ''.join(
            ['_' + c.lower() if c.isupper() else c.lower() 
             for c in name]
        ).lstrip('_')

    def to_dict(self) -> dict:
        """
        Convert model instance to secure dictionary representation.
        
        Returns:
            dict: Sanitized model data dictionary
        """
        data = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            
            # Format timestamps to ISO format with UTC
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    value = value.replace(tzinfo=UTC)
                value = value.isoformat()
                
            # Only include non-None values
            if value is not None:
                data[column.name] = value
                
        return data

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp securely with current UTC time."""
        self.updated_at = datetime.now(UTC)