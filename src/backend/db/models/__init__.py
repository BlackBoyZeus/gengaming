"""
SQLAlchemy models initialization module for GameGen-X database.
Provides centralized access to all database models with comprehensive type safety and documentation.

SQLAlchemy version: ^1.4.41
"""

from db.models.user import User, UserRole
from db.models.video import Video, VideoStatus
from db.models.control import Control
from db.models.generation import Generation, GenerationStatus

# Export all models and enums for type-safe access
__all__ = [
    # User management models
    "User",
    "UserRole",
    
    # Video data models
    "Video", 
    "VideoStatus",
    
    # Control signal models
    "Control",
    
    # Generation request models
    "Generation",
    "GenerationStatus"
]