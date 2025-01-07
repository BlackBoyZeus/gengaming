"""
Database migrations package initialization module for GameGen-X.
Provides secure schema versioning, migration tracking, and validated access to migration utilities.

Alembic version: ^1.11.1
"""

import os
import logging
from typing import Optional, Callable, Any
from functools import wraps
from alembic import context
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

from db.base import Base

# Configure secure paths for migration files
MIGRATION_FOLDER = os.path.dirname(os.path.abspath(__file__))
VERSION_FOLDER = os.path.join(MIGRATION_FOLDER, 'versions')
MIGRATION_HISTORY_FILE = os.path.join(MIGRATION_FOLDER, 'history.log')

# Configure secure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add secure file handler with UTC timestamps
file_handler = logging.FileHandler(MIGRATION_HISTORY_FILE)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s UTC - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)

def validate_context(func: Callable) -> Callable:
    """
    Decorator that validates alembic context integrity and security.
    
    Args:
        func: Function to wrap with validation
        
    Returns:
        Wrapped function with context validation
        
    Raises:
        RuntimeError: If context validation fails
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not context.is_offline_mode():
            logger.error("Migration context must be in offline mode")
            raise RuntimeError("Invalid migration context: Must be in offline mode")
            
        if not hasattr(context, 'config'):
            logger.error("Migration context missing configuration")
            raise RuntimeError("Invalid migration context: Missing configuration")
            
        # Validate script directory exists and is secure
        script_dir = ScriptDirectory.from_config(context.config)
        if not os.path.exists(script_dir.dir):
            logger.error(f"Migration script directory not found: {script_dir.dir}")
            raise RuntimeError("Invalid migration context: Script directory not found")
            
        # Validate database connection is configured
        if not context.get_bind():
            logger.error("Database connection not configured in context")
            raise RuntimeError("Invalid migration context: No database connection")
            
        logger.info(f"Context validation passed for {func.__name__}")
        return func(*args, **kwargs)
        
    return wrapper

@validate_context
def get_current_revision() -> Optional[str]:
    """
    Securely retrieves and validates current database schema revision with logging.
    
    Returns:
        Current revision identifier or None if not found
        
    Raises:
        RuntimeError: If revision validation fails
    """
    logger.info("Attempting to retrieve current revision")
    
    try:
        # Get current revision with error handling
        migration_context = MigrationContext.configure(context.get_bind())
        current = migration_context.get_current_revision()
        
        # Validate revision format
        if current and not isinstance(current, str):
            logger.error(f"Invalid revision format: {type(current)}")
            raise RuntimeError("Invalid revision format")
            
        if current:
            logger.info(f"Successfully retrieved current revision: {current}")
        else:
            logger.info("No current revision found")
            
        return current
        
    except Exception as e:
        logger.error(f"Error retrieving current revision: {str(e)}")
        raise RuntimeError(f"Failed to get current revision: {str(e)}")

@validate_context        
def get_head_revision() -> Optional[str]:
    """
    Securely retrieves and validates latest available migration revision with logging.
    
    Returns:
        Head revision identifier or None if not found
        
    Raises:
        RuntimeError: If revision validation fails
    """
    logger.info("Attempting to retrieve head revision")
    
    try:
        # Get head revision with error handling
        script_dir = ScriptDirectory.from_config(context.config)
        head = script_dir.get_current_head()
        
        # Validate revision format
        if head and not isinstance(head, str):
            logger.error(f"Invalid head revision format: {type(head)}")
            raise RuntimeError("Invalid head revision format")
            
        if head:
            logger.info(f"Successfully retrieved head revision: {head}")
        else:
            logger.info("No head revision found")
            
        return head
        
    except Exception as e:
        logger.error(f"Error retrieving head revision: {str(e)}")
        raise RuntimeError(f"Failed to get head revision: {str(e)}")

# Export secure revision access functions
__all__ = ['get_current_revision', 'get_head_revision']