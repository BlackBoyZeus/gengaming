"""
Alembic migrations environment configuration module for GameGen-X.
Implements secure database schema migrations with FreeBSD compatibility.

Alembic version: ^1.11.0
SQLAlchemy version: ^1.4.41
"""

import logging
from logging.handlers import SysLogHandler
import os
from typing import Optional, Union

from alembic import context
from sqlalchemy import engine_from_config, pool, create_engine
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from core.config import settings
from db.base import Base

# Initialize secure logging
logger = logging.getLogger('alembic.env')
logger.addHandler(SysLogHandler(address='/var/run/log' if os.path.exists('/var/run/log') else '/dev/log'))
logger.setLevel(logging.INFO)

# Configure Alembic context
config = context.config

# Set SQLAlchemy metadata for migrations
target_metadata = Base.metadata

# Security-sensitive database objects requiring special handling
RESTRICTED_OBJECTS = [
    'user_credentials',
    'security_tokens',
    'api_keys',
    'audit_logs'
]

class MigrationContext:
    """Secure context manager for database migrations with FreeBSD compatibility."""
    
    def __init__(self, connection: Optional[Connection] = None):
        self.connection = connection
        self.transaction = None
        
    def __enter__(self):
        """Enter migration context with security checks."""
        if self.connection:
            # Set FreeBSD-compatible isolation level
            self.connection.execution_options(
                isolation_level="REPEATABLE READ"
            )
            self.transaction = self.connection.begin()
            
        logger.info("Entered secure migration context")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit migration context with proper cleanup."""
        try:
            if exc_type is not None:
                if self.transaction:
                    logger.error(f"Rolling back migration due to: {exc_val}")
                    self.transaction.rollback()
            elif self.transaction:
                logger.info("Committing successful migration")
                self.transaction.commit()
        finally:
            if self.connection:
                self.connection.close()
            logger.info("Exited migration context")

def include_object(object: Union[str, object], name: str, type_: str, *args, **kwargs) -> bool:
    """
    Security filter for database objects during migration.
    
    Args:
        object: Database object to evaluate
        name: Object name
        type_: Object type
        
    Returns:
        bool: Whether object should be included in migration
    """
    # Restrict sensitive objects in non-production
    if settings.environment != "production" and name in RESTRICTED_OBJECTS:
        logger.warning(f"Restricted object {name} excluded from migration in {settings.environment}")
        return False
        
    # Log all schema modifications
    if type_ == "table":
        logger.info(f"Including table {name} in migration")
    elif type_ == "index":
        logger.info(f"Including index {name} in migration")
        
    return True

def get_engine_url() -> str:
    """Get database URL with FreeBSD-compatible options."""
    url = str(settings.database_url)
    
    # Add FreeBSD-specific connection parameters
    if settings.environment == "production":
        url += "?sslmode=verify-full"
    else:
        url += "?sslmode=prefer"
        
    return url

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode for script generation.
    Implements secure logging and FreeBSD compatibility.
    """
    try:
        url = get_engine_url()
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            include_object=include_object,
            include_schemas=True,
            version_table_schema=target_metadata.schema
        )

        logger.info("Starting offline migration")
        with context.begin_transaction():
            context.run_migrations()
        logger.info("Completed offline migration")
            
    except Exception as e:
        logger.error(f"Offline migration failed: {str(e)}")
        raise

def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode with secure transaction management.
    Implements FreeBSD compatibility and comprehensive logging.
    """
    try:
        # Configure FreeBSD-compatible engine
        engine_config = config.get_section(config.config_ini_section)
        engine_config.update({
            "url": get_engine_url(),
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 10,
            "connect_args": {
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        })
        
        engine = engine_from_config(
            engine_config,
            prefix="sqlalchemy.",
            poolclass=pool.QueuePool
        )

        with engine.connect() as connection:
            with MigrationContext(connection) as migration_ctx:
                context.configure(
                    connection=connection,
                    target_metadata=target_metadata,
                    include_object=include_object,
                    include_schemas=True,
                    version_table_schema=target_metadata.schema,
                    transaction_per_migration=True
                )
                
                logger.info("Starting online migration")
                with context.begin_transaction():
                    context.run_migrations()
                logger.info("Completed online migration")
                
    except SQLAlchemyError as e:
        logger.error(f"Database migration failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Migration error: {str(e)}")
        raise

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()