"""
Database session management module for GameGen-X.
Implements FreeBSD-compatible connection pooling, secure session handling,
and comprehensive monitoring.

SQLAlchemy version: ^1.4.41
"""

from contextlib import contextmanager
from typing import Generator, Dict, Any
import ssl
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Internal imports
from core.config import settings
from db.base import Base
from core.logging import logger

# Create SSL context for secure database connections
ssl_context = ssl.create_default_context()
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Configure database engine with FreeBSD-optimized settings
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=20,
    pool_recycle=300,
    connect_args={
        'ssl_context': ssl_context,
        'keepalives': 1,
        'keepalives_idle': 30,
        'keepalives_interval': 10,
        'keepalives_count': 5
    }
)

# Create session factory with security settings
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Create thread-local session registry
db_session = scoped_session(SessionLocal)

class DatabaseSession:
    """Secure context manager for database session handling with monitoring."""
    
    def __init__(self):
        self.session = None
        self.metrics: Dict[str, Any] = {
            'start_time': None,
            'queries': 0,
            'errors': 0
        }

    def __enter__(self):
        """Enters session context with validation and monitoring."""
        self.session = SessionLocal()
        
        # Configure session monitoring
        @event.listens_for(self.session, 'after_execute')
        def receive_after_execute(conn, clauseelement, multiparams, params, execution_options):
            self.metrics['queries'] += 1
        
        @event.listens_for(self.session, 'handle_error')
        def receive_error(context):
            self.metrics['errors'] += 1
            logger.error(f"Database error: {str(context.original_exception)}")
        
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exits session context with cleanup validation."""
        try:
            if exc_type is not None:
                self.session.rollback()
                logger.error(f"Session error: {str(exc_val)}")
            self.session.close()
        finally:
            self.session.remove()

@contextmanager
def get_session() -> Generator:
    """Creates and returns a new database session with monitoring."""
    session = DatabaseSession()
    try:
        with session as db:
            yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        if session.metrics['errors'] > 0:
            logger.warning(f"Session completed with {session.metrics['errors']} errors")

def init_db() -> None:
    """Initializes database schema with validation and monitoring."""
    try:
        # Import all models to register with Base
        import db.models  # noqa
        
        # Create database schema
        Base.metadata.create_all(bind=engine)
        
        # Setup engine monitoring
        @event.listens_for(engine, 'connect')
        def receive_connect(dbapi_connection, connection_record):
            logger.info("New database connection established")
        
        @event.listens_for(engine, 'checkout')
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            logger.info("Database connection checked out from pool")
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def dispose_engine() -> None:
    """Safely disposes database engine with cleanup validation."""
    try:
        logger.info("Initiating database engine disposal")
        db_session.remove()
        engine.dispose()
        logger.info("Database engine disposed successfully")
    except Exception as e:
        logger.error(f"Engine disposal failed: {str(e)}")
        raise

def monitor_pool() -> Dict[str, Any]:
    """Monitors database connection pool health and metrics."""
    try:
        pool_status = {
            'size': engine.pool.size(),
            'checkedin': engine.pool.checkedin(),
            'overflow': engine.pool.overflow(),
            'checkedout': engine.pool.checkedout(),
        }
        logger.info("Pool status", extra=pool_status)
        return pool_status
    except Exception as e:
        logger.error(f"Pool monitoring failed: {str(e)}")
        raise

# Configure engine event listeners for monitoring
@event.listens_for(engine, 'engine_connect')
def engine_connect(conn, branch):
    logger.info("Engine connection established")

@event.listens_for(engine, 'engine_disconnect')
def engine_disconnect(conn, branch):
    logger.info("Engine connection closed")

@event.listens_for(engine, 'begin')
def begin_transaction(conn):
    logger.info("Transaction started")

@event.listens_for(engine, 'commit')
def commit_transaction(conn):
    logger.info("Transaction committed")

@event.listens_for(engine, 'rollback')
def rollback_transaction(conn):
    logger.info("Transaction rolled back")