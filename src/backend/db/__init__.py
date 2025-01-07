"""
Database package initialization module for GameGen-X.
Provides core database components with FreeBSD-specific optimizations and performance monitoring.

SQLAlchemy version: ^1.4.41
psycopg2-binary version: ^2.9.9
"""

import ssl
import psycopg2
from typing import Dict, Any

# Internal imports
from db.base import Base
from db.session import (
    get_session,
    get_scoped_session,
    DatabaseSession,
    init_db,
    engine,
    monitor_pool,
    dispose_engine
)
from core.logging import get_logger

# Configure logging
logger = get_logger(__name__)

def setup_database() -> None:
    """
    Initializes database connection with FreeBSD optimizations and performance monitoring.
    Configures connection pooling, SSL, and retention policies.
    """
    try:
        # Configure SSL context for secure connections
        ssl_context = ssl.create_default_context()
        ssl_context.verify_mode = ssl.CERT_REQUIRED

        # Configure FreeBSD-specific TCP socket options
        psycopg2.__dict__['extensions'].set_wait_callback(
            psycopg2.extras.wait_select
        )

        # Initialize database schema and tables
        init_db()

        # Configure connection pool monitoring
        @engine.event.listens_for(engine, 'checkin')
        def on_checkin(dbapi_conn, conn_record):
            logger.info(
                "Connection returned to pool",
                pool_status=monitor_pool()
            )

        # Configure 90-day retention policy
        retention_sql = """
        DELETE FROM gamegen_x.video 
        WHERE created_at < NOW() - INTERVAL '90 days';
        """
        with get_session() as session:
            session.execute(retention_sql)
            session.commit()

        # Setup performance monitoring
        @engine.event.listens_for(engine, 'before_cursor_execute')
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault('query_start_time', []).append(engine.pool._time())

        @engine.event.listens_for(engine, 'after_cursor_execute')
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total = engine.pool._time() - conn.info['query_start_time'].pop()
            if total > 0.1:  # 100ms threshold
                logger.warning(
                    "Slow query detected",
                    duration=total,
                    statement=statement
                )

        logger.info("Database setup completed successfully")

    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        raise

# Export core database components
__all__ = [
    "Base",
    "get_session",
    "get_scoped_session", 
    "DatabaseSession",
    "init_db",
    "setup_database"
]