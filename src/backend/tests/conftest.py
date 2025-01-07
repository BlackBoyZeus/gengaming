# External imports with versions
import pytest  # pytest ^7.3.1
from fastapi.testclient import TestClient  # fastapi ^0.95.0
from sqlalchemy import create_engine  # sqlalchemy ^1.4.41
from httpx import AsyncClient  # httpx ^0.24.0
import time
import logging
from typing import Dict, Any, Generator
import asyncio

# Internal imports
from api.main import app
from db.base import Base
from db.session import get_session, DatabaseSession

# Global test constants
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_PERFORMANCE_THRESHOLD = 100  # 100ms max response time
TEST_SECURITY_HEADERS = {"X-Test-Security": "enabled"}

def pytest_configure(config):
    """Configure test environment with comprehensive monitoring."""
    # Configure test logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize test metrics
    config.test_metrics = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'performance_violations': 0,
        'start_time': time.time()
    }
    
    # Configure test database
    config.test_db_engine = create_test_database()
    
    # Set up test environment variables
    config.test_environment = {
        'database_url': TEST_DATABASE_URL,
        'performance_threshold': TEST_PERFORMANCE_THRESHOLD,
        'security_headers': TEST_SECURITY_HEADERS
    }

def create_test_database():
    """Creates isolated in-memory test database."""
    engine = create_engine(
        TEST_DATABASE_URL,
        pool_pre_ping=True,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    return engine

@pytest.fixture
def client() -> Generator:
    """FastAPI test client fixture with security and monitoring."""
    with TestClient(app) as test_client:
        # Configure security headers
        test_client.headers.update(TEST_SECURITY_HEADERS)
        
        # Add performance monitoring middleware
        @app.middleware("http")
        async def add_performance_metrics(request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Check performance threshold
            if process_time > (TEST_PERFORMANCE_THRESHOLD / 1000):
                logging.warning(f"Performance threshold exceeded: {process_time*1000:.2f}ms")
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
        
        yield test_client

@pytest.fixture
async def async_client() -> Generator:
    """Async client fixture for WebSocket testing."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        # Configure WebSocket settings
        ac.ws_connect_timeout = 5.0
        ac.ws_receive_timeout = 5.0
        
        # Add WebSocket error tracking
        ac.ws_errors = []
        
        # Configure WebSocket close handler
        async def on_ws_close(ws, code, reason):
            if code != 1000:  # Normal closure
                ac.ws_errors.append({"code": code, "reason": reason})
        
        ac._ws_close_handler = on_ws_close
        yield ac

@pytest.fixture
def db_session(request) -> Generator:
    """Database session fixture with isolation and cleanup."""
    # Create test database session
    connection = request.config.test_db_engine.connect()
    transaction = connection.begin()
    session = DatabaseSession()
    
    # Begin nested transaction for test isolation
    nested = connection.begin_nested()
    
    # Patch session maker to use test transaction
    session.session = session.session.configure(
        bind=connection,
        join_transaction_mode="create_savepoint"
    )
    
    yield session.session
    
    # Rollback test transaction
    if nested.is_active:
        nested.rollback()
    transaction.rollback()
    connection.close()
    session.session.remove()

@pytest.fixture
def test_app(client, db_session):
    """Main test application fixture combining client and database."""
    app.dependency_overrides[get_session] = lambda: db_session
    return client

@pytest.fixture(autouse=True)
def test_metrics(request):
    """Automatic test metrics collection."""
    start_time = time.time()
    
    yield
    
    # Update test metrics
    duration = time.time() - start_time
    request.config.test_metrics['total_tests'] += 1
    
    if request.node.rep_call.passed:
        request.config.test_metrics['passed_tests'] += 1
    else:
        request.config.test_metrics['failed_tests'] += 1
    
    if duration > (TEST_PERFORMANCE_THRESHOLD / 1000):
        request.config.test_metrics['performance_violations'] += 1
        logging.warning(f"Test performance threshold exceeded: {duration*1000:.2f}ms")

@pytest.fixture(scope="session", autouse=True)
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()