# External imports with versions
import pytest  # pytest ^7.3.1
import psutil  # psutil ^5.9.0
import elasticsearch  # elasticsearch ^8.0.0
from typing import Dict, Any, Optional
import time
import logging
import os
from pathlib import Path

# Internal imports
from core.config import Settings
from tests.conftest import test_client, async_client, db_session

# Global constants
TEST_ENVIRONMENT = "test"
PYTEST_TIMEOUT = 600  # 10 minutes timeout for tests
MEMORY_THRESHOLD_MB = 1024  # 1GB memory threshold
LOG_INDEX = "gamegen_test_logs"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pytest_configure(config):
    """
    PyTest configuration hook for setting up test environment with enhanced monitoring 
    and FreeBSD compatibility.
    """
    try:
        # Set test environment
        os.environ["ENVIRONMENT"] = TEST_ENVIRONMENT
        Settings().environment = TEST_ENVIRONMENT

        # Configure test timeouts
        config.addinivalue_line("timeout", str(PYTEST_TIMEOUT))

        # Initialize test metrics tracking
        config.test_metrics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'performance_violations': 0,
            'memory_usage': [],
            'response_times': [],
            'start_time': time.time()
        }

        # Configure FreeBSD-specific test settings
        if os.path.exists("/usr/sbin/sysctl"):
            _configure_freebsd_test_env()

        # Initialize ELK logging for test results
        _setup_elk_logging(config)

        # Configure memory monitoring
        _setup_memory_monitoring(config)

        # Register custom markers
        config.addinivalue_line(
            "markers", 
            "performance: mark test for performance monitoring"
        )
        config.addinivalue_line(
            "markers",
            "freebsd: mark test for FreeBSD compatibility"
        )

    except Exception as e:
        logger.error(f"Test configuration failed: {str(e)}")
        raise

def pytest_runtest_protocol(item, nextitem):
    """
    PyTest hook for test execution monitoring with performance tracking.
    """
    try:
        # Start performance monitoring
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        # Track test execution
        result = True
        try:
            # Execute test with monitoring
            result = item.ihook.pytest_runtest_protocol(item=item, nextitem=nextitem)

            # Calculate metrics
            execution_time = time.time() - start_time
            memory_used = psutil.Process().memory_info().rss - start_memory

            # Update test metrics
            item.config.test_metrics['response_times'].append(execution_time)
            item.config.test_metrics['memory_usage'].append(memory_used)

            # Check performance thresholds
            if execution_time > 0.1:  # 100ms threshold
                item.config.test_metrics['performance_violations'] += 1
                logger.warning(
                    f"Performance threshold exceeded: {execution_time*1000:.2f}ms"
                )

            # Check memory thresholds
            if memory_used > MEMORY_THRESHOLD_MB * 1024 * 1024:
                logger.warning(
                    f"Memory threshold exceeded: {memory_used/(1024*1024):.2f}MB"
                )

            return result

        finally:
            # Cleanup test resources
            _cleanup_test_resources()

    except Exception as e:
        logger.error(f"Test protocol error: {str(e)}")
        raise

def _configure_freebsd_test_env():
    """Configure FreeBSD-specific test environment settings."""
    try:
        # Set FreeBSD-specific system parameters
        os.system("sysctl kern.ipc.shm_allow_removed=1")
        os.system("sysctl kern.ipc.somaxconn=2048")
        os.system("sysctl kern.maxfiles=65535")

    except Exception as e:
        logger.warning(f"FreeBSD configuration failed: {str(e)}")

def _setup_elk_logging(config):
    """Initialize ELK stack logging for test results."""
    try:
        # Configure Elasticsearch client
        es_client = elasticsearch.Elasticsearch(
            ["http://localhost:9200"],
            basic_auth=("elastic", "changeme")
        )

        # Create test log index
        if not es_client.indices.exists(index=LOG_INDEX):
            es_client.indices.create(
                index=LOG_INDEX,
                mappings={
                    "properties": {
                        "timestamp": {"type": "date"},
                        "test_name": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "execution_time": {"type": "float"},
                        "memory_used": {"type": "long"}
                    }
                }
            )

        config.elk_client = es_client

    except Exception as e:
        logger.warning(f"ELK logging setup failed: {str(e)}")

def _setup_memory_monitoring(config):
    """Initialize memory usage monitoring for tests."""
    try:
        # Configure memory tracking
        config.memory_tracker = {
            'peak_usage': 0,
            'total_allocated': 0,
            'collection_enabled': True
        }

        # Set up periodic memory collection
        def collect_memory_stats():
            if config.memory_tracker['collection_enabled']:
                current_memory = psutil.Process().memory_info().rss
                config.memory_tracker['peak_usage'] = max(
                    config.memory_tracker['peak_usage'],
                    current_memory
                )
                config.memory_tracker['total_allocated'] += current_memory

        config.memory_collector = collect_memory_stats

    except Exception as e:
        logger.warning(f"Memory monitoring setup failed: {str(e)}")

def _cleanup_test_resources():
    """Clean up test resources and temporary files."""
    try:
        # Clear test cache directory
        cache_dir = Path("tests/.pytest_cache")
        if cache_dir.exists():
            for file in cache_dir.glob("*"):
                if file.is_file():
                    file.unlink()

        # Clear memory
        if hasattr(pytest, "gc"):
            pytest.gc.collect()

    except Exception as e:
        logger.warning(f"Resource cleanup failed: {str(e)}")