# External imports with versions
import pytest  # pytest ^7.3.1
from prometheus_client import Counter, Histogram  # prometheus_client ^0.16.0
from typing import Dict, Any, Optional
import logging
import time
import os

# Internal imports
from core.config import Settings
from core.metrics import track_generation_latency
from core.exceptions import GameGenBaseException

# Global test constants
TEST_API_PREFIX = '/api/v1'
TEST_CONTROL_ENDPOINT = '/control'
TEST_GENERATION_ENDPOINT = '/generation'
TEST_STATUS_ENDPOINT = '/status'

# Security headers for test requests
TEST_SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block'
}

# Performance thresholds for test execution
TEST_PERFORMANCE_THRESHOLDS = {
    'response_time_ms': 100,  # 100ms max response time
    'memory_mb': 512  # 512MB max memory usage
}

# Test metrics collectors
TEST_EXECUTION_TIME = Histogram(
    'test_execution_time_seconds',
    'Test execution time in seconds',
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

TEST_MEMORY_USAGE = Histogram(
    'test_memory_usage_bytes',
    'Test memory usage in bytes',
    buckets=[1e6, 1e7, 1e8, 1e9]  # 1MB to 1GB
)

TEST_ERRORS = Counter(
    'test_errors_total',
    'Total number of test errors',
    ['error_type']
)

def pytest_configure(config):
    """
    Configures pytest environment for API tests with security, performance monitoring,
    and FreeBSD compatibility.
    """
    try:
        # Initialize test settings
        settings = Settings()
        
        # Configure test logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join('/var/log/gamegen-x', 'test.log'))
            ]
        )
        
        # Configure test metrics
        config.test_metrics = {
            'start_time': time.time(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'performance_violations': 0
        }
        
        # Configure FreeBSD-specific test settings
        config.freebsd_settings = {
            'test_jail': {
                'path': '/usr/jail/test',
                'allow_mount': True,
                'allow_raw_sockets': True
            },
            'resource_limits': {
                'openfiles': 1024,
                'memorylocked': TEST_PERFORMANCE_THRESHOLDS['memory_mb'] * 1024 * 1024
            }
        }
        
        # Configure test security
        config.security_settings = {
            'headers': TEST_SECURITY_HEADERS,
            'token_validation': True,
            'rate_limiting': True
        }
        
        # Configure test database
        config.test_db = {
            'isolation_level': 'REPEATABLE READ',
            'pool_size': 5,
            'max_overflow': 10
        }
        
        # Register test result processor
        config.pluginmanager.register(TestResultProcessor())
        
    except Exception as e:
        logging.error(f"Test configuration failed: {str(e)}")
        raise

@pytest.hookimpl(tryfirst=True)
def collect_test_metrics(test_result):
    """
    Collects and reports test execution metrics.
    
    Args:
        test_result: Test execution result object
    """
    try:
        # Calculate execution time
        execution_time = time.time() - test_result.start_time
        TEST_EXECUTION_TIME.observe(execution_time)
        
        # Track memory usage
        memory_info = test_result.memory_info
        if memory_info:
            TEST_MEMORY_USAGE.observe(memory_info.rss)
            
            # Check memory threshold
            if memory_info.rss > TEST_PERFORMANCE_THRESHOLDS['memory_mb'] * 1024 * 1024:
                logging.warning(f"Memory usage exceeded threshold: {memory_info.rss / (1024*1024):.2f}MB")
                test_result.config.test_metrics['performance_violations'] += 1
        
        # Track test status
        if test_result.passed:
            test_result.config.test_metrics['passed_tests'] += 1
        elif test_result.failed:
            test_result.config.test_metrics['failed_tests'] += 1
            TEST_ERRORS.labels(error_type=test_result.error_type).inc()
        elif test_result.skipped:
            test_result.config.test_metrics['skipped_tests'] += 1
            
        # Log test completion
        logging.info(
            "Test completed",
            extra={
                'test_id': test_result.nodeid,
                'execution_time': execution_time,
                'status': 'passed' if test_result.passed else 'failed',
                'memory_usage': memory_info.rss if memory_info else None
            }
        )
        
    except Exception as e:
        logging.error(f"Failed to collect test metrics: {str(e)}")
        raise

class TestResultProcessor:
    """Processes test results and maintains test execution state."""
    
    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item):
        """Initializes test execution context."""
        item.start_time = time.time()
        item.error_type = None
        
    @pytest.hookimpl(trylast=True)
    def pytest_runtest_makereport(self, item, call):
        """Processes test execution results."""
        if call.excinfo:
            item.error_type = call.excinfo.type.__name__
            
    @pytest.hookimpl
    def pytest_sessionfinish(self, session):
        """Finalizes test execution and reports metrics."""
        metrics = session.config.test_metrics
        total_time = time.time() - metrics['start_time']
        
        logging.info(
            "Test session completed",
            extra={
                'total_tests': metrics['total_tests'],
                'passed_tests': metrics['passed_tests'],
                'failed_tests': metrics['failed_tests'],
                'skipped_tests': metrics['skipped_tests'],
                'performance_violations': metrics['performance_violations'],
                'total_time': total_time
            }
        )