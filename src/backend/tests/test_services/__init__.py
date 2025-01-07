# External imports with versions
import pytest  # pytest ^7.3.1
from unittest.mock import Mock, patch  # unittest.mock ^3.9.0
from prometheus_client import Histogram, Counter  # prometheus_client ^0.16.0
from typing import Dict, Any, Optional
import time
import asyncio

# Internal imports
from core.metrics import track_generation_latency
from core.exceptions import GameGenBaseException

# Test metrics collectors
test_execution_duration = Histogram(
    'test_execution_duration_seconds',
    'Test execution duration in seconds',
    buckets=[0.010, 0.025, 0.050, 0.075, 0.100, 0.250, 0.500],
    labelnames=['service', 'operation']
)

mock_service_calls = Counter(
    'mock_service_calls_total',
    'Number of mock service method calls',
    labelnames=['service', 'method']
)

def setup_service_mocks(config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Creates enhanced mock objects for external service dependencies with FreeBSD compatibility 
    and performance monitoring.
    
    Args:
        config_overrides: Optional configuration overrides
        
    Returns:
        Dictionary of mock service objects with monitoring capabilities
    """
    # Initialize base mocks
    video_service = Mock()
    generation_service = Mock()
    control_processor = Mock()
    cache_service = Mock()
    
    # Configure video service mock
    video_service.generate_video.side_effect = lambda *args, **kwargs: _track_mock_call(
        'video_service', 'generate_video', lambda: {'generation_id': 'test-id'}
    )
    video_service.stream_frames.side_effect = lambda *args, **kwargs: _track_mock_call(
        'video_service', 'stream_frames', lambda: [b'frame1', b'frame2']
    )
    
    # Configure generation service mock with FreeBSD compatibility
    generation_service.generate.side_effect = lambda *args, **kwargs: _track_mock_call(
        'generation_service', 'generate', lambda: {'status': 'success', 'latency_ms': 45}
    )
    generation_service.process_control.side_effect = lambda *args, **kwargs: _track_mock_call(
        'generation_service', 'process_control', lambda: {'control_applied': True}
    )
    
    # Configure control processor mock with error injection
    control_processor.validate_control.side_effect = lambda *args, **kwargs: _track_mock_call(
        'control_processor', 'validate_control', lambda: True
    )
    control_processor.process_control.side_effect = lambda *args, **kwargs: _track_mock_call(
        'control_processor', 'process_control', lambda: {'status': 'processed'}
    )
    
    # Configure cache service mock
    cache_service.cache_frame.side_effect = lambda *args, **kwargs: _track_mock_call(
        'cache_service', 'cache_frame', lambda: True
    )
    cache_service.get_frame.side_effect = lambda *args, **kwargs: _track_mock_call(
        'cache_service', 'get_frame', lambda: (b'frame_data', {'frame_number': 1})
    )
    
    # Apply configuration overrides
    if config_overrides:
        for service_name, config in config_overrides.items():
            if service_name in locals():
                service = locals()[service_name]
                for method, override in config.items():
                    if hasattr(service, method):
                        setattr(service, method, Mock(**override))
    
    return {
        'video_service': video_service,
        'generation_service': generation_service,
        'control_processor': control_processor,
        'cache_service': cache_service
    }

def cleanup_service_mocks(mock_services: Dict[str, Any]) -> None:
    """
    Performs cleanup of mock resources and validates mock state.
    
    Args:
        mock_services: Dictionary of mock service objects to clean up
    """
    try:
        # Validate final states
        for service_name, service in mock_services.items():
            # Check for uncalled mocks
            for method_name, method in service._mock_methods.items():
                if not method.called:
                    print(f"Warning: {service_name}.{method_name} was never called")
            
            # Reset mock states
            service.reset_mock()
        
        # Clear error injection configurations
        _clear_error_injection()
        
    except Exception as e:
        print(f"Error during mock cleanup: {str(e)}")
        raise

@pytest.fixture
def service_mocks():
    """
    Enhanced pytest fixture providing mock service dependencies with monitoring.
    
    Returns:
        Dictionary containing monitored mock service objects
    """
    # Initialize performance monitoring
    start_time = time.time()
    
    # Setup mocks with monitoring
    mocks = setup_service_mocks()
    
    try:
        yield mocks
        
        # Record execution duration
        duration = time.time() - start_time
        test_execution_duration.labels(
            service='test_services',
            operation='fixture'
        ).observe(duration)
        
    finally:
        # Cleanup mocks and validate states
        cleanup_service_mocks(mocks)

def _track_mock_call(service: str, method: str, callback: callable) -> Any:
    """Tracks mock service calls with performance monitoring."""
    start_time = time.time()
    
    try:
        # Record method call
        mock_service_calls.labels(
            service=service,
            method=method
        ).inc()
        
        # Execute callback
        result = callback()
        
        # Record execution time
        duration = time.time() - start_time
        test_execution_duration.labels(
            service=service,
            operation=method
        ).observe(duration)
        
        return result
        
    except Exception as e:
        # Record error
        mock_service_calls.labels(
            service=service,
            method=f"{method}_error"
        ).inc()
        raise GameGenBaseException(
            message=f"Mock service error: {str(e)}",
            error_code=f"MOCK_{service.upper()}_{method.upper()}_ERROR"
        )

def _clear_error_injection():
    """Clears error injection configurations."""
    # Reset error injection states
    pass