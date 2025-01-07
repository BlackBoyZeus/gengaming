# External imports with versions
import pytest  # pytest ^7.3.1
import pytest_asyncio  # pytest-asyncio ^0.21.0
from unittest.mock import Mock, patch  # mock ^3.9.0
from datetime import datetime, UTC
import time
from uuid import uuid4

# Internal imports
from services.control import ControlService
from models.instructnet.control import ControlProcessor
from api.schemas.control import ControlBase
from db.models.control import Control

# Test constants
VALIDATION_TEST_CASES = [
    ({
        'type': 'keyboard',
        'data': {'key': 'W', 'action': 'press', 'timestamp': time.time()},
        'platform': 'FreeBSD'
    }, True),
    ({
        'type': 'environment',
        'data': {'parameter': 'weather', 'value': 'rain', 'timestamp': time.time()},
        'platform': 'FreeBSD'
    }, True),
    ({
        'type': 'instruction',
        'data': {'instruction': 'move_forward', 'parameters': {'speed': 1.0}, 'timestamp': time.time()},
        'platform': 'FreeBSD'
    }, True),
    ({
        'type': 'invalid',
        'data': {},
        'platform': 'FreeBSD'
    }, False)
]

PERFORMANCE_THRESHOLDS = {
    'response_time_ms': 50,  # <50ms requirement
    'success_rate_percent': 50,  # >50% requirement
    'max_memory_mb': 512
}

TEST_PLATFORM_INFO = {
    'os': 'FreeBSD',
    'version': '13.0',
    'gpu_type': 'non_nvidia'
}

@pytest.fixture
def mock_processor():
    processor = Mock(spec=ControlProcessor)
    processor.process_control.return_value = {
        'status': 'success',
        'latency_ms': 45,
        'processed_data': {'result': 'processed'}
    }
    processor.get_performance_metrics.return_value = {
        'avg_response_time_ms': 45,
        'success_rate': 0.75,
        'memory_usage_mb': 256
    }
    return processor

@pytest.mark.asyncio
@pytest.mark.parametrize('control_type', ['keyboard', 'environment', 'instruction'])
async def test_process_control_success(db_session, mock_processor, benchmark, control_type):
    """Test successful control signal processing with performance validation."""
    # Setup test data
    generation_id = uuid4()
    video_id = uuid4()
    timestamp = time.time()
    
    control_data = {
        'type': control_type,
        'data': {
            'key': 'W' if control_type == 'keyboard' else None,
            'parameter': 'weather' if control_type == 'environment' else None,
            'instruction': 'move_forward' if control_type == 'instruction' else None,
            'timestamp': timestamp
        },
        'platform_info': TEST_PLATFORM_INFO
    }
    
    # Initialize service
    service = ControlService(db_session, mock_processor)
    
    # Benchmark control processing
    def benchmark_fn():
        return service.process_control(generation_id, video_id, ControlBase(**control_data))
    
    result = await benchmark(benchmark_fn)
    
    # Validate response time
    assert result['latency_ms'] < PERFORMANCE_THRESHOLDS['response_time_ms'], \
        f"Response time {result['latency_ms']}ms exceeds threshold"
    
    # Verify success rate tracking
    metrics = mock_processor.get_performance_metrics()
    assert metrics['success_rate'] * 100 > PERFORMANCE_THRESHOLDS['success_rate_percent'], \
        f"Success rate {metrics['success_rate']*100}% below threshold"
    
    # Validate FreeBSD compatibility
    assert TEST_PLATFORM_INFO['os'] == 'FreeBSD', "Not running on FreeBSD"
    
    # Check database persistence
    control_record = await db_session.query(Control).filter_by(
        generation_id=generation_id,
        video_id=video_id,
        type=control_type
    ).first()
    assert control_record is not None
    assert control_record.data['processed'] == True

@pytest.mark.asyncio
@pytest.mark.parametrize('platform', ['FreeBSD', 'Linux'])
async def test_process_control_validation_error(db_session, mock_processor, platform):
    """Test control signal validation with platform-specific error handling."""
    # Setup invalid test data
    control_data = {
        'type': 'keyboard',
        'data': {
            'key': 'INVALID_KEY',
            'timestamp': time.time()
        },
        'platform_info': {'os': platform}
    }
    
    service = ControlService(db_session, mock_processor)
    
    # Test validation error
    with pytest.raises(ValueError) as exc_info:
        await service.process_control(
            uuid4(),
            uuid4(),
            ControlBase(**control_data)
        )
    
    assert "Invalid key" in str(exc_info.value)
    
    # Verify no database record created
    control_count = await db_session.query(Control).count()
    assert control_count == 0
    
    # Validate error metrics
    metrics = mock_processor.get_performance_metrics()
    assert 'validation_errors' in metrics

@pytest.mark.asyncio
async def test_control_performance_metrics(db_session, benchmark):
    """Test control performance metrics and success rate tracking."""
    # Setup test sequence
    processor = ControlProcessor(Mock())
    service = ControlService(db_session, processor)
    
    test_controls = []
    for i in range(100):
        test_controls.append({
            'type': 'keyboard',
            'data': {
                'key': 'W',
                'action': 'press',
                'timestamp': time.time() + i * 0.01
            },
            'platform_info': TEST_PLATFORM_INFO
        })
    
    # Benchmark control sequence
    async def process_sequence():
        for control in test_controls:
            await service.process_control(
                uuid4(),
                uuid4(),
                ControlBase(**control)
            )
    
    # Execute and measure performance
    start_time = time.perf_counter()
    await benchmark(process_sequence)
    total_time = time.perf_counter() - start_time
    
    # Calculate metrics
    avg_response_time = (total_time * 1000) / len(test_controls)
    assert avg_response_time < PERFORMANCE_THRESHOLDS['response_time_ms'], \
        f"Average response time {avg_response_time}ms exceeds threshold"
    
    # Verify success rate
    metrics = processor.get_performance_metrics()
    success_rate = metrics['success_rate'] * 100
    assert success_rate > PERFORMANCE_THRESHOLDS['success_rate_percent'], \
        f"Success rate {success_rate}% below threshold"
    
    # Generate performance report
    performance_data = {
        'avg_response_time_ms': avg_response_time,
        'success_rate_percent': success_rate,
        'total_controls': len(test_controls),
        'platform': TEST_PLATFORM_INFO['os'],
        'timestamp': datetime.now(UTC).isoformat()
    }
    
    # Log performance results
    print(f"\nPerformance Report: {performance_data}")