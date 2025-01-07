# External imports with versions
import pytest  # pytest ^7.3.1
import uuid
import time
import asyncio
from typing import Dict, Any

# Internal imports
from tests.conftest import client, db_session, async_client
from api.schemas.control import ControlBase, ControlCreate, ControlResponse

# Test constants
TEST_GENERATION_ID = str(uuid.uuid4())
TEST_VIDEO_ID = str(uuid.uuid4())
PERFORMANCE_THRESHOLD_MS = 50  # 50ms response time requirement
SUCCESS_RATE_THRESHOLD = 0.5  # 50% success rate requirement

@pytest.fixture
def test_control_data():
    """Fixture providing test control data with comprehensive validation."""
    return {
        "keyboard_control": {
            "type": "keyboard",
            "data": {
                "key": "W",
                "action": "press",
                "timestamp": time.time()
            }
        },
        "environment_control": {
            "type": "environment",
            "data": {
                "parameter": "weather",
                "value": "clear",
                "timestamp": time.time()
            }
        },
        "instruction_control": {
            "type": "instruction",
            "data": {
                "instruction": "move_forward",
                "parameters": {"speed": 1.0},
                "timestamp": time.time()
            }
        }
    }

@pytest.mark.asyncio
async def test_create_control(client, db_session, test_control_data):
    """Tests control signal creation with performance validation."""
    
    # Test keyboard control
    start_time = time.perf_counter()
    response = client.post(
        "/api/v1/control/",
        json={
            **test_control_data["keyboard_control"],
            "generation_id": TEST_GENERATION_ID,
            "video_id": TEST_VIDEO_ID
        }
    )
    response_time = (time.perf_counter() - start_time) * 1000
    
    # Validate response
    assert response.status_code == 200
    assert "id" in response.json()
    assert response.json()["type"] == "keyboard"
    assert "latency_ms" in response.json()["metadata"]
    
    # Validate performance requirement (<50ms)
    assert response_time < PERFORMANCE_THRESHOLD_MS, \
        f"Control response time {response_time}ms exceeds {PERFORMANCE_THRESHOLD_MS}ms threshold"

    # Test environment control
    start_time = time.perf_counter()
    response = client.post(
        "/api/v1/control/",
        json={
            **test_control_data["environment_control"],
            "generation_id": TEST_GENERATION_ID,
            "video_id": TEST_VIDEO_ID
        }
    )
    response_time = (time.perf_counter() - start_time) * 1000
    
    assert response.status_code == 200
    assert response_time < PERFORMANCE_THRESHOLD_MS

    # Test instruction control
    start_time = time.perf_counter()
    response = client.post(
        "/api/v1/control/",
        json={
            **test_control_data["instruction_control"],
            "generation_id": TEST_GENERATION_ID,
            "video_id": TEST_VIDEO_ID
        }
    )
    response_time = (time.perf_counter() - start_time) * 1000
    
    assert response.status_code == 200
    assert response_time < PERFORMANCE_THRESHOLD_MS

@pytest.mark.asyncio
async def test_concurrent_controls(async_client, db_session):
    """Tests control endpoint behavior under concurrent load."""
    
    async def send_control(control_type: str) -> float:
        start_time = time.perf_counter()
        response = await async_client.post(
            "/api/v1/control/",
            json={
                "type": control_type,
                "data": {
                    "key": "W" if control_type == "keyboard" else None,
                    "action": "press" if control_type == "keyboard" else None,
                    "parameter": "weather" if control_type == "environment" else None,
                    "value": "clear" if control_type == "environment" else None,
                    "timestamp": time.time()
                },
                "generation_id": TEST_GENERATION_ID,
                "video_id": TEST_VIDEO_ID
            }
        )
        response_time = (time.perf_counter() - start_time) * 1000
        return response.status_code == 200 and response_time < PERFORMANCE_THRESHOLD_MS

    # Test concurrent keyboard controls
    keyboard_tasks = [
        send_control("keyboard") for _ in range(10)
    ]
    keyboard_results = await asyncio.gather(*keyboard_tasks)
    keyboard_success_rate = sum(keyboard_results) / len(keyboard_results)
    
    assert keyboard_success_rate >= SUCCESS_RATE_THRESHOLD, \
        f"Keyboard control success rate {keyboard_success_rate} below {SUCCESS_RATE_THRESHOLD} threshold"

    # Test concurrent environment controls
    environment_tasks = [
        send_control("environment") for _ in range(5)
    ]
    environment_results = await asyncio.gather(*environment_tasks)
    environment_success_rate = sum(environment_results) / len(environment_results)
    
    assert environment_success_rate >= SUCCESS_RATE_THRESHOLD

@pytest.mark.asyncio
async def test_control_error_handling(client, db_session):
    """Tests error handling scenarios for control endpoints."""
    
    # Test invalid control type
    response = client.post(
        "/api/v1/control/",
        json={
            "type": "invalid_type",
            "data": {"key": "X"},
            "generation_id": TEST_GENERATION_ID,
            "video_id": TEST_VIDEO_ID
        }
    )
    assert response.status_code == 400
    
    # Test missing required fields
    response = client.post(
        "/api/v1/control/",
        json={
            "type": "keyboard",
            "generation_id": TEST_GENERATION_ID,
            "video_id": TEST_VIDEO_ID
        }
    )
    assert response.status_code == 422
    
    # Test invalid data format
    response = client.post(
        "/api/v1/control/",
        json={
            "type": "keyboard",
            "data": "invalid_data",
            "generation_id": TEST_GENERATION_ID,
            "video_id": TEST_VIDEO_ID
        }
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_control_retrieval(client, db_session, test_control_data):
    """Tests control history retrieval with pagination."""
    
    # Create test controls
    for control_type in ["keyboard", "environment", "instruction"]:
        client.post(
            "/api/v1/control/",
            json={
                **test_control_data[f"{control_type}_control"],
                "generation_id": TEST_GENERATION_ID,
                "video_id": TEST_VIDEO_ID
            }
        )
    
    # Test retrieval with pagination
    response = client.get(f"/api/v1/control/{TEST_GENERATION_ID}?page_size=2&page_number=0")
    assert response.status_code == 200
    assert len(response.json()) == 2
    
    # Validate control data format
    for control in response.json():
        assert "id" in control
        assert "type" in control
        assert "data" in control
        assert "metadata" in control

@pytest.mark.asyncio
async def test_control_cleanup(client, db_session, test_control_data):
    """Tests control cleanup functionality."""
    
    # Create test controls
    for control_type in ["keyboard", "environment"]:
        client.post(
            "/api/v1/control/",
            json={
                **test_control_data[f"{control_type}_control"],
                "generation_id": TEST_GENERATION_ID,
                "video_id": TEST_VIDEO_ID
            }
        )
    
    # Test cleanup
    response = client.delete(f"/api/v1/control/{TEST_GENERATION_ID}")
    assert response.status_code == 200
    assert "cleared_count" in response.json()
    assert response.json()["cleared_count"] > 0
    
    # Verify cleanup
    response = client.get(f"/api/v1/control/{TEST_GENERATION_ID}")
    assert response.status_code == 200
    assert len(response.json()) == 0