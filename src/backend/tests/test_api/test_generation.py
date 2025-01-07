"""
Comprehensive test suite for video generation API endpoints.
Tests text-to-video generation, performance metrics, security controls, and WebSocket streaming.

pytest version: ^7.3.1
pytest_asyncio version: ^0.21.0
"""

import json
import uuid
from datetime import datetime, UTC
import pytest
import pytest_asyncio
from fastapi import status
from api.schemas.generation import GenerationCreate, GenerationResponse
from db.models.generation import GenerationStatus

# Test constants based on technical specifications
TEST_PROMPT = "A scenic mountain landscape with snow-capped peaks"
TEST_PARAMETERS = {
    "resolution_width": 1280,
    "resolution_height": 720,
    "frame_count": 102,
    "perspective": "third_person"
}

# Quality metric thresholds from technical specs
QUALITY_THRESHOLDS = {
    "fid_max": 300,
    "fvd_max": 1000,
    "success_rate_min": 0.5
}

# Performance requirements
LATENCY_THRESHOLD_MS = 100
TARGET_FPS = 24

@pytest.fixture
def mock_gpu(mocker):
    """Mock GPU operations for testing."""
    mock = mocker.patch("services.gpu.GPUService")
    mock.return_value.generate_frames.return_value = {
        "status": "success",
        "metrics": {
            "fid": 250,
            "fvd": 850,
            "success_rate": 0.85,
            "generation_time_ms": 95
        }
    }
    return mock

@pytest.fixture
def test_generation():
    """Fixture providing test generation data."""
    return {
        "id": uuid.uuid4(),
        "prompt": TEST_PROMPT,
        "parameters": TEST_PARAMETERS,
        "status": GenerationStatus.PENDING,
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC)
    }

@pytest.mark.asyncio
async def test_generate_video(client, db_session, mock_gpu):
    """Test successful video generation with quality metrics validation."""
    # Create generation request
    request_data = {
        "prompt": TEST_PROMPT,
        "parameters": TEST_PARAMETERS
    }

    # Send generation request
    response = await client.post(
        "/api/v1/generation/",
        json=request_data
    )

    # Verify successful response
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Validate response schema
    generation_response = GenerationResponse(**data)
    assert generation_response.id is not None
    assert generation_response.status == GenerationStatus.PENDING
    assert generation_response.prompt == TEST_PROMPT

    # Verify generation starts processing
    process_response = await client.get(f"/api/v1/generation/{generation_response.id}")
    process_data = process_response.json()
    assert process_data["status"] in [GenerationStatus.PROCESSING, GenerationStatus.COMPLETED]

    # Validate quality metrics
    if "metrics" in process_data:
        metrics = process_data["metrics"]
        if "fid" in metrics:
            assert metrics["fid"] < QUALITY_THRESHOLDS["fid_max"]
        if "fvd" in metrics:
            assert metrics["fvd"] < QUALITY_THRESHOLDS["fvd_max"]
        if "success_rate" in metrics:
            assert metrics["success_rate"] >= QUALITY_THRESHOLDS["success_rate_min"]
        if "generation_time_ms" in metrics:
            assert metrics["generation_time_ms"] < LATENCY_THRESHOLD_MS

@pytest.mark.asyncio
async def test_generate_video_invalid_parameters(client):
    """Test validation errors for invalid generation parameters."""
    
    # Test invalid resolution
    invalid_resolution = {
        "prompt": TEST_PROMPT,
        "parameters": {
            **TEST_PARAMETERS,
            "resolution_width": 1920,  # Invalid width
            "resolution_height": 1080  # Invalid height
        }
    }
    response = await client.post("/api/v1/generation/", json=invalid_resolution)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "resolution" in response.json()["detail"][0]["msg"].lower()

    # Test invalid frame count
    invalid_frames = {
        "prompt": TEST_PROMPT,
        "parameters": {
            **TEST_PARAMETERS,
            "frame_count": 60  # Not 102 frames
        }
    }
    response = await client.post("/api/v1/generation/", json=invalid_frames)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "frame" in response.json()["detail"][0]["msg"].lower()

    # Test missing prompt
    missing_prompt = {
        "parameters": TEST_PARAMETERS
    }
    response = await client.post("/api/v1/generation/", json=missing_prompt)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "prompt" in response.json()["detail"][0]["msg"].lower()

    # Test invalid perspective
    invalid_perspective = {
        "prompt": TEST_PROMPT,
        "parameters": {
            **TEST_PARAMETERS,
            "perspective": "invalid"
        }
    }
    response = await client.post("/api/v1/generation/", json=invalid_perspective)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "perspective" in response.json()["detail"][0]["msg"].lower()

@pytest.mark.asyncio
async def test_stream_frames(async_client, db_session, test_generation, mock_gpu):
    """Test WebSocket streaming with performance validation."""
    generation_id = test_generation["id"]

    # Connect to WebSocket
    async with async_client.websocket_connect(f"/api/v1/generation/stream/{generation_id}") as websocket:
        # Verify connection
        connection_msg = await websocket.receive_json()
        assert connection_msg["status"] == "connected"

        # Track frame timing for FPS validation
        frame_times = []
        frame_count = 0
        
        # Receive frames
        try:
            while frame_count < TEST_PARAMETERS["frame_count"]:
                frame = await websocket.receive_bytes()
                frame_times.append(datetime.now(UTC))
                frame_count += 1

                # Validate frame format
                assert len(frame) > 0  # Frame data present
                
                # Calculate current FPS every 24 frames
                if len(frame_times) >= 24:
                    time_diff = (frame_times[-1] - frame_times[-24]).total_seconds()
                    current_fps = 24 / time_diff
                    assert current_fps >= TARGET_FPS * 0.9  # Allow 10% tolerance

        except Exception as e:
            pytest.fail(f"Frame streaming failed: {str(e)}")

        # Verify total frames received
        assert frame_count == TEST_PARAMETERS["frame_count"]

        # Verify graceful closure
        await websocket.close()
        assert websocket.closed