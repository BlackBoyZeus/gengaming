# External imports with versions
import pytest  # pytest ^7.3.1
import asyncio  # asyncio ^3.9.0
import numpy as np  # numpy ^1.23.0
import psutil  # psutil ^5.9.0
import time
from uuid import uuid4
from typing import Dict, Any, Optional

# Internal imports
from api.routes.websocket import WebSocketManager
from core.metrics import track_generation_latency
from core.exceptions import VideoGenerationError

class WebSocketTestCase:
    """Base test class for WebSocket functionality with FreeBSD support."""

    def __init__(self):
        """Initialize WebSocket test case with FreeBSD configuration."""
        self.base_url = "ws://test"
        self.test_data = {
            "generation_id": str(uuid4()),
            "video_id": str(uuid4()),
            "frame_size": (1280, 720),
            "frame_count": 102,
            "fps": 24
        }
        self.performance_metrics = {
            "latency_ms": [],
            "frame_rate": [],
            "memory_usage": []
        }

    async def setup_method(self, method):
        """Set up test method with resource monitoring."""
        # Reset test state
        self.performance_metrics = {
            "latency_ms": [],
            "frame_rate": [],
            "memory_usage": []
        }
        
        # Initialize test data
        self.test_frames = np.random.randint(
            0, 255, 
            (self.test_data["frame_count"], 720, 1280, 3), 
            dtype=np.uint8
        )
        
        # Configure FreeBSD test environment
        self.initial_memory = psutil.Process().memory_info().rss
        self.start_time = time.time()

    async def teardown_method(self, method):
        """Clean up after test method with resource verification."""
        # Verify resource cleanup
        final_memory = psutil.Process().memory_info().rss
        memory_diff = final_memory - self.initial_memory
        
        # Log performance metrics
        avg_latency = np.mean(self.performance_metrics["latency_ms"])
        avg_fps = np.mean(self.performance_metrics["frame_rate"])
        
        assert avg_latency < 100, f"Average latency {avg_latency}ms exceeds 100ms threshold"
        assert avg_fps >= 24, f"Average frame rate {avg_fps} FPS below 24 FPS target"
        assert memory_diff < 100 * 1024 * 1024, f"Memory leak detected: {memory_diff / 1024 / 1024:.2f}MB"

@pytest.mark.asyncio
@pytest.mark.websocket
@pytest.mark.freebsd
class TestWebSocketConnection(WebSocketTestCase):
    """Test WebSocket connection establishment and authentication."""

    async def test_connection_establishment(self, async_client):
        """Test WebSocket connection with FreeBSD compatibility."""
        # Configure test client
        client = async_client
        generation_id = self.test_data["generation_id"]
        
        try:
            # Connect to WebSocket
            async with client.websocket_connect(
                f"{self.base_url}/ws/stream/{generation_id}"
            ) as websocket:
                # Verify connection
                assert websocket.client_state.connected
                
                # Test connection limits
                concurrent_connections = 100
                connections = []
                for _ in range(concurrent_connections):
                    conn = await client.websocket_connect(
                        f"{self.base_url}/ws/stream/{uuid4()}"
                    )
                    connections.append(conn)
                
                # Verify resource usage
                memory_usage = psutil.Process().memory_info().rss
                self.performance_metrics["memory_usage"].append(memory_usage)
                
                # Cleanup connections
                for conn in connections:
                    await conn.close()
                
        except Exception as e:
            pytest.fail(f"WebSocket connection failed: {str(e)}")

@pytest.mark.asyncio
@pytest.mark.websocket
@pytest.mark.performance
class TestVideoStreaming(WebSocketTestCase):
    """Test video frame streaming with performance validation."""

    async def test_frame_streaming(self, async_client):
        """Test video streaming with frame rate and latency monitoring."""
        client = async_client
        generation_id = self.test_data["generation_id"]
        
        try:
            async with client.websocket_connect(
                f"{self.base_url}/ws/stream/{generation_id}"
            ) as websocket:
                frame_count = 0
                start_time = time.time()
                
                # Stream test frames
                for frame in self.test_frames:
                    frame_start = time.time()
                    
                    # Send frame
                    await websocket.send_bytes(frame.tobytes())
                    
                    # Receive frame acknowledgment
                    response = await websocket.receive_json()
                    assert response["status"] == "success"
                    
                    # Track metrics
                    frame_count += 1
                    latency = (time.time() - frame_start) * 1000
                    self.performance_metrics["latency_ms"].append(latency)
                    
                    if frame_count % 24 == 0:  # Calculate FPS every second
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        self.performance_metrics["frame_rate"].append(fps)
                        
                        # Verify performance
                        assert latency < 100, f"Frame latency {latency}ms exceeds 100ms threshold"
                        assert fps >= 24, f"Frame rate {fps} FPS below 24 FPS target"
                
        except Exception as e:
            pytest.fail(f"Frame streaming failed: {str(e)}")

@pytest.mark.asyncio
@pytest.mark.websocket
@pytest.mark.control
class TestControlMessages(WebSocketTestCase):
    """Test control message processing with response time validation."""

    async def test_control_processing(self, async_client):
        """Test control message handling with latency monitoring."""
        client = async_client
        generation_id = self.test_data["generation_id"]
        
        try:
            async with client.websocket_connect(
                f"{self.base_url}/ws/stream/{generation_id}"
            ) as websocket:
                # Test different control types
                control_messages = [
                    {
                        "type": "keyboard",
                        "data": {"key": "W", "action": "press", "timestamp": time.time()}
                    },
                    {
                        "type": "environment",
                        "data": {"parameter": "weather", "value": "rain", "timestamp": time.time()}
                    },
                    {
                        "type": "instruction",
                        "data": {"instruction": "move_forward", "parameters": {}, "timestamp": time.time()}
                    }
                ]
                
                for control in control_messages:
                    start_time = time.time()
                    
                    # Send control message
                    await websocket.send_json(control)
                    
                    # Receive response
                    response = await websocket.receive_json()
                    
                    # Verify response and timing
                    assert response["status"] == "success"
                    latency = (time.time() - start_time) * 1000
                    self.performance_metrics["latency_ms"].append(latency)
                    
                    # Verify control latency
                    assert latency < 50, f"Control latency {latency}ms exceeds 50ms threshold"
                    
                    # Allow for state update
                    await asyncio.sleep(0.1)
                
        except Exception as e:
            pytest.fail(f"Control processing failed: {str(e)}")