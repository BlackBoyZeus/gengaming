# External imports with versions
from fastapi import WebSocket, WebSocketDisconnect  # fastapi ^0.95.0
from prometheus_client import Counter, Histogram  # prometheus_client ^0.16.0
import asyncio  # asyncio ^3.9.0
import numpy as np  # numpy ^1.23.0
from typing import Dict, Any, Optional
from uuid import UUID
import time
import logging

# Internal imports
from api.dependencies import get_current_user, get_cache
from services.video import VideoService
from services.control import ControlService
from core.metrics import track_generation_latency, track_jail_metrics
from core.exceptions import GameGenBaseException

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize metrics
WEBSOCKET_CONNECTIONS = Counter(
    'websocket_connections_total',
    'Total WebSocket connections',
    ['status']
)
WEBSOCKET_LATENCY = Histogram(
    'websocket_latency_ms',
    'WebSocket message latency in milliseconds',
    buckets=[10, 25, 50, 75, 100]
)
FRAME_DELIVERY_LATENCY = Histogram(
    'frame_delivery_latency_ms',
    'Frame delivery latency in milliseconds',
    buckets=[5, 10, 25, 50, 75]
)

class WebSocketManager:
    """Manages WebSocket connections with performance monitoring and resource management."""
    
    def __init__(self, video_service: VideoService, control_service: ControlService):
        """Initialize WebSocket manager with services and monitoring."""
        self._active_connections: Dict[str, WebSocket] = {}
        self._connection_metrics: Dict[str, Dict[str, float]] = {}
        self._video_service = video_service
        self._control_service = control_service
        self._frame_buffer_size = 24  # 1 second buffer at 24 FPS
        
        # Initialize FreeBSD jail monitoring
        track_jail_metrics("gamegen-websocket")

    async def connect(self, websocket: WebSocket, generation_id: UUID) -> None:
        """Handle new WebSocket connection with resource management."""
        try:
            await websocket.accept()
            connection_id = str(generation_id)
            self._active_connections[connection_id] = websocket
            self._connection_metrics[connection_id] = {
                "connected_at": time.time(),
                "frames_sent": 0,
                "controls_processed": 0
            }
            WEBSOCKET_CONNECTIONS.labels(status="connected").inc()
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            WEBSOCKET_CONNECTIONS.labels(status="failed").inc()
            raise

    async def disconnect(self, generation_id: UUID) -> None:
        """Handle WebSocket disconnection with cleanup."""
        try:
            connection_id = str(generation_id)
            if connection_id in self._active_connections:
                await self._active_connections[connection_id].close()
                del self._active_connections[connection_id]
                del self._connection_metrics[connection_id]
                WEBSOCKET_CONNECTIONS.labels(status="disconnected").inc()
                
        except Exception as e:
            logger.error(f"WebSocket disconnection error: {str(e)}")

    async def broadcast_frame(self, generation_id: UUID, frame: np.ndarray) -> None:
        """Broadcast video frame with latency monitoring."""
        try:
            start_time = time.perf_counter()
            connection_id = str(generation_id)
            
            if connection_id in self._active_connections:
                websocket = self._active_connections[connection_id]
                # Optimize frame for transmission
                frame_bytes = frame.tobytes()
                await websocket.send_bytes(frame_bytes)
                
                # Update metrics
                self._connection_metrics[connection_id]["frames_sent"] += 1
                latency = (time.perf_counter() - start_time) * 1000
                FRAME_DELIVERY_LATENCY.observe(latency)
                
        except Exception as e:
            logger.error(f"Frame broadcast failed: {str(e)}")
            await self.disconnect(generation_id)

websocket_manager = WebSocketManager(VideoService(), ControlService())

async def stream_endpoint(
    websocket: WebSocket,
    generation_id: UUID,
    current_user = get_current_user,
    cache = get_cache
):
    """WebSocket endpoint for real-time video streaming and control."""
    try:
        # Initialize connection
        await websocket_manager.connect(websocket, generation_id)
        
        # Start frame streaming task
        streaming_task = asyncio.create_task(
            stream_frames(websocket, generation_id, websocket_manager._video_service)
        )
        
        # Handle control messages
        while True:
            try:
                # Receive control message with timeout
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=1.0
                )
                
                start_time = time.perf_counter()
                
                # Process control message
                result = await handle_control_message(
                    message,
                    generation_id,
                    websocket_manager._control_service,
                    cache
                )
                
                # Send control response
                await websocket.send_json(result)
                
                # Track latency
                latency = (time.perf_counter() - start_time) * 1000
                WEBSOCKET_LATENCY.observe(latency)
                
            except asyncio.TimeoutError:
                # Check connection health
                if not websocket.client_state.connected:
                    break
                continue
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {generation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Cleanup
        await websocket_manager.disconnect(generation_id)
        streaming_task.cancel()

@track_generation_latency
async def handle_control_message(
    message: Dict[str, Any],
    generation_id: UUID,
    control_service: ControlService,
    cache
) -> Dict[str, Any]:
    """Process control messages with validation and rate limiting."""
    try:
        # Validate message format
        if not all(k in message for k in ["type", "data"]):
            raise ValueError("Invalid control message format")
            
        # Process control through service
        result = await control_service.process_control(
            generation_id=generation_id,
            video_id=message.get("video_id"),
            control_data=message["data"]
        )
        
        return {
            "status": "success",
            "control_id": result.get("control_id"),
            "latency_ms": result.get("latency_ms")
        }
        
    except Exception as e:
        logger.error(f"Control processing failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

async def stream_frames(
    websocket: WebSocket,
    generation_id: UUID,
    video_service: VideoService
) -> None:
    """Stream video frames with performance optimization and error recovery."""
    try:
        frame_interval = 1.0 / 24  # 24 FPS
        last_frame_time = time.perf_counter()
        
        async for frame in video_service.stream_video(str(generation_id)):
            current_time = time.perf_counter()
            elapsed = current_time - last_frame_time
            
            # Maintain frame rate
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)
            
            # Send frame
            await websocket_manager.broadcast_frame(generation_id, frame)
            last_frame_time = time.perf_counter()
            
    except Exception as e:
        logger.error(f"Frame streaming failed: {str(e)}")
        raise