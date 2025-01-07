# External imports with versions
from fastapi import APIRouter, BackgroundTasks, WebSocket, Depends, HTTPException  # fastapi ^0.95.0
from prometheus_client import Histogram, Gauge  # prometheus_client ^0.16.0
from typing import Dict, Any, Optional
import asyncio
import time
import uuid

# Internal imports
from api.schemas.generation import GenerationCreate, GenerationResponse
from services.generation import FreeBSDGenerationService
from core.metrics import track_generation_latency
from core.exceptions import VideoGenerationError
from core.logging import get_logger

# Initialize router
router = APIRouter(prefix='/api/v1/generation', tags=['generation'])

# Initialize logger
logger = get_logger(__name__)

# Performance metrics
GENERATION_LATENCY = Histogram(
    'video_generation_latency_seconds',
    'Latency of video generation operations',
    buckets=[.010, .025, .050, .075, .100, .250, .500]
)

FRAME_RATE = Gauge(
    'video_generation_fps',
    'Current frame generation rate'
)

@router.post('/', response_model=GenerationResponse)
@track_generation_latency
async def generate_video(
    generation_data: GenerationCreate,
    background_tasks: BackgroundTasks,
    generation_service: FreeBSDGenerationService = Depends()
) -> GenerationResponse:
    """
    Initiates video generation with FreeBSD-optimized performance.
    
    Args:
        generation_data: Validated generation request data
        background_tasks: FastAPI background task manager
        generation_service: Injected generation service
        
    Returns:
        GenerationResponse with ID and initial status
    """
    try:
        # Record generation start time
        start_time = time.time()
        
        # Generate unique ID
        generation_id = str(uuid.uuid4())
        
        # Validate generation parameters
        if not _validate_generation_params(generation_data.parameters):
            raise HTTPException(
                status_code=400,
                detail="Invalid generation parameters"
            )
        
        # Initialize generation in background
        background_tasks.add_task(
            generation_service.generate_video,
            generation_id=generation_id,
            prompt=generation_data.prompt,
            parameters=generation_data.parameters
        )
        
        # Record initial metrics
        GENERATION_LATENCY.observe(time.time() - start_time)
        
        return GenerationResponse(
            id=generation_id,
            status="pending",
            metrics={
                "initialization_time": time.time() - start_time,
                "target_fps": 24,
                "target_resolution": "720p"
            }
        )
        
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Video generation failed: {str(e)}"
        )

@router.websocket('/stream/{generation_id}')
async def stream_frames(
    websocket: WebSocket,
    generation_id: str,
    generation_service: FreeBSDGenerationService = Depends()
):
    """
    Streams generated video frames with FreeBSD-optimized performance.
    
    Args:
        websocket: WebSocket connection
        generation_id: Unique generation identifier
        generation_service: Injected generation service
    """
    try:
        # Accept WebSocket connection
        await websocket.accept()
        
        # Initialize frame streaming
        frame_count = 0
        stream_start = time.time()
        
        # Stream frames with performance monitoring
        async for frame in generation_service.stream_frames(generation_id):
            # Send frame data
            await websocket.send_bytes(frame)
            
            # Update metrics
            frame_count += 1
            elapsed = time.time() - stream_start
            if elapsed > 0:
                FRAME_RATE.set(frame_count / elapsed)
                
            # Check performance thresholds
            if elapsed > 0 and (frame_count / elapsed) < 24:
                logger.warning(f"Frame rate below target: {frame_count / elapsed:.2f} FPS")
                
    except VideoGenerationError as e:
        logger.error(f"Frame streaming failed: {str(e)}", exc_info=True)
        await websocket.close(code=1011, reason=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected streaming error: {str(e)}", exc_info=True)
        await websocket.close(code=1011, reason="Internal server error")
        
    finally:
        # Cleanup resources
        await websocket.close()

@router.post('/{generation_id}/control')
@track_generation_latency
async def process_control(
    generation_id: str,
    control_data: Dict[str, Any],
    generation_service: FreeBSDGenerationService = Depends()
) -> Dict[str, Any]:
    """
    Processes real-time control signals with <50ms response time.
    
    Args:
        generation_id: Unique generation identifier
        control_data: Control signal data
        generation_service: Injected generation service
        
    Returns:
        Control processing status and metrics
    """
    try:
        # Record control start time
        start_time = time.time()
        
        # Process control signal
        result = await generation_service.process_control(
            generation_id=generation_id,
            control_signal=control_data,
            immediate_feedback=True
        )
        
        # Record control latency
        latency = time.time() - start_time
        GENERATION_LATENCY.observe(latency)
        
        if latency > 0.05:  # 50ms threshold
            logger.warning(f"Control latency exceeded threshold: {latency*1000:.2f}ms")
        
        return {
            "status": "success",
            "latency": latency,
            "control_applied": result.get("control_applied", False)
        }
        
    except Exception as e:
        logger.error(f"Control processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Control processing failed: {str(e)}"
        )

def _validate_generation_params(parameters: Dict[str, Any]) -> bool:
    """Validates generation parameters against system requirements."""
    try:
        # Validate resolution
        if parameters.get("resolution_width") != 1280 or parameters.get("resolution_height") != 720:
            return False
            
        # Validate frame count
        if parameters.get("frame_count") != 102:  # Required for 24 FPS
            return False
            
        # Validate perspective
        if parameters.get("perspective") not in ["first_person", "third_person"]:
            return False
            
        return True
        
    except Exception:
        return False