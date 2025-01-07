"""
FastAPI route handler for real-time video control operations.
Implements keyboard inputs, environment controls, and instruction modifications
with <50ms response time and FreeBSD optimization.

FastAPI version: ^0.95.0
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from uuid import UUID
import structlog
import time

# Internal imports
from api.schemas.control import ControlBase, ControlCreate, ControlResponse
from services.control import ControlService
from api.dependencies import get_db, get_current_user, check_rate_limit
from core.metrics import CONTROL_RESPONSE, track_jail_metrics

# Initialize router with prefix and tags
router = APIRouter(prefix='/api/v1/control', tags=['control'])

# Initialize logger
logger = structlog.get_logger(__name__)

# Constants
RATE_LIMIT = "60/min"  # 60 requests per minute
RESPONSE_TIMEOUT = 0.05  # 50ms timeout

@router.post('/', response_model=ControlResponse)
@check_rate_limit(limit=RATE_LIMIT)
async def create_control(
    control: ControlCreate,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """
    Creates and processes a new control signal for video generation with FreeBSD optimizations.
    
    Args:
        control: Control signal data
        db: Database session
        current_user: Authenticated user
        
    Returns:
        ControlResponse: Processed control response with performance metrics
        
    Raises:
        HTTPException: If control processing fails or timeout occurs
    """
    start_time = time.perf_counter()
    
    try:
        # Initialize control service
        control_service = ControlService(db_session=db, processor=None)  # Processor injected by dependency system
        
        # Process control signal
        result = await control_service.process_control(
            generation_id=control.generation_id,
            video_id=control.video_id,
            control_data=control
        )
        
        # Calculate response time
        response_time = time.perf_counter() - start_time
        
        # Check response time threshold
        if response_time > RESPONSE_TIMEOUT:
            logger.warning(
                "Control response time exceeded threshold",
                response_time=response_time,
                threshold=RESPONSE_TIMEOUT
            )
        
        # Track metrics
        CONTROL_RESPONSE.observe(response_time)
        track_jail_metrics("control_service")
        
        # Return response with metrics
        return ControlResponse(
            id=result["control_id"],
            generation_id=control.generation_id,
            video_id=control.video_id,
            type=control.type,
            data=control.data,
            timestamp=control.timestamp,
            metadata={
                "latency_ms": result["latency_ms"],
                "processed": True
            }
        )
        
    except Exception as e:
        logger.error(
            "Control processing failed",
            error=str(e),
            control_type=control.type,
            generation_id=str(control.generation_id)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Control processing failed: {str(e)}"
        )

@router.get('/{generation_id}', response_model=List[ControlResponse])
async def get_controls(
    generation_id: UUID,
    db=Depends(get_db),
    current_user=Depends(get_current_user),
    page_size: int = 100,
    page_number: int = 0
):
    """
    Retrieves active controls for a specific video generation with caching.
    
    Args:
        generation_id: Generation ID to retrieve controls for
        db: Database session
        current_user: Authenticated user
        page_size: Number of controls per page
        page_number: Page number for pagination
        
    Returns:
        List[ControlResponse]: List of active controls with performance data
    """
    try:
        # Initialize control service
        control_service = ControlService(db_session=db, processor=None)
        
        # Get control history with pagination
        controls = await control_service.get_control_history(
            video_id=generation_id,
            limit=page_size,
            offset=page_number * page_size
        )
        
        # Track metrics
        track_jail_metrics("control_service")
        
        return [
            ControlResponse(**control) for control in controls
        ]
        
    except Exception as e:
        logger.error(
            "Failed to retrieve controls",
            error=str(e),
            generation_id=str(generation_id)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve controls: {str(e)}"
        )

@router.delete('/{generation_id}')
async def clear_controls(
    generation_id: UUID,
    db=Depends(get_db),
    current_user=Depends(get_current_user)
):
    """
    Clears all active controls for a video generation with audit logging.
    
    Args:
        generation_id: Generation ID to clear controls for
        db: Database session
        current_user: Authenticated user
        
    Returns:
        dict: Success message with cleanup metrics
    """
    try:
        # Initialize control service
        control_service = ControlService(db_session=db, processor=None)
        
        # Clear controls and get metrics
        start_time = time.perf_counter()
        cleared_count = await control_service.clear_frame_cache(str(generation_id))
        cleanup_time = time.perf_counter() - start_time
        
        # Track metrics
        track_jail_metrics("control_service")
        
        return {
            "message": "Controls cleared successfully",
            "generation_id": str(generation_id),
            "cleared_count": cleared_count,
            "cleanup_time_ms": round(cleanup_time * 1000, 2)
        }
        
    except Exception as e:
        logger.error(
            "Failed to clear controls",
            error=str(e),
            generation_id=str(generation_id)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear controls: {str(e)}"
        )