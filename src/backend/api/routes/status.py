"""
FastAPI route handlers for system status, resource metrics, and performance monitoring endpoints.
Provides real-time monitoring data for the GameGen-X system with FreeBSD-specific monitoring capabilities.
"""

# External imports with versions
from fastapi import APIRouter, Depends, HTTPException, status  # ^0.95.0

# Internal imports
from api.schemas.status import (
    SystemStatus,
    ResourceMetrics,
    PerformanceMetrics,
    StatusResponse
)
from services.monitoring import MonitoringService

# Initialize router with prefix and tags
router = APIRouter(prefix='/api/v1/status', tags=['status'])

@router.get('/', response_model=StatusResponse)
@router.get('/health', response_model=StatusResponse)
async def get_system_status(
    monitoring_service: MonitoringService = Depends()
) -> StatusResponse:
    """
    Get current system status and health metrics with enhanced FreeBSD monitoring.
    
    Returns:
        StatusResponse: Combined system status response with FreeBSD metrics
    """
    try:
        # Get system metrics from monitoring service
        metrics = monitoring_service.get_system_metrics()
        
        # Validate metrics against thresholds
        monitoring_service.validate_metrics(metrics)
        
        # Construct system status response
        status_response = StatusResponse(
            system_status=SystemStatus(
                status=metrics['status'],
                uptime_seconds=metrics['uptime'],
                version=metrics['version'],
                environment=metrics['environment']
            ),
            resource_metrics=ResourceMetrics(
                cpu_usage_percent=metrics['resources']['cpu'],
                memory_usage_percent=metrics['resources']['memory'],
                gpu_usage_percent=metrics['resources']['gpu'],
                disk_usage_percent=metrics['resources']['storage']
            ),
            performance_metrics=PerformanceMetrics(
                generation_latency_ms=metrics['performance']['generation_latency'],
                frame_rate_fps=metrics['performance']['frame_rate'],
                control_response_ms=metrics['performance']['control_response'],
                concurrent_users=metrics['performance']['concurrent_users']
            )
        )
        
        return status_response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )

@router.get('/resources', response_model=ResourceMetrics)
async def get_resource_metrics(
    monitoring_service: MonitoringService = Depends()
) -> ResourceMetrics:
    """
    Get current resource utilization metrics with FreeBSD-specific monitoring.
    
    Returns:
        ResourceMetrics: Resource utilization metrics including FreeBSD specifics
    """
    try:
        # Get resource metrics with FreeBSD specifics
        metrics = monitoring_service.get_resource_metrics()
        
        # Construct resource metrics response
        resource_metrics = ResourceMetrics(
            cpu_usage_percent=metrics['cpu_usage'],
            memory_usage_percent=metrics['memory_usage'],
            gpu_usage_percent=metrics['gpu_usage'],
            disk_usage_percent=metrics['disk_usage'],
            freebsd_gpu_metrics=metrics.get('freebsd_gpu_metrics', {})
        )
        
        # Validate against thresholds
        if resource_metrics.cpu_usage_percent > 90:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="CPU usage exceeds threshold"
            )
            
        if resource_metrics.memory_usage_percent > 85:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Memory usage exceeds threshold"
            )
            
        return resource_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get resource metrics: {str(e)}"
        )

@router.get('/performance', response_model=PerformanceMetrics)
async def get_performance_metrics(
    monitoring_service: MonitoringService = Depends()
) -> PerformanceMetrics:
    """
    Get current performance metrics with strict validation against requirements.
    
    Returns:
        PerformanceMetrics: System performance metrics with validation results
    """
    try:
        # Get performance metrics
        metrics = monitoring_service.get_performance_metrics()
        
        # Construct performance metrics with validation
        performance_metrics = PerformanceMetrics(
            generation_latency_ms=metrics['generation_latency'],
            frame_rate_fps=metrics['frame_rate'],
            control_response_ms=metrics['control_response'],
            concurrent_users=metrics['concurrent_users']
        )
        
        # Validate against strict requirements
        if performance_metrics.generation_latency_ms > 100:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Generation latency exceeds 100ms requirement"
            )
            
        if performance_metrics.frame_rate_fps < 24:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Frame rate below 24 FPS requirement"
            )
            
        if performance_metrics.control_response_ms > 50:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Control response exceeds 50ms requirement"
            )
            
        if performance_metrics.concurrent_users > 100:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Concurrent users exceed maximum capacity"
            )
            
        return performance_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )