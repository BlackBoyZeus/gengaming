"""
GameGen-X Utilities Module
Provides centralized access to FreeBSD system utilities, GPU management, video processing,
and other core functionality with comprehensive error handling and performance monitoring.

Version: 1.0.0
"""

# External imports with versions
from typing import Dict, Any, Optional

# Internal imports
from utils.freebsd import (
    FreeBSDManager,
    set_resource_limits,
    optimize_system
)
from utils.gpu import (
    GPUManager,
    initialize_gpu,
    get_gpu_info
)
from utils.video import (
    VideoProcessor,
    validate_frame,
    calculate_frame_rate
)

# Version information
__version__ = '1.0.0'

# Initialize logging
from core.logging import get_logger
logger = get_logger(__name__)

# Export core functionality
__all__ = [
    # FreeBSD system utilities
    'FreeBSDManager',
    'set_resource_limits',
    'optimize_system',
    
    # GPU management
    'GPUManager',
    'initialize_gpu',
    'get_gpu_info',
    
    # Video processing
    'VideoProcessor',
    'validate_frame',
    'calculate_frame_rate',
    
    # Version info
    '__version__'
]

def initialize_system(
    gpu_settings: Optional[Dict[str, Any]] = None,
    video_settings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Initialize core system components with FreeBSD compatibility and GPU optimization.
    
    Args:
        gpu_settings: Optional GPU configuration settings
        video_settings: Optional video processing settings
    
    Returns:
        Dict containing initialization status and configurations
    """
    try:
        # Initialize FreeBSD system
        freebsd_manager = FreeBSDManager()
        system_status = freebsd_manager.initialize()
        
        # Initialize GPU subsystem
        gpu_manager = GPUManager(gpu_settings or {})
        gpu_status = gpu_manager.initialize()
        
        # Initialize video processor
        video_processor = VideoProcessor(
            gpu_settings=gpu_settings,
            video_settings=video_settings
        )
        
        return {
            'status': 'initialized',
            'system': system_status,
            'gpu': gpu_status,
            'video': {
                'frame_rate': video_processor._frame_rate,
                'resolution': video_processor._resolution
            }
        }
        
    except Exception as e:
        logger.error(f"System initialization failed: {str(e)}")
        raise

def cleanup_system() -> None:
    """
    Perform cleanup of system resources and handles.
    """
    try:
        # Cleanup GPU resources
        gpu_manager = GPUManager({})
        gpu_manager.cleanup()
        
        logger.info("System cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"System cleanup failed: {str(e)}")
        raise

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information including FreeBSD, GPU, and video subsystems.
    
    Returns:
        Dict containing system information and status
    """
    try:
        return {
            'version': __version__,
            'freebsd': FreeBSDManager().get_system_info(),
            'gpu': get_gpu_info(),
            'video': {
                'supported_resolutions': [(1280, 720)],
                'target_frame_rate': 24
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {str(e)}")
        raise