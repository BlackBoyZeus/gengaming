# External imports with versions
from prometheus_client import Counter, Gauge  # prometheus_client ^0.16.0
import sentry_sdk  # sentry_sdk ^1.28.1
from typing import Dict, Any, Optional
import logging

# Internal imports
from services.video import VideoService
from services.generation import GenerationService
from services.control import ControlService
from services.cache import FrameCache

# Configure logging
logger = logging.getLogger(__name__)

# Global performance thresholds
PERFORMANCE_THRESHOLD_MS = 100  # Maximum latency threshold
FRAME_RATE_TARGET = 24  # Target FPS
QUALITY_THRESHOLD_FID = 300  # Maximum FID score
QUALITY_THRESHOLD_FVD = 1000  # Maximum FVD score

# Initialize performance metrics
GENERATION_COUNTER = Counter(
    'video_generation_total',
    'Total number of video generations',
    ['status']
)

CONTROL_LATENCY = Gauge(
    'control_latency_milliseconds',
    'Control operation latency in milliseconds'
)

FRAME_RATE = Gauge(
    'video_frame_rate',
    'Current video frame rate'
)

class ServiceInitializationError(Exception):
    """Custom exception for service initialization failures."""
    pass

def initialize_services(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initialize all services with comprehensive error handling and monitoring."""
    try:
        # Initialize core services
        video_service = VideoService(settings)
        generation_service = GenerationService(settings)
        control_service = ControlService(settings)
        frame_cache = FrameCache(settings)

        # Validate service initialization
        if not _validate_services(
            video_service,
            generation_service,
            control_service,
            frame_cache
        ):
            raise ServiceInitializationError("Service validation failed")

        # Initialize performance monitoring
        _setup_monitoring()

        return {
            'video_service': video_service,
            'generation_service': generation_service,
            'control_service': control_service,
            'frame_cache': frame_cache
        }

    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise ServiceInitializationError(f"Service initialization failed: {str(e)}")

def _validate_services(*services) -> bool:
    """Validate initialized services meet performance requirements."""
    try:
        for service in services:
            if not hasattr(service, 'validate_performance'):
                logger.warning(f"Service {service.__class__.__name__} missing validation method")
                continue

            if not service.validate_performance():
                logger.error(f"Service {service.__class__.__name__} failed validation")
                return False

        return True

    except Exception as e:
        logger.error(f"Service validation failed: {str(e)}")
        return False

def _setup_monitoring():
    """Configure performance monitoring and error tracking."""
    try:
        # Initialize Sentry error tracking
        sentry_sdk.init(
            dsn="your-sentry-dsn",
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0
        )

        # Set up performance metric thresholds
        CONTROL_LATENCY.set_function(lambda: PERFORMANCE_THRESHOLD_MS)
        FRAME_RATE.set_function(lambda: FRAME_RATE_TARGET)

    except Exception as e:
        logger.error(f"Failed to setup monitoring: {str(e)}")
        raise

# Export service classes and initialization function
__all__ = [
    "VideoService",
    "GenerationService",
    "ControlService",
    "FrameCache",
    "initialize_services",
    "PERFORMANCE_THRESHOLD_MS",
    "FRAME_RATE_TARGET",
    "QUALITY_THRESHOLD_FID",
    "QUALITY_THRESHOLD_FVD"
]