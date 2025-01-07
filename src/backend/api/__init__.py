# External imports with versions
import logging  # ^3.9
from prometheus_client import start_http_server  # ^0.16.0
import time

# Internal imports
from .config import api_settings
from .main import app

# Initialize logger
logger = logging.getLogger(__name__)

# Export version from settings
__version__ = api_settings.version

def initialize_api() -> None:
    """Initialize API components, security controls, and monitoring with FreeBSD optimization."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO if not api_settings.debug_mode else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Initialize FreeBSD-specific optimizations
        if api_settings.freebsd_worker_config:
            for sysctl, value in api_settings.freebsd_worker_config.items():
                logger.info(f"Configuring FreeBSD sysctl {sysctl}: {value}")

        # Initialize security middleware
        app.middleware.append(
            "api.middleware.RequestLoggingMiddleware",
            "api.middleware.MetricsMiddleware",
            "api.middleware.ErrorHandlingMiddleware"
        )

        # Start Prometheus metrics server
        start_http_server(9090)
        logger.info("Metrics server started on port 9090")

        # Initialize API state
        app.state.startup_time = time.time()
        app.state.environment = api_settings.environment
        app.state.version = __version__

        logger.info(
            f"API initialized successfully in {api_settings.environment} environment "
            f"(version {__version__})"
        )

    except Exception as e:
        logger.error(f"API initialization failed: {str(e)}")
        raise

def health_check() -> dict:
    """API health check function for monitoring."""
    try:
        return {
            "status": "healthy",
            "version": __version__,
            "environment": api_settings.environment,
            "uptime": time.time() - app.state.startup_time,
            "settings_loaded": bool(api_settings),
            "app_initialized": bool(app)
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Export core components
__all__ = [
    "app",
    "api_settings",
    "__version__",
    "health_check"
]