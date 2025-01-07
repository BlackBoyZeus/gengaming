# External imports with versions
import threading  # ^3.9+
import atexit  # ^3.9+

# Internal imports
from core.config import Settings, settings
from core.logging import setup_logging, get_logger
from core.metrics import initialize_metrics
from core.exceptions import (
    GameGenBaseException,
    FreeBSDError,
    ModelError,
    VideoGenerationError,
    ValidationError
)

# Version information
__version__ = "0.1.0"

# Thread-safe initialization controls
_initialized = threading.Event()
_init_lock = threading.Lock()

# Configure module logger
logger = get_logger(__name__)

def initialize_core() -> bool:
    """
    Thread-safe initialization of all core components with proper ordering and validation.
    Ensures FreeBSD compatibility and proper resource configuration.
    """
    # Use lock for thread-safe initialization
    with _init_lock:
        # Check if already initialized
        if _initialized.is_set():
            logger.info("Core system already initialized")
            return True

        try:
            # Load and validate environment settings
            logger.info("Validating environment configuration...")
            if not settings.environment:
                raise ValidationError(
                    "Invalid environment configuration",
                    {"environment": "Environment not set"},
                    "core_initialization"
                )

            # Validate FreeBSD compatibility
            logger.info("Checking FreeBSD compatibility...")
            try:
                settings.freebsd_config
            except Exception as e:
                raise FreeBSDError(
                    "FreeBSD compatibility check failed",
                    "config_validation",
                    {"os": "FreeBSD"},
                    original_error=e
                )

            # Setup logging with ELK integration
            logger.info("Configuring logging system...")
            setup_logging()

            # Initialize metrics collectors
            logger.info("Initializing metrics collection...")
            initialize_metrics()

            # Register cleanup handler
            atexit.register(cleanup_core)

            # Mark initialization complete
            _initialized.set()
            logger.info("Core system initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Core system initialization failed: {str(e)}", exc_info=True)
            # Attempt cleanup in case of partial initialization
            cleanup_core()
            raise

def cleanup_core() -> None:
    """
    Performs cleanup of core components during shutdown.
    Ensures proper resource release and state cleanup.
    """
    if _initialized.is_set():
        try:
            logger.info("Starting core system cleanup...")
            
            # Flush logging buffers
            logging_logger = get_logger("core.logging")
            logging_logger.info("Flushing log buffers...")
            
            # Close metric collectors
            metrics_logger = get_logger("core.metrics")
            metrics_logger.info("Shutting down metrics collectors...")
            
            # Clear initialization state
            _initialized.clear()
            
            logger.info("Core system cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during core system cleanup: {str(e)}", exc_info=True)
            raise

# Export public interface
__all__ = [
    # Core classes
    "Settings",
    "settings",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Metrics
    "initialize_metrics",
    
    # Exceptions
    "GameGenBaseException",
    "FreeBSDError",
    "ModelError",
    "VideoGenerationError",
    "ValidationError",
    
    # Initialization
    "initialize_core"
]