# External imports with versions
import logging  # ^3.9
import structlog  # ^23.1.0
from typing import Dict, Optional, Any, Type, Union  # ^3.9
from elasticsearch import Elasticsearch  # ^8.0.0
from pythonjsonlogger.jsonlogger import JsonFormatter  # ^2.0.0
import uuid
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Internal imports
from core.config import settings

# Global constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(correlation_id)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
DEFAULT_LOG_LEVEL = logging.INFO
SENSITIVE_FIELDS = ["password", "token", "api_key", "secret"]
MAX_LOG_SIZE = 100 * 1024 * 1024  # 100MB
LOG_BACKUP_COUNT = 10

def setup_logging(log_level: Optional[str] = None, elk_config: Optional[Dict] = None) -> None:
    """
    Configures application-wide logging with proper handlers, formatters, and ELK integration.
    Implements FreeBSD-compatible paths and comprehensive error tracking.
    """
    # Determine log level
    level = getattr(logging, log_level or 'INFO' if not settings.debug else 'DEBUG')
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt=LOG_DATE_FORMAT),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _mask_sensitive_data,
        structlog.processors.JSONRenderer()
    ]

    # Configure FreeBSD-compatible log directory
    log_dir = Path("/var/log/gamegen-x")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "gamegen-x.log"

    # Configure handlers
    handlers = []
    
    # Console handler with color support
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JsonFormatter(LOG_FORMAT))
    handlers.append(console_handler)
    
    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(JsonFormatter(LOG_FORMAT))
    handlers.append(file_handler)
    
    # ELK handler configuration
    if elk_config or settings.elk_config:
        config = elk_config or settings.elk_config
        try:
            es_client = Elasticsearch(
                hosts=config.get('hosts', ['localhost:9200']),
                basic_auth=(config.get('username'), config.get('password')),
                verify_certs=config.get('verify_certs', True)
            )
            handlers.append(_create_elk_handler(es_client))
        except Exception as e:
            logging.warning(f"Failed to initialize ELK handler: {str(e)}")

    # Configure logging
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers
    )

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str, context: Optional[Dict] = None) -> structlog.BoundLogger:
    """
    Returns a configured logger instance with context binding and performance sampling.
    """
    logger = structlog.get_logger(name)
    
    # Bind default context
    base_context = {
        'environment': settings.environment,
        'correlation_id': str(uuid.uuid4()),
        'timestamp': time.time()
    }
    
    # Merge with provided context
    if context:
        base_context.update(context)
    
    return logger.bind(**base_context)

class LoggerContextManager:
    """
    Context manager for handling logging contexts with comprehensive error tracking and recovery.
    """
    
    def __init__(self, operation: str, logger_name: str, context: Optional[Dict] = None):
        self.operation = operation
        self.logger_name = logger_name
        self.correlation_id = str(uuid.uuid4())
        self.context = context or {}
        self.context['correlation_id'] = self.correlation_id
        self.logger = get_logger(logger_name, self.context)
        self.start_time = None

    def __enter__(self) -> 'LoggerContextManager':
        self.start_time = time.time()
        self.logger.info(
            f"Starting operation: {self.operation}",
            operation=self.operation,
            correlation_id=self.correlation_id
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any]
    ) -> None:
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            # Log error with full context
            self.logger.error(
                f"Operation failed: {self.operation}",
                operation=self.operation,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                duration=duration,
                correlation_id=self.correlation_id,
                exc_info=(exc_type, exc_val, exc_tb)
            )
        else:
            # Log successful completion
            self.logger.info(
                f"Operation completed: {self.operation}",
                operation=self.operation,
                duration=duration,
                correlation_id=self.correlation_id
            )

def _mask_sensitive_data(
    logger: Union[logging.Logger, structlog.BoundLogger],
    method_name: str,
    event_dict: Dict
) -> Dict:
    """Masks sensitive data in log messages."""
    for key in event_dict:
        if any(sensitive in key.lower() for sensitive in SENSITIVE_FIELDS):
            event_dict[key] = '***MASKED***'
    return event_dict

def _create_elk_handler(es_client: Elasticsearch) -> logging.Handler:
    """Creates a custom handler for ELK integration."""
    class ElkHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                log_entry = {
                    'timestamp': record.created,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'logger': record.name,
                    'correlation_id': getattr(record, 'correlation_id', None),
                    'environment': settings.environment
                }
                
                if hasattr(record, 'exc_info') and record.exc_info:
                    log_entry['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': self.formatter.formatException(record.exc_info)
                    }
                
                es_client.index(
                    index=f"gamegen-x-logs-{time.strftime('%Y.%m.%d')}",
                    document=log_entry
                )
            except Exception as e:
                # Fallback to console logging if ELK fails
                print(f"Failed to send log to ELK: {str(e)}")

    return ElkHandler()

__all__ = ['setup_logging', 'get_logger', 'LoggerContextManager']