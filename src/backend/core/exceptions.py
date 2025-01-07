# External imports with versions
from typing import Dict, Optional, List  # ^3.9.0
import uuid
import time

# Internal imports
from core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

class GameGenBaseException(Exception):
    """Base exception class for all GameGen-X custom exceptions with enhanced error tracking and recovery hints."""
    
    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        recovery_hints: Optional[Dict[str, any]] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
        self.recovery_hints = recovery_hints or {}
        self.error_code = error_code or f"GGX-{str(uuid.uuid4())[:8]}"
        self.timestamp = time.time()
        
        # Log exception with full context
        logger.error(
            f"{self.__class__.__name__}: {message}",
            error_code=self.error_code,
            original_error=str(original_error) if original_error else None,
            recovery_hints=self.recovery_hints,
            exc_info=True
        )

    def get_recovery_strategy(self) -> Dict[str, any]:
        """Returns recommended recovery strategy based on error context."""
        strategy = {
            "error_code": self.error_code,
            "timestamp": self.timestamp,
            "recovery_hints": self.recovery_hints,
            "retry_recommended": bool(self.recovery_hints),
            "severity": "error"
        }
        return strategy

class ModelError(GameGenBaseException):
    """Exception for model-specific errors with detailed context and recovery options."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        model_context: Dict[str, any],
        original_error: Optional[Exception] = None
    ):
        self.model_name = model_name
        self.model_context = model_context
        self.is_recoverable = self._analyze_recoverability()
        
        recovery_hints = {
            "model_name": model_name,
            "is_recoverable": self.is_recoverable,
            "recovery_options": self.get_model_recovery_options()
        }
        
        super().__init__(
            message=message,
            original_error=original_error,
            recovery_hints=recovery_hints,
            error_code=f"MODEL-{str(uuid.uuid4())[:8]}"
        )

    def _analyze_recoverability(self) -> bool:
        """Analyzes if the model error is recoverable based on context."""
        # Implement recoverability analysis logic
        return True

    def get_model_recovery_options(self) -> Dict[str, any]:
        """Returns model-specific recovery options."""
        return {
            "fallback_model": f"{self.model_name}_fallback",
            "retry_params": {
                "max_attempts": 3,
                "backoff_factor": 2.0
            },
            "alternative_approaches": [
                "reduce_batch_size",
                "use_cpu_fallback",
                "load_checkpoint"
            ]
        }

class VideoGenerationError(GameGenBaseException):
    """Exception for video generation errors with state recovery support."""
    
    def __init__(
        self,
        message: str,
        generation_id: str,
        generation_state: Dict[str, any],
        original_error: Optional[Exception] = None
    ):
        self.generation_id = generation_id
        self.generation_state = generation_state
        self.affected_frames = self._identify_affected_frames()
        
        recovery_hints = {
            "generation_id": generation_id,
            "affected_frames": self.affected_frames,
            "recovery_plan": self.get_state_recovery_plan()
        }
        
        super().__init__(
            message=message,
            original_error=original_error,
            recovery_hints=recovery_hints,
            error_code=f"GEN-{str(uuid.uuid4())[:8]}"
        )

    def _identify_affected_frames(self) -> List[str]:
        """Identifies frames affected by the generation error."""
        # Implement frame analysis logic
        return []

    def get_state_recovery_plan(self) -> Dict[str, any]:
        """Generates recovery plan for failed generation."""
        return {
            "checkpoint_frame": self.generation_state.get("last_successful_frame", 0),
            "recovery_strategy": "incremental",
            "frame_recovery_sequence": self.affected_frames,
            "estimated_recovery_time": len(self.affected_frames) * 0.1  # 100ms per frame
        }

class FreeBSDError(GameGenBaseException):
    """Exception for FreeBSD-specific system errors with compatibility layer support."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        system_context: Dict[str, any],
        original_error: Optional[Exception] = None
    ):
        self.operation = operation
        self.system_context = system_context
        self.compatibility_layer = self._identify_compatibility_layer()
        
        recovery_hints = {
            "operation": operation,
            "compatibility_options": self.get_compatibility_options(),
            "system_context": system_context
        }
        
        super().__init__(
            message=message,
            original_error=original_error,
            recovery_hints=recovery_hints,
            error_code=f"BSD-{str(uuid.uuid4())[:8]}"
        )

    def _identify_compatibility_layer(self) -> str:
        """Identifies the relevant FreeBSD compatibility layer."""
        return "native"

    def get_compatibility_options(self) -> Dict[str, any]:
        """Returns compatibility options for FreeBSD operation."""
        return {
            "alternative_syscalls": [
                self.operation + "_alternative",
                self.operation + "_compat"
            ],
            "compatibility_modes": [
                "native",
                "linux_compat",
                "32bit_compat"
            ],
            "system_workarounds": {
                "use_jail": bool(self.system_context.get("jail_support")),
                "force_native": True
            }
        }

class ValidationError(GameGenBaseException):
    """Exception for input validation errors with detailed error context."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Dict[str, any],
        validation_context: str
    ):
        self.validation_errors = validation_errors
        self.validation_context = validation_context
        self.correction_suggestions = self.get_correction_suggestions()
        
        recovery_hints = {
            "validation_context": validation_context,
            "correction_suggestions": self.correction_suggestions,
            "validation_errors": validation_errors
        }
        
        super().__init__(
            message=message,
            recovery_hints=recovery_hints,
            error_code=f"VAL-{str(uuid.uuid4())[:8]}"
        )

    def get_correction_suggestions(self) -> List[str]:
        """Returns suggestions for correcting validation errors."""
        suggestions = []
        for field, error in self.validation_errors.items():
            if isinstance(error, dict):
                suggestions.append(f"Field '{field}': {error.get('message', 'Invalid value')}")
            else:
                suggestions.append(f"Field '{field}': {str(error)}")
        return suggestions