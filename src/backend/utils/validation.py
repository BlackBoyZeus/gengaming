# External imports with versions
from typing import Dict, Tuple, Optional  # ^3.9.0
from pydantic import BaseModel, validator, Field  # ^1.9.0
import time

# Internal imports
from core.exceptions import ValidationError

# Global constants
VALID_RESOLUTIONS = ["720p"]
MIN_FRAMES = 24
MAX_FRAMES = 102
VALID_PERSPECTIVES = ["first_person", "third_person"]
VALID_CONTROL_TYPES = ["keyboard", "environment"]
MAX_PROCESSING_TIME_MS = 100
TARGET_FPS = 24
MAX_PROMPT_LENGTH = 1000

# Validation schemas
class GenerationParameters(BaseModel):
    resolution: str
    frame_count: int
    perspective: str
    frame_rate: Optional[int] = TARGET_FPS

    @validator('resolution')
    def validate_resolution(cls, v):
        if v not in VALID_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {VALID_RESOLUTIONS}")
        return v

    @validator('frame_count')
    def validate_frame_count(cls, v):
        if not MIN_FRAMES <= v <= MAX_FRAMES:
            raise ValueError(f"Frame count must be between {MIN_FRAMES} and {MAX_FRAMES}")
        return v

    @validator('perspective')
    def validate_perspective(cls, v):
        if v not in VALID_PERSPECTIVES:
            raise ValueError(f"Perspective must be one of {VALID_PERSPECTIVES}")
        return v

    @validator('frame_rate')
    def validate_frame_rate(cls, v):
        if v != TARGET_FPS:
            raise ValueError(f"Frame rate must be {TARGET_FPS} FPS")
        return v

def validate_generation_parameters(parameters: Dict) -> Dict:
    """
    Validates video generation parameters against defined constraints including performance requirements.
    
    Args:
        parameters: Dictionary containing generation parameters
        
    Returns:
        Dict: Validated parameters dictionary
        
    Raises:
        ValidationError: If parameters fail validation with detailed error context
    """
    start_time = time.time()
    
    try:
        validated_params = GenerationParameters(**parameters).dict()
        
        # Check processing time
        processing_time = (time.time() - start_time) * 1000
        if processing_time > MAX_PROCESSING_TIME_MS:
            raise ValueError(f"Parameter validation exceeded time limit: {processing_time:.2f}ms")
            
        return validated_params
        
    except Exception as e:
        validation_errors = {
            "error": str(e),
            "parameters": parameters,
            "processing_time": f"{(time.time() - start_time) * 1000:.2f}ms"
        }
        raise ValidationError(
            message="Generation parameter validation failed",
            validation_errors=validation_errors,
            validation_context="generation_parameters"
        )

def validate_control_input(control_type: str, control_data: Dict) -> Tuple[str, Dict]:
    """
    Validates real-time control input data with performance checks.
    
    Args:
        control_type: Type of control input
        control_data: Control input data
        
    Returns:
        Tuple[str, Dict]: Validated control type and data
        
    Raises:
        ValidationError: If control input fails validation
    """
    start_time = time.time()
    
    try:
        if control_type not in VALID_CONTROL_TYPES:
            raise ValueError(f"Invalid control type: {control_type}")
            
        # Validate control data structure based on type
        if control_type == "keyboard":
            required_keys = ["key", "action"]
            if not all(key in control_data for key in required_keys):
                raise ValueError(f"Missing required keys for keyboard control: {required_keys}")
                
        elif control_type == "environment":
            required_keys = ["parameter", "value"]
            if not all(key in control_data for key in required_keys):
                raise ValueError(f"Missing required keys for environment control: {required_keys}")
                
        # Check processing time
        processing_time = (time.time() - start_time) * 1000
        if processing_time > MAX_PROCESSING_TIME_MS:
            raise ValueError(f"Control validation exceeded time limit: {processing_time:.2f}ms")
            
        return control_type, control_data
        
    except Exception as e:
        validation_errors = {
            "error": str(e),
            "control_type": control_type,
            "control_data": control_data,
            "processing_time": f"{(time.time() - start_time) * 1000:.2f}ms"
        }
        raise ValidationError(
            message="Control input validation failed",
            validation_errors=validation_errors,
            validation_context="control_input"
        )

def validate_prompt(prompt: str) -> str:
    """
    Validates generation prompt text with enhanced checks.
    
    Args:
        prompt: Input prompt text
        
    Returns:
        str: Validated prompt text
        
    Raises:
        ValidationError: If prompt fails validation
    """
    start_time = time.time()
    
    try:
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
            
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_LENGTH} characters")
            
        # Check processing time
        processing_time = (time.time() - start_time) * 1000
        if processing_time > MAX_PROCESSING_TIME_MS:
            raise ValueError(f"Prompt validation exceeded time limit: {processing_time:.2f}ms")
            
        return prompt.strip()
        
    except Exception as e:
        validation_errors = {
            "error": str(e),
            "prompt_length": len(prompt) if prompt else 0,
            "processing_time": f"{(time.time() - start_time) * 1000:.2f}ms"
        }
        raise ValidationError(
            message="Prompt validation failed",
            validation_errors=validation_errors,
            validation_context="prompt"
        )

class GenerationValidator:
    """Enhanced class for validating video generation requests with performance monitoring."""
    
    def __init__(self):
        """Initializes validation rules."""
        self._validation_rules = {
            "resolution": VALID_RESOLUTIONS,
            "frame_count": {"min": MIN_FRAMES, "max": MAX_FRAMES},
            "perspective": VALID_PERSPECTIVES,
            "frame_rate": TARGET_FPS
        }
        
    def validate(self, request_data: Dict) -> Dict:
        """
        Validates complete generation request with performance checks.
        
        Args:
            request_data: Generation request data
            
        Returns:
            Dict: Validated request data
            
        Raises:
            ValidationError: If request fails validation
        """
        start_time = time.time()
        
        try:
            # Validate prompt
            prompt = validate_prompt(request_data.get("prompt", ""))
            
            # Validate generation parameters
            parameters = validate_generation_parameters(request_data.get("parameters", {}))
            
            # Check total validation time
            total_time = (time.time() - start_time) * 1000
            if total_time > MAX_PROCESSING_TIME_MS:
                raise ValueError(f"Total validation exceeded time limit: {total_time:.2f}ms")
                
            return {
                "prompt": prompt,
                "parameters": parameters
            }
            
        except Exception as e:
            validation_errors = {
                "error": str(e),
                "request_data": request_data,
                "processing_time": f"{(time.time() - start_time) * 1000:.2f}ms"
            }
            raise ValidationError(
                message="Generation request validation failed",
                validation_errors=validation_errors,
                validation_context="generation_request"
            )

class ControlValidator:
    """Enhanced class for validating control requests with performance monitoring."""
    
    def __init__(self):
        """Initializes control validation rules."""
        self._validation_rules = {
            "keyboard": ["key", "action"],
            "environment": ["parameter", "value"]
        }
        
    def validate(self, request_data: Dict) -> Dict:
        """
        Validates control request with performance checks.
        
        Args:
            request_data: Control request data
            
        Returns:
            Dict: Validated control data
            
        Raises:
            ValidationError: If request fails validation
        """
        start_time = time.time()
        
        try:
            control_type = request_data.get("type")
            control_data = request_data.get("data", {})
            
            # Validate control input
            validated_type, validated_data = validate_control_input(control_type, control_data)
            
            # Check validation time
            processing_time = (time.time() - start_time) * 1000
            if processing_time > MAX_PROCESSING_TIME_MS:
                raise ValueError(f"Control validation exceeded time limit: {processing_time:.2f}ms")
                
            return {
                "type": validated_type,
                "data": validated_data
            }
            
        except Exception as e:
            validation_errors = {
                "error": str(e),
                "request_data": request_data,
                "processing_time": f"{(time.time() - start_time) * 1000:.2f}ms"
            }
            raise ValidationError(
                message="Control request validation failed",
                validation_errors=validation_errors,
                validation_context="control_request"
            )