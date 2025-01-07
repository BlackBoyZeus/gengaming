# External imports with versions
from pydantic import BaseModel  # pydantic ^2.0.0
from typing import Dict, List, Any, Optional  # typing ^3.9.0

# Internal imports
from core.config import Settings

# Global constants
DEFAULT_HIDDEN_DIM = 1024
DEFAULT_NUM_LAYERS = 8
DEFAULT_LATENT_SCALE = 0.1
DEFAULT_CONTROL_STRENGTH = 1.0
SUPPORTED_CONTROL_TYPES = ["keyboard", "environment", "character"]
PERFORMANCE_THRESHOLDS = {
    "response_time_ms": 50,  # <50ms response time requirement
    "control_accuracy": 0.5  # >50% control accuracy requirement
}
HARDWARE_REQUIREMENTS = {
    "freebsd_compatible": True,
    "non_nvidia_support": True
}

class InstructNetConfig(BaseModel):
    """Configuration class for InstructNet model architecture and runtime settings."""
    
    # Model architecture parameters
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    num_layers: int = DEFAULT_NUM_LAYERS
    latent_scale: float = DEFAULT_LATENT_SCALE
    control_strength: float = DEFAULT_CONTROL_STRENGTH
    supported_control_types: List[str] = SUPPORTED_CONTROL_TYPES
    
    # System settings from core config
    gpu_settings: Dict[str, Any] = Settings().gpu_settings
    resource_limits: Dict[str, int] = Settings().resource_limits
    
    # Performance and hardware settings
    performance_constraints: Dict[str, float] = {
        "max_response_time": PERFORMANCE_THRESHOLDS["response_time_ms"] / 1000.0,
        "min_control_accuracy": PERFORMANCE_THRESHOLDS["control_accuracy"],
        "max_memory_usage": resource_limits["max_gpu_memory"],
        "max_batch_size": resource_limits["max_batch_size"]
    }
    
    hardware_compatibility: Dict[str, bool] = {
        "freebsd_compatible": HARDWARE_REQUIREMENTS["freebsd_compatible"],
        "non_nvidia_support": HARDWARE_REQUIREMENTS["non_nvidia_support"]
    }
    
    monitoring_settings: Dict[str, Any] = {
        "track_response_time": True,
        "track_control_accuracy": True,
        "metrics_buffer_size": 1000,
        "alert_threshold_ms": 45  # Alert before hitting 50ms limit
    }

    def validate_control_settings(self) -> bool:
        """Validates control-related configuration settings with performance checks."""
        try:
            # Validate model architecture settings
            if self.hidden_dim < 512 or self.hidden_dim > 2048:
                return False
            
            # Validate control parameters
            if self.control_strength <= 0 or self.control_strength > 2.0:
                return False
                
            # Validate control types
            if not all(ct in SUPPORTED_CONTROL_TYPES for ct in self.supported_control_types):
                return False
                
            # Validate performance constraints
            if self.performance_constraints["max_response_time"] > 0.05:  # 50ms
                return False
                
            # Validate hardware compatibility
            if not all(self.hardware_compatibility.values()):
                return False
                
            return True
            
        except Exception:
            return False

    def get_gpu_requirements(self) -> Dict[str, Any]:
        """Returns GPU memory and compute requirements with hardware-specific optimizations."""
        return {
            "memory_requirements": {
                "model": self.hidden_dim * self.num_layers * 4,  # Bytes per parameter
                "workspace": 1024 * 1024 * 1024,  # 1GB workspace
                "buffer": 512 * 1024 * 1024  # 512MB buffer
            },
            "compute_requirements": {
                "min_compute_capability": "vulkan",
                "min_memory_bandwidth": 900,  # GB/s
                "preferred_batch_size": self.performance_constraints["max_batch_size"]
            },
            "optimization_settings": {
                "memory_allocation": "dynamic",
                "compute_precision": "mixed",
                "kernel_optimization": "aggressive"
            }
        }

    def validate_performance(self) -> Dict[str, Any]:
        """Validates configuration against performance requirements."""
        validation_results = {
            "response_time_valid": False,
            "control_accuracy_valid": False,
            "hardware_compatible": False,
            "resource_limits_valid": False,
            "metrics": {}
        }
        
        # Validate response time
        validation_results["response_time_valid"] = (
            self.performance_constraints["max_response_time"] <= 0.05
        )
        
        # Validate control accuracy
        validation_results["control_accuracy_valid"] = (
            self.performance_constraints["min_control_accuracy"] >= 0.5
        )
        
        # Validate hardware compatibility
        validation_results["hardware_compatible"] = all(self.hardware_compatibility.values())
        
        # Validate resource limits
        validation_results["resource_limits_valid"] = (
            self.resource_limits["max_gpu_memory"] >= (
                self.hidden_dim * self.num_layers * 4 +  # Model parameters
                1024 * 1024 * 1024 +  # Workspace
                512 * 1024 * 1024  # Buffer
            )
        )
        
        # Collect metrics
        validation_results["metrics"] = {
            "expected_response_time": self.performance_constraints["max_response_time"] * 1000,
            "expected_control_accuracy": self.performance_constraints["min_control_accuracy"],
            "memory_requirement": self.get_gpu_requirements()["memory_requirements"]
        }
        
        return validation_results

def load_config(config_path: str) -> InstructNetConfig:
    """Loads and validates InstructNet configuration with performance checks."""
    try:
        # Initialize config with default values
        config = InstructNetConfig()
        
        # Validate control settings
        if not config.validate_control_settings():
            raise ValueError("Invalid control settings configuration")
            
        # Validate GPU requirements
        gpu_reqs = config.get_gpu_requirements()
        if gpu_reqs["memory_requirements"]["model"] > config.resource_limits["max_gpu_memory"]:
            raise ValueError("Insufficient GPU memory for model requirements")
            
        # Validate performance constraints
        perf_validation = config.validate_performance()
        if not all([
            perf_validation["response_time_valid"],
            perf_validation["control_accuracy_valid"],
            perf_validation["hardware_compatible"],
            perf_validation["resource_limits_valid"]
        ]):
            raise ValueError("Performance validation failed")
            
        return config
        
    except Exception as e:
        raise RuntimeError(f"Failed to load InstructNet configuration: {str(e)}")