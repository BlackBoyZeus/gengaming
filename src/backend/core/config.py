# External imports with versions
from pydantic import BaseSettings, Field, validator  # pydantic ^2.0.0
from dotenv import load_dotenv  # python-dotenv ^1.0.0
from typing import Dict, Any, Optional  # typing ^3.9.0
import os
import json
import logging
from pathlib import Path

# Global constants
PROJECT_NAME = "GameGen-X"
ENVIRONMENT_FILE = ".env"
DEFAULT_MODEL_PATH = "models/weights"
MIN_GPU_MEMORY = 24 * 1024 * 1024 * 1024  # 24GB in bytes
MAX_BATCH_SIZE = 4
DEFAULT_FRAME_RATE = 24

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Comprehensive configuration class managing system-wide settings with validation.
    Implements FreeBSD compatibility, non-NVIDIA GPU optimization, and performance requirements.
    """
    
    # Core settings
    environment: str = Field(default="development", regex="^(development|staging|production)$")
    debug: bool = Field(default=False)
    project_name: str = Field(default=PROJECT_NAME)
    model_path: str = Field(default=DEFAULT_MODEL_PATH)
    
    # GPU settings for non-NVIDIA hardware
    gpu_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "min_memory": MIN_GPU_MEMORY,
        "compute_units": "auto",
        "memory_bandwidth": "high",
        "driver_version": "latest",
        "optimization_level": "aggressive",
        "thermal_limit": 85,
        "power_limit": "auto"
    })
    
    # FreeBSD compatibility settings
    freebsd_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "os_version": "13.0",
        "compatibility_layer": "native",
        "sysctls": {
            "kern.ipc.shm_max": 67108864,
            "kern.ipc.shm_use_phys": 1,
            "hw.nvidia.registry.SoftwareOnly": 0
        },
        "jail_parameters": {
            "allow.raw_sockets": 1,
            "allow.sysvipc": 1
        }
    })
    
    # Resource limits
    resource_limits: Dict[str, int] = Field(default_factory=lambda: {
        "max_memory": 512 * 1024 * 1024 * 1024,  # 512GB
        "max_gpu_memory": MIN_GPU_MEMORY,
        "max_storage": 1024 * 1024 * 1024 * 1024,  # 1TB
        "max_batch_size": MAX_BATCH_SIZE
    })
    
    # Performance thresholds
    performance_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "max_generation_latency": 0.1,  # 100ms
        "min_frame_rate": DEFAULT_FRAME_RATE,
        "max_control_latency": 0.05,  # 50ms
        "min_gpu_memory_bandwidth": 900.0  # GB/s
    })
    
    # Compatibility flags
    compatibility_flags: Dict[str, str] = Field(default_factory=lambda: {
        "gpu_api": "vulkan",
        "memory_allocator": "jemalloc",
        "threading_model": "native"
    })

    class Config:
        env_file = ENVIRONMENT_FILE
        case_sensitive = True
        validate_assignment = True

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize core system settings with comprehensive validation."""
        super().__init__(**kwargs)
        
        # Load environment variables
        load_dotenv(self.Config.env_file)
        
        # Apply settings override if provided
        if settings_override:
            for key, value in settings_override.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # Validate complete configuration
        self.validate_gpu_settings()
        self._validate_freebsd_compatibility()
        self._validate_performance_requirements()

    @validator("gpu_settings")
    def validate_gpu_settings(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive GPU configuration validation."""
        if v["min_memory"] < MIN_GPU_MEMORY:
            raise ValueError(f"GPU memory must be at least {MIN_GPU_MEMORY / (1024**3)}GB")
        
        # Validate GPU compatibility
        required_features = ["vulkan", "compute-capability", "memory-bandwidth"]
        for feature in required_features:
            if not cls._check_gpu_feature(feature):
                raise ValueError(f"Required GPU feature not available: {feature}")
        
        return v

    def get_resource_limits(self) -> Dict[str, Dict[str, int]]:
        """Retrieves and validates current resource limits."""
        current_limits = {
            "memory": self._get_memory_usage(),
            "gpu": self._get_gpu_usage(),
            "storage": self._get_storage_usage(),
            "compute": self._get_compute_usage()
        }
        
        # Validate against configured limits
        for resource, usage in current_limits.items():
            if usage["current"] > usage["limit"]:
                logger.warning(f"Resource limit exceeded for {resource}")
        
        return current_limits

    def _validate_freebsd_compatibility(self):
        """Validates FreeBSD compatibility settings."""
        if not os.path.exists("/usr/sbin/sysctl"):
            raise RuntimeError("FreeBSD sysctl not found")
        
        # Verify system parameters
        for sysctl, value in self.freebsd_settings["sysctls"].items():
            current = os.system(f"sysctl -n {sysctl}")
            if current != value:
                logger.warning(f"Sysctl {sysctl} value mismatch: expected {value}, got {current}")

    def _validate_performance_requirements(self):
        """Validates system performance requirements."""
        if not self._check_generation_latency():
            raise ValueError("System cannot meet generation latency requirements")
        if not self._check_frame_rate():
            raise ValueError("System cannot meet frame rate requirements")

    @staticmethod
    def _check_gpu_feature(feature: str) -> bool:
        """Checks for specific GPU feature availability."""
        # Implementation would check actual hardware capabilities
        return True  # Placeholder

    def _get_memory_usage(self) -> Dict[str, int]:
        """Gets current memory usage statistics."""
        return {
            "current": 0,  # Placeholder
            "limit": self.resource_limits["max_memory"]
        }

    def _get_gpu_usage(self) -> Dict[str, int]:
        """Gets current GPU memory usage statistics."""
        return {
            "current": 0,  # Placeholder
            "limit": self.resource_limits["max_gpu_memory"]
        }

    def _get_storage_usage(self) -> Dict[str, int]:
        """Gets current storage usage statistics."""
        return {
            "current": 0,  # Placeholder
            "limit": self.resource_limits["max_storage"]
        }

    def _get_compute_usage(self) -> Dict[str, int]:
        """Gets current compute resource usage statistics."""
        return {
            "current": 0,  # Placeholder
            "limit": self.resource_limits["max_batch_size"]
        }

def load_settings(
    env_file: Optional[str] = None,
    override_settings: Optional[Dict[str, Any]] = None
) -> Settings:
    """Loads and validates application settings with comprehensive checks."""
    try:
        settings = Settings(
            settings_override=override_settings,
            env_file=env_file or ENVIRONMENT_FILE
        )
        logger.info("Settings loaded successfully")
        return settings
    except Exception as e:
        logger.error(f"Failed to load settings: {str(e)}")
        raise

def get_gpu_config(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
    """Generates optimized GPU configuration for non-NVIDIA hardware."""
    config = {
        "api": "vulkan",
        "memory": {
            "allocation": "dynamic",
            "min_guaranteed": MIN_GPU_MEMORY,
            "preferred_batch": MAX_BATCH_SIZE
        },
        "compute": {
            "threads": hardware_info.get("compute_units", "auto"),
            "optimization": "aggressive",
            "precision": "mixed"
        },
        "performance": {
            "thermal_limit": 85,
            "power_mode": "performance",
            "memory_bandwidth": "maximum"
        }
    }
    return config

# Create singleton instance
settings = load_settings()

# Export settings instance and classes
__all__ = ["settings", "Settings", "load_settings", "get_gpu_config"]