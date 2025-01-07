# External imports with versions
import torch  # v2.0.0
import wandb  # v0.15.0
from typing import Dict, Any, Optional, List
import logging
from functools import wraps

# Internal imports
from utils.gpu import GPUManager
from core.metrics import MetricsCollector
from core.logging import get_logger

# Configure logging
logger = get_logger(__name__)

# Script type definitions
SCRIPT_TYPES = [
    "benchmark",
    "train_vae",
    "train_msdit", 
    "train_instructnet",
    "export_models",
    "validate_freebsd"
]

# GPU configuration for H800 optimization
GPU_CONFIG = {
    "min_memory": "24GB",
    "preferred_memory": "80GB",
    "required_compute": 8.0,
    "freebsd_specific": {
        "jail_config": "gpu.allow",
        "driver_version": "latest",
        "h800_optimization": True
    }
}

def validate_script_type(func):
    """Decorator to validate script type against allowed types"""
    @wraps(func)
    def wrapper(script_type: str, *args, **kwargs):
        if script_type not in SCRIPT_TYPES:
            raise ValueError(f"Invalid script type: {script_type}. Must be one of {SCRIPT_TYPES}")
        return func(script_type, *args, **kwargs)
    return wrapper

@MetricsCollector(metric_type='gpu_setup', jail_aware=True)
def setup_gpu_environment(script_type: str, freebsd_config: Dict[str, Any]) -> GPUManager:
    """Initializes FreeBSD-compatible GPU environment with H800 optimization"""
    try:
        # Initialize GPU manager with FreeBSD configuration
        gpu_manager = GPUManager(GPU_CONFIG, freebsd_config)
        
        # Apply H800-specific optimizations
        gpu_manager.optimize_for_freebsd({
            "compute_mode": "maximum",
            "memory_bandwidth": "high",
            "cache_mode": "prefer_shared",
            "jail_isolation": True
        })
        
        # Configure CUDA environment for FreeBSD
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Verify GPU configuration
        if not gpu_manager.validate_hardware():
            raise RuntimeError("GPU hardware validation failed")
            
        logger.info(f"GPU environment initialized for script type: {script_type}")
        return gpu_manager
        
    except Exception as e:
        logger.error(f"Failed to setup GPU environment: {str(e)}")
        raise

@MetricsCollector(metric_type='gpu_cleanup', jail_aware=True)
def cleanup_gpu_environment(gpu_manager: GPUManager, jail_config: Dict[str, Any]) -> bool:
    """Cleans up GPU resources with FreeBSD-specific handling"""
    try:
        # Release GPU memory
        torch.cuda.empty_cache()
        
        # Cleanup GPU manager resources
        gpu_manager.cleanup()
        
        # Reset jail GPU permissions
        if jail_config.get("gpu.allow"):
            gpu_manager.reset_jail_permissions(jail_config)
            
        logger.info("GPU environment cleanup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to cleanup GPU environment: {str(e)}")
        return False

class ScriptContext:
    """Context manager for script execution environment with FreeBSD jail support"""
    
    def __init__(self, script_type: str, jail_config: Dict[str, Any]):
        """Initialize script context with FreeBSD-specific configuration"""
        if script_type not in SCRIPT_TYPES:
            raise ValueError(f"Invalid script type: {script_type}")
            
        self.script_type = script_type
        self.jail_config = jail_config
        self.gpu_manager = None
        self.freebsd_metrics = {
            "jail_id": jail_config.get("jail_id"),
            "gpu_allocation": jail_config.get("gpu_allocation", {}),
            "resource_limits": jail_config.get("resource_limits", {})
        }
        
        logger.info(f"Initialized script context for {script_type}")

    def __enter__(self) -> 'ScriptContext':
        """Setup script execution environment with jail awareness"""
        try:
            # Initialize GPU environment
            self.gpu_manager = setup_gpu_environment(
                self.script_type,
                self.jail_config
            )
            
            # Initialize wandb logging with jail metrics
            wandb.init(
                project="gamegen-x",
                config={
                    "script_type": self.script_type,
                    "jail_config": self.jail_config,
                    "gpu_config": GPU_CONFIG
                },
                tags=[self.script_type, "freebsd", "h800"]
            )
            
            # Verify H800 optimization status
            optimization_status = self.gpu_manager.get_optimization_status()
            if not optimization_status.get("h800_optimized"):
                logger.warning("H800-specific optimizations not fully applied")
                
            return self
            
        except Exception as e:
            logger.error(f"Failed to setup script context: {str(e)}")
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup script execution environment and jail resources"""
        try:
            # Cleanup GPU environment
            cleanup_gpu_environment(self.gpu_manager, self.jail_config)
            
            # Finish wandb logging
            wandb.finish()
            
            # Export final jail metrics
            if exc_type is None:
                logger.info(
                    "Script completed successfully",
                    extra={"jail_metrics": self.freebsd_metrics}
                )
            else:
                logger.error(
                    f"Script failed: {str(exc_value)}",
                    extra={
                        "jail_metrics": self.freebsd_metrics,
                        "error_type": exc_type.__name__
                    }
                )
                
        except Exception as e:
            logger.error(f"Error during context cleanup: {str(e)}")
            raise

# Export public interface
__all__ = [
    'ScriptContext',
    'setup_gpu_environment',
    'cleanup_gpu_environment',
    'SCRIPT_TYPES',
    'GPU_CONFIG'
]