# External imports with versions
import torch  # torch ^2.0.0
import logging  # logging ^3.0.0
import torchmetrics  # torchmetrics ^1.0.0
from performance_monitor import monitor_performance  # performance_monitor ^1.0.0
from typing import Dict, Optional, Any

# Internal imports
from models.msdit.config import MSDiTConfig
from models.msdit.transformer import MSDiTTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Global constants
DEFAULT_CONFIG = MSDiTConfig()
QUALITY_THRESHOLDS = {
    "fid": 300,
    "fvd": 1000,
    "frame_consistency": 0.8
}
PERFORMANCE_TARGETS = {
    "generation_latency": 0.1,  # 100ms
    "frame_rate": 24,  # FPS
    "control_latency": 0.05  # 50ms
}

def error_handler(func):
    """Decorator for error handling and logging."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            LOGGER.error(f"Error in {func.__name__}: {str(e)}")
            torch.cuda.empty_cache()
            raise
    return wrapper

@monitor_performance
@error_handler
def create_model(
    override_config: Optional[Dict[str, Any]] = None,
    enable_monitoring: bool = True
) -> MSDiTTransformer:
    """
    Factory function to create and initialize an MSDiT model instance with FreeBSD optimizations.
    
    Args:
        override_config: Optional configuration overrides
        enable_monitoring: Enable performance monitoring
        
    Returns:
        Initialized MSDiTTransformer instance
    """
    # Initialize configuration with overrides
    config = MSDiTConfig(
        architecture_override=override_config.get("architecture") if override_config else None,
        training_override=override_config.get("training") if override_config else None,
        inference_override=override_config.get("inference") if override_config else None,
        diffusion_override=override_config.get("diffusion") if override_config else None
    )
    
    # Validate FreeBSD compatibility
    if not config.architecture["freebsd_optimization"]["enabled"]:
        raise ValueError("FreeBSD optimization must be enabled")
    
    # Initialize model with validated config
    model = MSDiTTransformer(config)
    
    # Configure performance monitoring
    if enable_monitoring:
        metrics = torchmetrics.MetricCollection([
            torchmetrics.FID(),
            torchmetrics.FVD(),
            torchmetrics.Accuracy(task="multiclass", num_classes=2)
        ])
        model.metrics = metrics
    
    # Move model to configured device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Verify performance targets
    if not _verify_performance(model):
        raise RuntimeError("Model does not meet performance requirements")
    
    LOGGER.info("Model initialized successfully with FreeBSD optimizations")
    return model

@monitor_performance
@error_handler
def load_pretrained(
    checkpoint_path: str,
    verify_quality: bool = True
) -> MSDiTTransformer:
    """
    Loads a pretrained MSDiT model from checkpoint with FreeBSD compatibility checks.
    
    Args:
        checkpoint_path: Path to model checkpoint
        verify_quality: Enable quality metric verification
        
    Returns:
        Loaded pretrained model
    """
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    # Extract configuration
    config = checkpoint.get("config")
    if not config:
        raise ValueError("Checkpoint does not contain model configuration")
    
    # Initialize model with loaded config
    model = create_model(override_config=config, enable_monitoring=verify_quality)
    
    # Load state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Verify quality metrics if enabled
    if verify_quality:
        quality_metrics = checkpoint.get("quality_metrics", {})
        if not _verify_quality(quality_metrics):
            raise ValueError("Model does not meet quality requirements")
    
    LOGGER.info(f"Pretrained model loaded successfully from {checkpoint_path}")
    return model

def _verify_performance(model: MSDiTTransformer) -> bool:
    """Verifies model meets performance targets."""
    metrics = model.performance_metrics
    
    return (
        metrics["generation_latency"] <= PERFORMANCE_TARGETS["generation_latency"] and
        metrics["frames_processed"] / metrics["generation_latency"] >= PERFORMANCE_TARGETS["frame_rate"]
    )

def _verify_quality(metrics: Dict[str, float]) -> bool:
    """Verifies model meets quality thresholds."""
    return (
        metrics.get("fid", float('inf')) <= QUALITY_THRESHOLDS["fid"] and
        metrics.get("fvd", float('inf')) <= QUALITY_THRESHOLDS["fvd"] and
        metrics.get("frame_consistency", 0.0) >= QUALITY_THRESHOLDS["frame_consistency"]
    )

# Export public interface
__all__ = [
    "MSDiTConfig",
    "MSDiTTransformer",
    "create_model",
    "load_pretrained",
    "QUALITY_THRESHOLDS",
    "PERFORMANCE_TARGETS"
]