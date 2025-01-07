# External imports with versions
from pydantic import BaseModel  # pydantic ^2.0.0
from typing import Dict, Any, Optional, Tuple, List  # typing ^3.9.0

# Internal imports
from core.config import Settings, MIN_GPU_MEMORY, DEFAULT_FRAME_RATE

# Global constants for VAE configuration
DEFAULT_LATENT_DIM = 512
DEFAULT_SEQUENCE_LENGTH = 102
DEFAULT_FRAME_RATE = 24
DEFAULT_RESOLUTION = {"width": 1280, "height": 720}

class VAEConfig(BaseModel):
    """
    Configuration class for 3D Spatio-Temporal VAE model with FreeBSD compatibility 
    and non-NVIDIA GPU optimization.
    """
    
    def __init__(self):
        """
        Initializes VAE configuration with architecture, training, and inference parameters
        optimized for FreeBSD.
        """
        super().__init__()
        self.architecture = self.get_architecture_config()
        self.training = self.get_training_config()
        self.inference = self.get_inference_config()
        
        # Validate GPU compatibility and memory requirements
        self._validate_gpu_requirements()
        
        # Initialize caching for frequent operations
        self._setup_caching()
        
        # Configure performance monitoring
        self._setup_monitoring()

    def get_architecture_config(self) -> Dict[str, Any]:
        """
        Returns model architecture configuration with FreeBSD-specific optimizations.
        """
        return {
            "input_dims": {
                "height": DEFAULT_RESOLUTION["height"],
                "width": DEFAULT_RESOLUTION["width"],
                "channels": 3,
                "sequence_length": DEFAULT_SEQUENCE_LENGTH
            },
            "encoder": {
                "conv_layers": [
                    {"filters": 64, "kernel_size": 3, "stride": 2},
                    {"filters": 128, "kernel_size": 3, "stride": 2},
                    {"filters": 256, "kernel_size": 3, "stride": 2},
                    {"filters": 512, "kernel_size": 3, "stride": 2}
                ],
                "temporal_layers": [
                    {"type": "self_attention", "heads": 8, "dim": 512},
                    {"type": "feed_forward", "dim": 2048}
                ],
                "normalization": "layer_norm",
                "activation": "gelu"
            },
            "latent": {
                "dimension": DEFAULT_LATENT_DIM,
                "temporal_reduction": 4,
                "spatial_reduction": 16
            },
            "decoder": {
                "conv_layers": [
                    {"filters": 512, "kernel_size": 3, "stride": 2},
                    {"filters": 256, "kernel_size": 3, "stride": 2},
                    {"filters": 128, "kernel_size": 3, "stride": 2},
                    {"filters": 64, "kernel_size": 3, "stride": 2}
                ],
                "temporal_layers": [
                    {"type": "self_attention", "heads": 8, "dim": 512},
                    {"type": "feed_forward", "dim": 2048}
                ],
                "normalization": "layer_norm",
                "activation": "gelu"
            },
            "freebsd_optimizations": {
                "memory_alignment": 4096,
                "compute_precision": "mixed_float16",
                "kernel_optimization": "vulkan",
                "thread_affinity": "native"
            }
        }

    def get_training_config(self) -> Dict[str, Any]:
        """
        Returns training configuration parameters optimized for memory efficiency.
        """
        return {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999
            },
            "loss_weights": {
                "reconstruction": 1.0,
                "kl_divergence": 0.01,
                "perceptual": 0.1
            },
            "batch_size": {
                "train": 4,
                "validation": 2
            },
            "data_augmentation": {
                "enabled": True,
                "random_crop": True,
                "horizontal_flip": True,
                "color_jitter": 0.1
            },
            "memory_optimization": {
                "gradient_accumulation_steps": 4,
                "mixed_precision": True,
                "checkpoint_frequency": 1000,
                "max_grad_norm": 1.0
            },
            "performance_monitoring": {
                "log_frequency": 100,
                "validation_frequency": 1000,
                "metrics": ["loss", "psnr", "ssim", "fvd"]
            }
        }

    def get_inference_config(self) -> Dict[str, Any]:
        """
        Returns inference configuration parameters with runtime optimizations.
        """
        return {
            "batch_processing": {
                "max_batch_size": 2,
                "dynamic_batching": True,
                "prefetch_factor": 2
            },
            "memory_optimization": {
                "cache_latents": True,
                "optimize_memory_usage": True,
                "clear_cache_frequency": 100
            },
            "frame_generation": {
                "fps": DEFAULT_FRAME_RATE,
                "frame_buffer_size": 24,
                "interpolation_enabled": True
            },
            "hardware_acceleration": {
                "vulkan_compute": True,
                "thread_pinning": True,
                "memory_pool": "shared"
            },
            "error_handling": {
                "auto_recovery": True,
                "max_retries": 3,
                "fallback_resolution": (640, 480)
            },
            "caching": {
                "latent_cache_size": 1000,
                "frame_cache_size": 100,
                "cache_type": "lru"
            }
        }

    def _validate_gpu_requirements(self):
        """Validates GPU compatibility and memory requirements."""
        settings = Settings()
        
        if settings.gpu_settings["min_memory"] < MIN_GPU_MEMORY:
            raise ValueError(f"Insufficient GPU memory. Required: {MIN_GPU_MEMORY / (1024**3)}GB")
        
        if settings.compatibility_flags["gpu_api"] != "vulkan":
            raise ValueError("Vulkan API required for VAE operations")

    def _setup_caching(self):
        """Initializes caching for frequently accessed parameters."""
        self._cache = {
            "latent_projections": {},
            "temporal_attention": {},
            "frame_buffer": []
        }

    def _setup_monitoring(self):
        """Configures performance monitoring hooks."""
        self._monitors = {
            "memory_usage": [],
            "computation_time": [],
            "frame_latency": [],
            "cache_hits": 0,
            "cache_misses": 0
        }

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"