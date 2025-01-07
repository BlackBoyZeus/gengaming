# External imports with versions
from pydantic import BaseModel  # pydantic ^2.0.0
from typing import Dict, Any, Optional  # typing ^3.9.0

# Internal imports
from core.config import Settings
from models.vae.config import VAEConfig

# Global constants
DEFAULT_NUM_LAYERS = 24
DEFAULT_HIDDEN_DIM = 1024
DEFAULT_NUM_HEADS = 16
DEFAULT_DROPOUT = 0.1
DEFAULT_ATTENTION_HEAD_DIM = 64
DEFAULT_MEMORY_CHUNK_SIZE = '256MB'
DEFAULT_FREEBSD_OPTIMIZATION = True
DEFAULT_QUALITY_THRESHOLD_FID = 300
DEFAULT_QUALITY_THRESHOLD_FVD = 1000

class MSDiTConfig(BaseModel):
    """
    Configuration class for Masked Spatial-Temporal Diffusion Transformer model 
    with FreeBSD optimization and performance validation.
    """

    def __init__(
        self,
        architecture_override: Optional[Dict[str, Any]] = None,
        training_override: Optional[Dict[str, Any]] = None,
        inference_override: Optional[Dict[str, Any]] = None,
        diffusion_override: Optional[Dict[str, Any]] = None
    ):
        """Initialize MSDiT configuration with comprehensive validation."""
        super().__init__()
        
        # Initialize core settings and validate FreeBSD compatibility
        self._settings = Settings()
        self._settings.validate_freebsd_compatibility()
        
        # Initialize VAE configuration for latent validation
        self._vae_config = VAEConfig()
        
        # Initialize configuration dictionaries
        self.architecture = self.get_architecture_config()
        self.training = self.get_training_config()
        self.inference = self.get_inference_config()
        self.diffusion = self.get_diffusion_config()
        
        # Apply overrides if provided
        if architecture_override:
            self.architecture.update(architecture_override)
        if training_override:
            self.training.update(training_override)
        if inference_override:
            self.inference.update(inference_override)
        if diffusion_override:
            self.diffusion.update(diffusion_override)
            
        # Validate final configuration
        self.validate_performance()

    def get_architecture_config(self) -> Dict[str, Any]:
        """Returns model architecture configuration with FreeBSD optimizations."""
        return {
            "layers": {
                "num_layers": DEFAULT_NUM_LAYERS,
                "hidden_dim": DEFAULT_HIDDEN_DIM,
                "num_heads": DEFAULT_NUM_HEADS,
                "head_dim": DEFAULT_ATTENTION_HEAD_DIM,
                "dropout": DEFAULT_DROPOUT,
                "activation": "gelu"
            },
            "attention": {
                "spatial": {
                    "type": "masked_self_attention",
                    "causal": True,
                    "chunk_size": DEFAULT_MEMORY_CHUNK_SIZE
                },
                "temporal": {
                    "type": "global_attention",
                    "max_positions": 102,
                    "relative_pos_embeddings": True
                }
            },
            "memory_optimization": {
                "gradient_checkpointing": True,
                "attention_memory_efficient": True,
                "mixed_precision": True,
                "static_shapes": True
            },
            "freebsd_optimization": {
                "enabled": DEFAULT_FREEBSD_OPTIMIZATION,
                "thread_affinity": "native",
                "memory_allocation": "jemalloc",
                "gpu_api": "vulkan"
            }
        }

    def get_training_config(self) -> Dict[str, Any]:
        """Returns training configuration with quality metric validation."""
        return {
            "optimizer": {
                "type": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.01,
                "beta1": 0.9,
                "beta2": 0.999
            },
            "scheduler": {
                "type": "cosine",
                "warmup_steps": 1000,
                "max_steps": 100000
            },
            "quality_validation": {
                "fid_threshold": DEFAULT_QUALITY_THRESHOLD_FID,
                "fvd_threshold": DEFAULT_QUALITY_THRESHOLD_FVD,
                "validation_frequency": 1000,
                "early_stopping_patience": 5
            },
            "data": {
                "batch_size": 4,
                "sequence_length": 102,
                "resolution": {
                    "height": 720,
                    "width": 1280
                }
            },
            "classifier_free_guidance": {
                "enabled": True,
                "guidance_scale": 7.5,
                "dropout_rate": 0.1
            }
        }

    def get_inference_config(self) -> Dict[str, Any]:
        """Returns inference configuration with performance validation."""
        return {
            "sampling": {
                "steps": 50,
                "strategy": "ddim",
                "eta": 0.0,
                "guidance_scale": 7.5
            },
            "performance": {
                "max_latency_ms": 100,
                "target_fps": 24,
                "batch_size": 1,
                "dynamic_batching": True
            },
            "memory": {
                "max_video_length": 102,
                "frame_buffer_size": 24,
                "cache_latents": True,
                "optimize_attention": True
            },
            "output": {
                "format": "rgb",
                "resolution": {
                    "height": 720,
                    "width": 1280
                },
                "interpolation": "bilinear"
            }
        }

    def get_diffusion_config(self) -> Dict[str, Any]:
        """Returns optimized diffusion process configuration."""
        return {
            "schedule": {
                "type": "cosine",
                "num_steps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02
            },
            "sampling": {
                "method": "ddim",
                "steps": 50,
                "eta": 0.0,
                "clip_denoised": True
            },
            "optimization": {
                "parallel_sampling": True,
                "memory_efficient": True,
                "use_fp16": True,
                "compile_unet": True
            },
            "freebsd_settings": {
                "compute_precision": "mixed_float16",
                "memory_allocation": "jemalloc",
                "thread_pinning": True
            }
        }

    def validate_performance(self) -> bool:
        """Validates configuration meets performance requirements."""
        try:
            # Validate latency requirements
            if self.inference["performance"]["max_latency_ms"] > 100:
                return False
                
            # Validate FPS capability
            if self.inference["performance"]["target_fps"] < 24:
                return False
                
            # Validate memory optimization
            if not self.architecture["memory_optimization"]["gradient_checkpointing"]:
                return False
                
            # Validate FreeBSD compatibility
            if not self.architecture["freebsd_optimization"]["enabled"]:
                return False
                
            # Validate quality thresholds
            if (self.training["quality_validation"]["fid_threshold"] > DEFAULT_QUALITY_THRESHOLD_FID or
                self.training["quality_validation"]["fvd_threshold"] > DEFAULT_QUALITY_THRESHOLD_FVD):
                return False
                
            return True
            
        except Exception as e:
            raise ValueError(f"Performance validation failed: {str(e)}")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "forbid"