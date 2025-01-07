# External imports with versions
import torch  # ^2.0.0
import argparse  # ^3.9.0
import asyncio  # ^3.9.0
import logging  # ^3.9.0
import psutil  # ^5.9.0
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional

# Internal imports
from models.vae.config import VAEConfig
from models.msdit.config import MSDiTConfig
from models.instructnet.config import InstructNetConfig
from services.storage import StorageService
from core.exceptions import ModelError, FreeBSDError
from core.logging import get_logger
from core.metrics import track_generation_latency

# Global constants
SUPPORTED_MODELS = ["vae", "msdit", "instructnet"]
MODEL_EXPORT_PATH = "models/exported"
FREEBSD_OPTIMIZATION_CONFIG = {
    "memory_format": "channels_last",
    "cuda_graphs": False,
    "inference_mode": True
}
EXPORT_TIMEOUT_SECONDS = 3600

# Initialize logger
logger = get_logger(__name__)

def timeout(seconds):
    """Decorator for enforcing timeout on async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise ModelError(
                    message=f"Export operation timed out after {seconds} seconds",
                    model_name=args[0] if args else "unknown",
                    model_context={"timeout": seconds},
                    original_error=None
                )
        return wrapper
    return decorator

@timeout(EXPORT_TIMEOUT_SECONDS)
async def export_vae(checkpoint_path: str, output_path: str, optimization_config: Dict[str, Any]) -> str:
    """Exports VAE model with FreeBSD optimization and TorchScript conversion"""
    try:
        # Initialize VAE config and validate FreeBSD compatibility
        vae_config = VAEConfig()
        if not vae_config.validate_freebsd_compatibility():
            raise FreeBSDError(
                message="VAE model not compatible with FreeBSD",
                operation="export_vae",
                system_context={"config": vae_config},
            )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = checkpoint["model"]
        
        # Apply FreeBSD optimizations
        model.eval()
        model = model.to(memory_format=torch.channels_last)
        for param in model.parameters():
            param.requires_grad_(False)

        # Convert to TorchScript with optimization
        with torch.inference_mode():
            scripted_model = torch.jit.script(model)
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        # Export model and config
        export_path = Path(output_path) / "vae_model.pt"
        torch.jit.save(scripted_model, str(export_path))
        
        # Store with backup verification
        storage = StorageService(endpoint="localhost", access_key="", secret_key="")
        await storage.store_model("vae", export_path.read_bytes())
        await storage.verify_backup(str(export_path))
        
        # Cleanup temporary files
        await storage.cleanup_temp_files()
        
        logger.info(f"Successfully exported VAE model to {export_path}")
        return str(export_path)

    except Exception as e:
        raise ModelError(
            message=f"Failed to export VAE model: {str(e)}",
            model_name="vae",
            model_context={"checkpoint": checkpoint_path},
            original_error=e
        )

@timeout(EXPORT_TIMEOUT_SECONDS)
async def export_msdit(checkpoint_path: str, output_path: str, optimization_config: Dict[str, Any]) -> str:
    """Exports MSDiT model with FreeBSD compatibility and performance optimization"""
    try:
        # Initialize MSDiT config
        msdit_config = MSDiTConfig()
        if not msdit_config.optimize_for_freebsd():
            raise FreeBSDError(
                message="MSDiT model optimization failed for FreeBSD",
                operation="export_msdit",
                system_context={"config": msdit_config},
            )

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = checkpoint["model"]
        
        # Apply FreeBSD optimizations
        model.eval()
        model = model.to(memory_format=torch.channels_last)
        torch._C._jit_set_profiling_executor(False)
        
        # Convert to TorchScript with performance optimization
        with torch.inference_mode():
            scripted_model = torch.jit.script(model)
            scripted_model = torch.jit.optimize_for_inference(scripted_model)
            
        # Validate performance requirements
        if not msdit_config.validate_performance():
            raise ModelError(
                message="Model does not meet performance requirements",
                model_name="msdit",
                model_context={"performance": msdit_config.get_inference_config()},
            )

        # Export model and configs
        export_path = Path(output_path) / "msdit_model.pt"
        torch.jit.save(scripted_model, str(export_path))
        
        # Store with backup verification
        storage = StorageService(endpoint="localhost", access_key="", secret_key="")
        await storage.store_model("msdit", export_path.read_bytes())
        await storage.verify_backup(str(export_path))
        
        # Cleanup temporary files
        await storage.cleanup_temp_files()
        
        logger.info(f"Successfully exported MSDiT model to {export_path}")
        return str(export_path)

    except Exception as e:
        raise ModelError(
            message=f"Failed to export MSDiT model: {str(e)}",
            model_name="msdit",
            model_context={"checkpoint": checkpoint_path},
            original_error=e
        )

@timeout(EXPORT_TIMEOUT_SECONDS)
async def export_instructnet(checkpoint_path: str, output_path: str, optimization_config: Dict[str, Any]) -> str:
    """Exports InstructNet model with FreeBSD optimization and latency requirements"""
    try:
        # Initialize InstructNet config
        instruct_config = InstructNetConfig()
        if not instruct_config.freebsd_settings["enabled"]:
            raise FreeBSDError(
                message="InstructNet not configured for FreeBSD",
                operation="export_instructnet",
                system_context={"config": instruct_config},
            )

        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = checkpoint["model"]
        
        # Apply FreeBSD optimizations
        model.eval()
        model = model.to(memory_format=torch.channels_last)
        for param in model.parameters():
            param.requires_grad_(False)

        # Convert to TorchScript with latency optimization
        with torch.inference_mode():
            scripted_model = torch.jit.script(model)
            scripted_model = torch.jit.optimize_for_inference(scripted_model)

        # Export model and config
        export_path = Path(output_path) / "instructnet_model.pt"
        torch.jit.save(scripted_model, str(export_path))
        
        # Store with backup verification
        storage = StorageService(endpoint="localhost", access_key="", secret_key="")
        await storage.store_model("instructnet", export_path.read_bytes())
        await storage.verify_backup(str(export_path))
        
        # Cleanup temporary files
        await storage.cleanup_temp_files()
        
        logger.info(f"Successfully exported InstructNet model to {export_path}")
        return str(export_path)

    except Exception as e:
        raise ModelError(
            message=f"Failed to export InstructNet model: {str(e)}",
            model_name="instructnet",
            model_context={"checkpoint": checkpoint_path},
            original_error=e
        )

async def main():
    """Main script execution with enhanced error handling and monitoring"""
    parser = argparse.ArgumentParser(description="Export models with FreeBSD optimization")
    parser.add_argument("--model", type=str, required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=MODEL_EXPORT_PATH)
    args = parser.parse_args()

    try:
        # Initialize system monitoring
        start_memory = psutil.Process().memory_info().rss
        start_time = asyncio.get_event_loop().time()

        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export model based on type
        export_func = {
            "vae": export_vae,
            "msdit": export_msdit,
            "instructnet": export_instructnet
        }[args.model]

        exported_path = await export_func(
            args.checkpoint,
            str(output_path),
            FREEBSD_OPTIMIZATION_CONFIG
        )

        # Log performance metrics
        end_time = asyncio.get_event_loop().time()
        end_memory = psutil.Process().memory_info().rss
        duration = end_time - start_time
        memory_used = (end_memory - start_memory) / (1024 * 1024)  # MB

        logger.info(
            f"Export completed successfully",
            extra={
                "model": args.model,
                "duration_seconds": duration,
                "memory_mb": memory_used,
                "exported_path": exported_path
            }
        )

    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())