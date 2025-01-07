# External imports with versions
import torch  # ^2.0.0
import wandb  # ^0.15.0
import tqdm  # ^4.65.0
import numpy as np  # ^1.23.0
from typing import Dict, Tuple, Any

# Internal imports
from models.msdit.config import MSDiTConfig
from models.msdit.transformer import MSDiTTransformer
from core.metrics import MetricsCollector

# Global constants for bucket training
BUCKET_SCHEDULE = {'320x256': 0.4, '848x480': 0.4, '1280x720': 0.2}
MAX_FRAMES = 102
BATCH_SIZE = {'320x256': 32, '848x480': 16, '1280x720': 8}
FREEBSD_GPU_CONFIG = {
    'memory_fraction': 0.9,
    'cache_mode': 'memory_efficient',
    'optimization_level': 'O3'
}

def setup_training(config: MSDiTConfig) -> Tuple[MSDiTTransformer, torch.optim.Optimizer, 
                                                torch.optim.lr_scheduler._LRScheduler, MetricsCollector]:
    """Initializes FreeBSD-optimized training environment."""
    
    # Configure FreeBSD GPU settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # Initialize wandb logging
    wandb.init(
        project="gamegen-x",
        config={
            "architecture": config.get_architecture_config(),
            "training": config.get_training_config(),
            "freebsd_optimization": FREEBSD_GPU_CONFIG
        }
    )
    
    # Initialize model with FreeBSD optimizations
    model = MSDiTTransformer(config)
    model.cuda()
    model = torch.compile(model, mode='reduce-overhead')
    
    # Configure optimizer with memory efficiency
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get_training_config()["optimizer"]["learning_rate"],
        weight_decay=config.get_training_config()["optimizer"]["weight_decay"],
        betas=(config.get_training_config()["optimizer"]["beta1"],
               config.get_training_config()["optimizer"]["beta2"])
    )
    
    # Initialize scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.get_training_config()["scheduler"]["warmup_steps"],
        T_mult=2
    )
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    
    return model, optimizer, scheduler, metrics_collector

def train_epoch(model: MSDiTTransformer,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                dataloader: torch.utils.data.DataLoader,
                metrics_collector: MetricsCollector) -> Dict[str, float]:
    """Executes one epoch of FreeBSD-optimized training."""
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Initialize progress bar
    pbar = tqdm.tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        try:
            # Get current resolution bucket
            resolution = np.random.choice(
                list(BUCKET_SCHEDULE.keys()),
                p=list(BUCKET_SCHEDULE.values())
            )
            current_batch_size = BATCH_SIZE[resolution]
            
            # Process batch with memory optimization
            with torch.cuda.amp.autocast():
                # Apply classifier-free guidance
                guidance_mask = torch.rand(current_batch_size) > 0.1
                
                # Forward pass
                output = model(batch["video"], guidance_mask)
                loss = model.compute_loss(output, batch["target"])
                
                # Scale loss for gradient accumulation
                loss = loss / 8  # Accumulate over 8 steps
            
            # Backward pass with gradient scaling
            loss.backward()
            
            # Gradient accumulation step
            if (num_batches + 1) % 8 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Record metrics
            metrics_collector.record_metric("training_loss", loss.item())
            metrics_collector.record_metric("learning_rate", scheduler.get_last_lr()[0])
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Validate quality metrics every 1000 batches
            if num_batches % 1000 == 0:
                quality_metrics = validate(model, dataloader, metrics_collector)
                wandb.log({
                    "fid_score": quality_metrics["fid"],
                    "fvd_score": quality_metrics["fvd"],
                    "frame_rate": quality_metrics["fps"]
                })
                
        except RuntimeError as e:
            print(f"Error in batch: {str(e)}")
            torch.cuda.empty_cache()
            continue
    
    return {
        "loss": total_loss / num_batches,
        "batches": num_batches
    }

def validate(model: MSDiTTransformer,
            val_dataloader: torch.utils.data.DataLoader,
            metrics_collector: MetricsCollector) -> Dict[str, float]:
    """Performs comprehensive quality validation."""
    
    model.eval()
    metrics = {
        "fid": 0.0,
        "fvd": 0.0,
        "fps": 0.0,
        "success_rate": 0.0
    }
    
    with torch.no_grad():
        for batch in val_dataloader:
            try:
                # Generate validation videos
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                generated = model.generate(batch["condition"], MAX_FRAMES)
                end_time.record()
                
                torch.cuda.synchronize()
                generation_time = start_time.elapsed_time(end_time)
                
                # Calculate quality metrics
                metrics["fid"] += model._compute_fid(generated)
                metrics["fvd"] += model._compute_fvd(generated)
                metrics["fps"] += (MAX_FRAMES * 1000) / generation_time
                
                # Check success criteria
                if (metrics["fid"] < 300 and metrics["fvd"] < 1000 and 
                    metrics["fps"] >= 24):
                    metrics["success_rate"] += 1
                    
            except RuntimeError as e:
                print(f"Validation error: {str(e)}")
                continue
    
    # Average metrics
    num_batches = len(val_dataloader)
    for key in metrics:
        metrics[key] /= num_batches
        
    # Record validation metrics
    metrics_collector.record_metric("validation_fid", metrics["fid"])
    metrics_collector.record_metric("validation_fvd", metrics["fvd"])
    metrics_collector.record_metric("validation_fps", metrics["fps"])
    
    return metrics

def main():
    """Main training execution with FreeBSD compatibility."""
    
    try:
        # Initialize configuration
        config = MSDiTConfig()
        
        # Setup training environment
        model, optimizer, scheduler, metrics_collector = setup_training(config)
        
        # Training loop
        num_epochs = config.get_training_config()["training"]["num_epochs"]
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = train_epoch(
                model, optimizer, scheduler, train_dataloader, metrics_collector
            )
            
            # Validate
            val_metrics = validate(model, val_dataloader, metrics_collector)
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_fid": val_metrics["fid"],
                "val_fvd": val_metrics["fvd"],
                "val_fps": val_metrics["fps"],
                "val_success_rate": val_metrics["success_rate"]
            })
            
            # Save checkpoint
            if val_metrics["success_rate"] > 0.8:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "metrics": val_metrics
                }, f"checkpoints/msdit_epoch_{epoch}.pt")
                
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()