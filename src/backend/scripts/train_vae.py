# External imports with versions
import torch  # torch==2.0.0
import wandb  # wandb==0.15.0
import numpy as np  # numpy==1.23.0
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Internal imports
from models.vae.config import VAEConfig
from utils.gpu import GPUManager
from utils.metrics import PerformanceMetrics
from core.logging import get_logger, LoggerContextManager

# Configure logging
logger = get_logger(__name__)

# Global constants
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_NUM_EPOCHS = 100
DEFAULT_SEQUENCE_LENGTH = 102
FREEBSD_GPU_SETTINGS = {
    'memory_fraction': 0.9,
    'enable_optimization': True,
    'cleanup_interval': 10
}

def parse_args():
    """Parse command line arguments for VAE training."""
    parser = argparse.ArgumentParser(description='Train 3D Spatio-Temporal VAE')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument('--sequence_length', type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default='gamegen-x-vae')
    return parser.parse_args()

def initialize_training(config: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    """Initialize training environment with FreeBSD-specific optimizations."""
    with LoggerContextManager("training_initialization", __name__) as log_ctx:
        # Initialize VAE configuration
        vae_config = VAEConfig()
        arch_config = vae_config.get_architecture_config()
        train_config = vae_config.get_training_config()

        # Initialize GPU manager with FreeBSD optimizations
        gpu_manager = GPUManager(FREEBSD_GPU_SETTINGS)
        gpu_manager.allocate_memory(train_config['memory_optimization'])
        gpu_manager.optimize_performance({
            'compute_units': 'max',
            'memory_mode': 'high_bandwidth'
        })

        # Initialize models
        encoder = create_encoder(arch_config)
        decoder = create_decoder(arch_config)

        # Configure optimizer with memory optimization
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=config['lr'],
            weight_decay=train_config['optimizer']['weight_decay'],
            betas=(train_config['optimizer']['beta1'], 
                  train_config['optimizer']['beta2'])
        )

        # Initialize performance metrics
        metrics = PerformanceMetrics("vae_training_jail")

        return encoder, decoder, optimizer, metrics

def train_epoch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """Execute one training epoch with enhanced monitoring and error handling."""
    with LoggerContextManager("train_epoch", __name__) as log_ctx:
        epoch_metrics = {
            'loss': 0.0,
            'reconstruction_loss': 0.0,
            'kl_loss': 0.0,
            'perceptual_loss': 0.0
        }
        
        encoder.train()
        decoder.train()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Monitor GPU memory
                if batch_idx % FREEBSD_GPU_SETTINGS['cleanup_interval'] == 0:
                    torch.cuda.empty_cache()
                
                # Process batch with memory optimization
                videos = batch['video'].to('cuda', memory_format=torch.channels_last)
                
                # Forward pass with gradient accumulation
                optimizer.zero_grad()
                
                # Encode
                mu, log_var = encoder(videos)
                z = reparameterize(mu, log_var)
                
                # Decode
                reconstructed = decoder(z)
                
                # Calculate losses
                reconstruction_loss = torch.nn.functional.mse_loss(
                    reconstructed, videos, reduction='mean'
                )
                kl_loss = -0.5 * torch.mean(
                    1 + log_var - mu.pow(2) - log_var.exp()
                )
                perceptual_loss = calculate_perceptual_loss(
                    reconstructed, videos
                )
                
                # Total loss
                loss = (
                    config['loss_weights']['reconstruction'] * reconstruction_loss +
                    config['loss_weights']['kl_divergence'] * kl_loss +
                    config['loss_weights']['perceptual'] * perceptual_loss
                )
                
                # Backward pass with gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    config['memory_optimization']['max_grad_norm']
                )
                optimizer.step()
                
                # Update metrics
                epoch_metrics['loss'] += loss.item()
                epoch_metrics['reconstruction_loss'] += reconstruction_loss.item()
                epoch_metrics['kl_loss'] += kl_loss.item()
                epoch_metrics['perceptual_loss'] += perceptual_loss.item()
                
                # Log batch metrics
                if batch_idx % config['performance_monitoring']['log_frequency'] == 0:
                    log_batch_metrics(epoch_metrics, batch_idx, len(dataloader))
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    handle_oom_error(optimizer)
                    continue
                raise e
        
        # Average metrics
        num_batches = len(dataloader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics

def handle_training_error(error: Exception, training_state: Dict[str, Any]) -> bool:
    """Handle training errors with automatic recovery mechanisms."""
    with LoggerContextManager("error_handling", __name__) as log_ctx:
        try:
            logger.error(f"Training error occurred: {str(error)}")
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Attempt to reduce batch size if OOM
            if isinstance(error, RuntimeError) and "out of memory" in str(error):
                new_batch_size = training_state['batch_size'] // 2
                if new_batch_size >= 1:
                    training_state['batch_size'] = new_batch_size
                    logger.info(f"Reduced batch size to {new_batch_size}")
                    return True
            
            # Load last checkpoint if available
            if training_state.get('last_checkpoint'):
                load_checkpoint(training_state['last_checkpoint'])
                logger.info("Restored from last checkpoint")
                return True
            
            return False
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
            return False

def main():
    """Main training loop with comprehensive error handling and monitoring."""
    args = parse_args()
    
    with LoggerContextManager("vae_training", __name__) as log_ctx:
        try:
            # Initialize wandb
            wandb.init(project=args.wandb_project)
            
            # Load configuration
            config = {
                'batch_size': args.batch_size,
                'lr': args.lr,
                'epochs': args.epochs,
                'sequence_length': args.sequence_length,
                'checkpoint_dir': args.checkpoint_dir
            }
            
            # Initialize training components
            encoder, decoder, optimizer, metrics = initialize_training(config)
            
            # Training loop
            for epoch in range(args.epochs):
                try:
                    epoch_metrics = train_epoch(
                        encoder, decoder, train_dataloader, optimizer, config
                    )
                    
                    # Log metrics
                    wandb.log({
                        'epoch': epoch,
                        **epoch_metrics
                    })
                    
                    # Save checkpoint
                    if epoch % config['checkpoint_frequency'] == 0:
                        save_checkpoint(
                            encoder, decoder, optimizer, epoch, epoch_metrics,
                            Path(args.checkpoint_dir) / f"checkpoint_{epoch}.pt"
                        )
                        
                except Exception as e:
                    if not handle_training_error(e, {
                        'batch_size': args.batch_size,
                        'last_checkpoint': get_latest_checkpoint(args.checkpoint_dir)
                    }):
                        raise
            
            # Save final model
            save_checkpoint(
                encoder, decoder, optimizer, args.epochs, epoch_metrics,
                Path(args.checkpoint_dir) / "final_model.pt"
            )
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Cleanup
            wandb.finish()

if __name__ == "__main__":
    main()