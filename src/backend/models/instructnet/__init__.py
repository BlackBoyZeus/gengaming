# External imports with versions
import torch  # torch ^2.0.0
import time  # built-in
from performance_monitoring import Monitor  # performance_monitoring ^1.2.0

# Internal imports
from models.instructnet.config import InstructNetConfig
from models.instructnet.control import ControlProcessor
from models.instructnet.modification import ModificationModule

# Default configuration instance
DEFAULT_CONFIG = InstructNetConfig()

@torch.jit.script
class InstructNet:
    """
    Main InstructNet model class that integrates control processing and latent modification
    with performance monitoring and hardware optimization for FreeBSD environments.
    """
    
    def __init__(self, config: InstructNetConfig, monitor: Monitor):
        """
        Initializes InstructNet with configuration and performance monitoring.
        
        Args:
            config: Configuration instance with hardware and performance settings
            monitor: Performance monitoring instance for tracking metrics
        """
        # Validate hardware compatibility
        if not config.validate_control_settings():
            raise RuntimeError("Hardware compatibility validation failed")
            
        # Store configuration and monitor
        self.config = config
        self.monitor = monitor
        
        # Initialize core components with monitoring
        self.control_processor = ControlProcessor(config)
        self.latent_modifier = ModificationModule(config)
        
        # Initialize performance tracking
        self.last_control_latency = 0.0
        self.last_modification_accuracy = 0.0
        
        # Setup JIT optimization for FreeBSD
        torch.jit.optimize_for_inference(self)

    @torch.jit.script
    def process_control_signal(
        self,
        control_signal: dict,
        latent_states: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Processes incoming control signals with performance monitoring.
        
        Args:
            control_signal: Dictionary containing control signal data
            latent_states: Current video latent states tensor
            
        Returns:
            Tuple of (modified latent states, performance metrics)
        """
        try:
            # Start latency tracking
            start_time = time.perf_counter()
            
            # Validate control signal format
            if not all(k in control_signal for k in ["type", "data", "timestamp"]):
                raise ValueError("Invalid control signal format")
                
            # Process control signal through control processor
            control_embedding = self.control_processor.process_control(control_signal)
            
            # Apply modifications to latent states
            modified_states, mod_metrics = self.latent_modifier.modify_latents(
                latent_states,
                {"control": control_embedding},
                torch.zeros(self.config.hidden_dim)  # No instruction for control-only
            )
            
            # Track performance metrics
            end_time = time.perf_counter()
            self.last_control_latency = (end_time - start_time) * 1000  # Convert to ms
            self.last_modification_accuracy = mod_metrics["control_accuracy"]
            
            # Prepare performance metrics
            metrics = {
                "control_latency_ms": self.last_control_latency,
                "modification_accuracy": self.last_modification_accuracy,
                "success": self.last_control_latency < 50.0,  # <50ms requirement
                **mod_metrics
            }
            
            # Log metrics through monitor
            self.monitor.log_metrics("control_processing", metrics)
            
            return modified_states, metrics
            
        except Exception as e:
            self.monitor.log_error("control_processing", str(e))
            raise

    @torch.jit.script
    def modify_with_instruction(
        self,
        latent_states: torch.Tensor,
        instruction_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Modifies video content based on text instructions with performance tracking.
        
        Args:
            latent_states: Current video latent states tensor
            instruction_embedding: Embedded text instruction tensor
            
        Returns:
            Tuple of (modified latent states, performance metrics)
        """
        try:
            # Start latency tracking
            start_time = time.perf_counter()
            
            # Apply modifications to latent states
            modified_states, mod_metrics = self.latent_modifier.modify_latents(
                latent_states,
                {"control": torch.zeros(self.config.hidden_dim)},  # No control for instruction-only
                instruction_embedding
            )
            
            # Track performance metrics
            end_time = time.perf_counter()
            self.last_control_latency = (end_time - start_time) * 1000  # Convert to ms
            self.last_modification_accuracy = mod_metrics["control_accuracy"]
            
            # Prepare performance metrics
            metrics = {
                "instruction_latency_ms": self.last_control_latency,
                "modification_accuracy": self.last_modification_accuracy,
                "success": self.last_control_latency < 50.0,  # <50ms requirement
                **mod_metrics
            }
            
            # Log metrics through monitor
            self.monitor.log_metrics("instruction_processing", metrics)
            
            return modified_states, metrics
            
        except Exception as e:
            self.monitor.log_error("instruction_processing", str(e))
            raise

    def get_performance_metrics(self) -> dict:
        """
        Returns current performance metrics.
        
        Returns:
            Dictionary containing latency and accuracy metrics
        """
        return {
            "control_latency_ms": self.last_control_latency,
            "modification_accuracy": self.last_modification_accuracy,
            "control_processor_metrics": self.control_processor.get_processing_metrics(),
            "modification_metrics": self.latent_modifier.get_modification_accuracy()
        }

# Export public interface
__all__ = ["InstructNet", "DEFAULT_CONFIG"]