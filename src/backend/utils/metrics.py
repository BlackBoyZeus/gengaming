# External imports with versions
from prometheus_client import Histogram, Gauge, Counter  # prometheus_client ^0.16.0
import numpy as np  # numpy ^1.23.0
from typing import Dict, List, Optional  # typing ^3.9
import freebsd  # freebsd ^1.0.0
import time

# Internal imports
from core.metrics import initialize_metrics, MetricsCollector

def calculate_fps(frame_timestamps: List[float]) -> float:
    """Calculate frames per second from frame timestamps."""
    try:
        if len(frame_timestamps) < 2:
            raise ValueError("Need at least 2 timestamps to calculate FPS")
            
        # Calculate time differences between consecutive frames
        time_diffs = np.diff(frame_timestamps)
        
        # Calculate average FPS
        avg_time_per_frame = np.mean(time_diffs)
        fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0.0
        
        return float(fps)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate FPS: {str(e)}")

def calculate_latency(start_time: float, end_time: float) -> float:
    """Calculate operation latency from start and end timestamps."""
    try:
        if end_time < start_time:
            raise ValueError("End time cannot be before start time")
            
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000.0
        
        return latency_ms
    except Exception as e:
        raise RuntimeError(f"Failed to calculate latency: {str(e)}")

def get_jail_metrics(jail_name: str) -> Dict[str, float]:
    """Collect FreeBSD jail-specific performance metrics."""
    try:
        # Validate jail existence
        if not freebsd.jail.get_jid(jail_name):
            raise ValueError(f"Jail {jail_name} not found")
            
        # Collect CPU metrics
        cpu_stats = freebsd.jail.get_rctl(jail_name, "cpu")
        cpu_usage = cpu_stats.get("pcpu", 0.0)
        
        # Collect memory metrics
        mem_stats = freebsd.jail.get_rctl(jail_name, "vmem")
        memory_usage = mem_stats.get("memoryuse", 0.0)
        
        # Collect I/O metrics
        io_stats = freebsd.jail.get_rctl(jail_name, "io")
        read_bytes = io_stats.get("readbps", 0.0)
        write_bytes = io_stats.get("writebps", 0.0)
        
        return {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_bytes": memory_usage,
            "io_read_bytes": read_bytes,
            "io_write_bytes": write_bytes
        }
    except Exception as e:
        raise RuntimeError(f"Failed to collect jail metrics: {str(e)}")

class PerformanceMetrics:
    """Utility class for tracking and calculating performance metrics with FreeBSD jail support."""
    
    def __init__(self, jail_name: str):
        """Initialize performance metrics tracker with jail support."""
        self._collector = MetricsCollector()
        self._metrics_history = {
            "latency": [],
            "fps": [],
            "cpu_usage": [],
            "memory_usage": []
        }
        self._jail_name = jail_name
        self._jail_metrics = {}
        
        # Initialize metrics collectors
        initialize_metrics()
        
    def track_generation_performance(self, latency_ms: float, fps: float) -> Dict[str, float]:
        """Track video generation performance metrics including jail resources."""
        try:
            # Record generation metrics
            self._collector.record_metric("generation_latency", latency_ms)
            self._collector.record_metric("frame_rate", fps)
            
            # Update metrics history
            self._metrics_history["latency"].append(latency_ms)
            self._metrics_history["fps"].append(fps)
            
            # Collect jail metrics
            jail_metrics = get_jail_metrics(self._jail_name)
            self._jail_metrics = jail_metrics
            
            # Record jail metrics
            self._collector.record_metric("jail_cpu", jail_metrics["cpu_usage_percent"], 
                                        {"jail_name": self._jail_name})
            self._collector.record_metric("jail_memory", jail_metrics["memory_usage_bytes"], 
                                        {"jail_name": self._jail_name})
            
            # Calculate moving averages
            avg_latency = np.mean(self._metrics_history["latency"][-100:])
            avg_fps = np.mean(self._metrics_history["fps"][-100:])
            
            return {
                "current_latency_ms": latency_ms,
                "current_fps": fps,
                "average_latency_ms": float(avg_latency),
                "average_fps": float(avg_fps),
                "jail_cpu_percent": jail_metrics["cpu_usage_percent"],
                "jail_memory_bytes": jail_metrics["memory_usage_bytes"],
                "jail_io_read_bytes": jail_metrics["io_read_bytes"],
                "jail_io_write_bytes": jail_metrics["io_write_bytes"]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to track generation performance: {str(e)}")
            
    def track_jail_resources(self) -> Dict[str, float]:
        """Track FreeBSD jail resource utilization."""
        try:
            # Collect current jail metrics
            jail_metrics = get_jail_metrics(self._jail_name)
            
            # Update metrics history
            self._metrics_history["cpu_usage"].append(jail_metrics["cpu_usage_percent"])
            self._metrics_history["memory_usage"].append(jail_metrics["memory_usage_bytes"])
            
            # Calculate resource utilization
            avg_cpu = np.mean(self._metrics_history["cpu_usage"][-60:])  # 1 minute average
            avg_memory = np.mean(self._metrics_history["memory_usage"][-60:])
            
            # Record metrics
            self._collector.record_metric("jail_cpu", avg_cpu, {"jail_name": self._jail_name})
            self._collector.record_metric("jail_memory", avg_memory, {"jail_name": self._jail_name})
            
            return {
                "average_cpu_percent": float(avg_cpu),
                "average_memory_bytes": float(avg_memory),
                "current_cpu_percent": jail_metrics["cpu_usage_percent"],
                "current_memory_bytes": jail_metrics["memory_usage_bytes"],
                "io_read_bytes": jail_metrics["io_read_bytes"],
                "io_write_bytes": jail_metrics["io_write_bytes"]
            }
        except Exception as e:
            raise RuntimeError(f"Failed to track jail resources: {str(e)}")