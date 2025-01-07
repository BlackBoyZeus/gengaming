# External imports with versions
from prometheus_client import Histogram, Gauge, Counter, start_http_server  # prometheus_client ^0.16.0
import time  # ^3.9
from typing import Dict, Optional, Any  # ^3.9

# Internal imports
from core.config import Settings
from core.logging import logger

# Global metric collectors
GENERATION_LATENCY = Histogram(
    'generation_latency_ms',
    'Video generation latency in milliseconds',
    buckets=[10, 50, 100, 200, 500, 1000]
)

FRAME_RATE = Gauge(
    'frame_rate_fps',
    'Video frame rate in FPS'
)

CONTROL_RESPONSE = Histogram(
    'control_response_ms',
    'Control operation response time in milliseconds',
    buckets=[10, 25, 50, 100, 200]
)

CONCURRENT_USERS = Gauge(
    'concurrent_users',
    'Number of concurrent users'
)

JAIL_CPU_USAGE = Gauge(
    'jail_cpu_usage_percent',
    'CPU usage per FreeBSD jail',
    ['jail_name']
)

JAIL_MEMORY_USAGE = Gauge(
    'jail_memory_usage_bytes',
    'Memory usage per FreeBSD jail',
    ['jail_name']
)

JAIL_NETWORK_IO = Counter(
    'jail_network_io_bytes',
    'Network I/O per FreeBSD jail',
    ['jail_name', 'direction']
)

class MetricsCollector:
    """Core metrics collection and management class with FreeBSD support"""
    
    def __init__(self):
        """Initialize metrics collector with FreeBSD support"""
        self._collectors = {
            'generation_latency': GENERATION_LATENCY,
            'frame_rate': FRAME_RATE,
            'control_response': CONTROL_RESPONSE,
            'concurrent_users': CONCURRENT_USERS,
            'jail_cpu': JAIL_CPU_USAGE,
            'jail_memory': JAIL_MEMORY_USAGE,
            'jail_network': JAIL_NETWORK_IO
        }
        
        self._labels = {}
        self._thresholds = Settings().performance_thresholds
        self._jail_metrics = {}
        
        logger.info("MetricsCollector initialized with FreeBSD support")

    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value with labels and threshold validation"""
        try:
            # Validate metric exists
            if metric_name not in self._collectors:
                raise ValueError(f"Unknown metric: {metric_name}")

            # Get collector
            collector = self._collectors[metric_name]

            # Check thresholds
            if metric_name == 'generation_latency' and value > self._thresholds['max_generation_latency']:
                logger.error(f"Generation latency exceeded threshold: {value}ms")
            elif metric_name == 'frame_rate' and value < self._thresholds['min_frame_rate']:
                logger.error(f"Frame rate below threshold: {value}fps")
            elif metric_name == 'control_response' and value > self._thresholds['max_control_latency']:
                logger.error(f"Control latency exceeded threshold: {value}ms")

            # Record metric with labels if provided
            if labels and isinstance(collector, (Gauge, Counter)):
                collector.labels(**labels).set(value)
            elif isinstance(collector, Histogram):
                collector.observe(value)
            else:
                collector.set(value)

            logger.info(f"Recorded metric {metric_name}: {value}")

        except Exception as e:
            logger.error(f"Error recording metric {metric_name}: {str(e)}")
            raise

    def get_jail_metrics(self, jail_name: str) -> Dict[str, Any]:
        """Get FreeBSD jail-specific metrics"""
        try:
            # Collect jail metrics using FreeBSD utilities
            metrics = {
                'cpu_usage': JAIL_CPU_USAGE.labels(jail_name=jail_name)._value.get(),
                'memory_usage': JAIL_MEMORY_USAGE.labels(jail_name=jail_name)._value.get(),
                'network_in': JAIL_NETWORK_IO.labels(jail_name=jail_name, direction='in')._value.get(),
                'network_out': JAIL_NETWORK_IO.labels(jail_name=jail_name, direction='out')._value.get()
            }

            # Validate against resource limits
            resource_limits = Settings().resource_limits
            if metrics['memory_usage'] > resource_limits['max_memory']:
                logger.error(f"Jail {jail_name} exceeded memory limit")

            return metrics

        except Exception as e:
            logger.error(f"Error collecting jail metrics for {jail_name}: {str(e)}")
            raise

def initialize_metrics() -> None:
    """Initialize Prometheus metrics collectors and exporters with FreeBSD jail support"""
    try:
        # Start Prometheus HTTP server on port 9090
        start_http_server(9090)
        
        # Initialize collectors
        MetricsCollector()
        
        logger.info("Metrics system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize metrics system: {str(e)}")
        raise

def track_generation_latency(latency_ms: float) -> None:
    """Track video generation latency with threshold validation"""
    try:
        # Validate latency value
        if latency_ms < 0:
            raise ValueError("Latency cannot be negative")

        # Record latency metric
        GENERATION_LATENCY.observe(latency_ms)

        # Check threshold
        if latency_ms > Settings().performance_thresholds['max_generation_latency']:
            logger.error(f"Generation latency threshold exceeded: {latency_ms}ms")
        else:
            logger.info(f"Generation latency recorded: {latency_ms}ms")

    except Exception as e:
        logger.error(f"Error tracking generation latency: {str(e)}")
        raise

def track_jail_metrics(jail_name: str) -> None:
    """Track FreeBSD jail-specific metrics"""
    try:
        # Get jail metrics using FreeBSD utilities
        metrics_collector = MetricsCollector()
        jail_metrics = metrics_collector.get_jail_metrics(jail_name)

        # Update Prometheus metrics
        JAIL_CPU_USAGE.labels(jail_name=jail_name).set(jail_metrics['cpu_usage'])
        JAIL_MEMORY_USAGE.labels(jail_name=jail_name).set(jail_metrics['memory_usage'])
        JAIL_NETWORK_IO.labels(jail_name=jail_name, direction='in').inc(jail_metrics['network_in'])
        JAIL_NETWORK_IO.labels(jail_name=jail_name, direction='out').inc(jail_metrics['network_out'])

        logger.info(f"Updated metrics for jail: {jail_name}")

    except Exception as e:
        logger.error(f"Error tracking jail metrics: {str(e)}")
        raise