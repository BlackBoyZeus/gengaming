# External imports with versions
from prometheus_client import start_http_server  # prometheus_client ^0.16.0
from elasticsearch import Elasticsearch  # elasticsearch ^8.8.0
import psutil  # psutil ^5.9.0
from typing import Dict, Optional, Any
import time
import os

# Internal imports
from core.metrics import MetricsCollector
from core.logging import get_logger

# Initialize logger and metrics collector
logger = get_logger(__name__)
metrics_collector = MetricsCollector()

def initialize_monitoring(config: Dict) -> None:
    """Initialize monitoring services with FreeBSD-specific configuration"""
    try:
        # Configure Prometheus with FreeBSD jail support
        start_http_server(
            port=config.get('prometheus_port', 9090),
            addr=config.get('prometheus_addr', '0.0.0.0')
        )
        
        # Configure ELK Stack integration
        if config.get('elk_enabled', True):
            es_client = Elasticsearch(
                hosts=config['elk_hosts'],
                basic_auth=(config['elk_user'], config['elk_password']),
                verify_certs=config.get('verify_certs', True)
            )
            logger.info("ELK Stack integration configured")
            
        # Initialize FreeBSD jail monitoring
        if os.path.exists('/usr/sbin/jls'):
            logger.info("FreeBSD jail monitoring enabled")
            
        # Configure non-NVIDIA GPU monitoring
        _setup_gpu_monitoring(config.get('gpu_config', {}))
        
        logger.info("Monitoring services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring: {str(e)}")
        raise

def check_system_health() -> Dict:
    """Check overall system health including FreeBSD jails"""
    try:
        health_metrics = {
            'cpu': _check_cpu_health(),
            'memory': _check_memory_health(),
            'gpu': _check_gpu_health(),
            'storage': _check_storage_health(),
            'jails': _check_jail_health(),
            'timestamp': time.time()
        }
        
        return health_metrics
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise

class MonitoringService:
    """Main monitoring service class with FreeBSD support"""
    
    def __init__(self, config: Dict):
        """Initialize monitoring service with FreeBSD support"""
        self._metrics_collector = MetricsCollector()
        self._health_checks = {}
        self._alerts = {}
        self._jail_metrics = {}
        
        # Initialize monitoring components
        self._init_health_checks()
        self._init_alert_thresholds(config.get('alert_thresholds', {}))
        self._init_jail_monitoring()
        
        logger.info("MonitoringService initialized")

    def record_metrics(self, metric_name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Record system metrics including jail statistics"""
        try:
            # Add jail context if applicable
            if labels and 'jail_name' in labels:
                self._update_jail_metrics(labels['jail_name'], metric_name, value)
            
            # Record metric with timestamp
            self._metrics_collector.record_metric(
                metric_name=metric_name,
                value=value,
                labels=labels
            )
            
            # Check alert thresholds
            self._check_alert_thresholds(metric_name, value, labels)
            
        except Exception as e:
            logger.error(f"Failed to record metric {metric_name}: {str(e)}")
            raise

    def monitor_performance(self) -> Dict:
        """Monitor system performance metrics with FreeBSD specifics"""
        try:
            performance_metrics = {
                'generation': {
                    'latency': self._get_generation_latency(),
                    'frame_rate': self._get_frame_rate()
                },
                'control': {
                    'response_time': self._get_control_response_time()
                },
                'resources': {
                    'cpu': self._get_cpu_metrics(),
                    'memory': self._get_memory_metrics(),
                    'gpu': self._get_gpu_metrics(),
                    'storage': self._get_storage_metrics()
                },
                'jails': self._get_jail_metrics()
            }
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {str(e)}")
            raise

    def handle_alert(self, alert_type: str, alert_data: Dict) -> None:
        """Handle monitoring alerts with enhanced FreeBSD support"""
        try:
            # Validate alert data
            if not self._validate_alert(alert_type, alert_data):
                raise ValueError(f"Invalid alert data for type: {alert_type}")
            
            # Process alert based on type
            if alert_type == 'resource_limit':
                self._handle_resource_alert(alert_data)
            elif alert_type == 'performance':
                self._handle_performance_alert(alert_data)
            elif alert_type == 'jail':
                self._handle_jail_alert(alert_data)
            
            # Log alert to ELK Stack
            logger.warning(
                f"Alert triggered: {alert_type}",
                extra={'alert_data': alert_data}
            )
            
            # Update alert status
            self._alerts[alert_type] = {
                'status': 'active',
                'data': alert_data,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to handle alert {alert_type}: {str(e)}")
            raise

    def _init_health_checks(self) -> None:
        """Initialize system health checks"""
        self._health_checks = {
            'cpu': lambda: psutil.cpu_percent(interval=1),
            'memory': lambda: psutil.virtual_memory().percent,
            'storage': lambda: psutil.disk_usage('/').percent,
            'gpu': self._check_gpu_health
        }

    def _init_alert_thresholds(self, thresholds: Dict) -> None:
        """Initialize alert thresholds"""
        self._alert_thresholds = {
            'cpu_usage': thresholds.get('cpu_usage', 90),
            'memory_usage': thresholds.get('memory_usage', 85),
            'gpu_usage': thresholds.get('gpu_usage', 80),
            'storage_usage': thresholds.get('storage_usage', 85),
            'generation_latency': thresholds.get('generation_latency', 100),
            'frame_rate': thresholds.get('frame_rate', 24)
        }

    def _init_jail_monitoring(self) -> None:
        """Initialize FreeBSD jail monitoring"""
        if os.path.exists('/usr/sbin/jls'):
            self._jail_metrics = {}
            # Get active jails
            jails = os.popen('jls name').read().splitlines()[1:]
            for jail in jails:
                self._jail_metrics[jail] = {
                    'cpu': 0,
                    'memory': 0,
                    'network': {'in': 0, 'out': 0}
                }

    def _update_jail_metrics(self, jail_name: str, metric_name: str, value: float) -> None:
        """Update jail-specific metrics"""
        if jail_name in self._jail_metrics:
            if metric_name.startswith('cpu'):
                self._jail_metrics[jail_name]['cpu'] = value
            elif metric_name.startswith('memory'):
                self._jail_metrics[jail_name]['memory'] = value
            elif metric_name.startswith('network'):
                direction = 'in' if 'in' in metric_name else 'out'
                self._jail_metrics[jail_name]['network'][direction] = value

def _setup_gpu_monitoring(gpu_config: Dict) -> None:
    """Configure non-NVIDIA GPU monitoring"""
    try:
        # Configure GPU monitoring based on hardware
        if os.path.exists('/dev/dri'):
            logger.info("Configuring DRI-based GPU monitoring")
        else:
            logger.warning("No compatible GPU monitoring interface found")
    except Exception as e:
        logger.error(f"Failed to setup GPU monitoring: {str(e)}")
        raise