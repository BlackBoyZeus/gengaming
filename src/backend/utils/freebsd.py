# External imports with versions
import psutil  # psutil==5.9.0
import resource  # Python 3.9 stdlib
import logging  # Python 3.9 stdlib
import os
import json
import subprocess
from typing import Dict, Any, Optional
from functools import wraps

# Internal imports
from core.config import settings

# Global constants
DEFAULT_RLIMIT_AS = 34359738368  # 32GB default memory limit
DEFAULT_RLIMIT_NOFILE = 65536    # Default max file descriptors

# System optimization flags
FREEBSD_OPTIMIZATION_FLAGS = {
    "kern.ipc.shm_use_phys": "1",
    "kern.ipc.shmmax": "34359738368",
    "kern.sched.slice": "3",
    "kern.maxproc": "32768",
    "kern.ipc.somaxconn": "4096"
}

ORBIS_OPTIMIZATION_FLAGS = {
    "orbis.sys.process.priority": "high",
    "orbis.sys.memory.mode": "performance",
    "orbis.sys.cpu.mode": "maximum"
}

# Configure logging
logger = logging.getLogger(__name__)

def system_compatible(cls):
    """Decorator to verify FreeBSD and Orbis OS compatibility"""
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if not os.path.exists("/usr/sbin/sysctl"):
            raise RuntimeError("FreeBSD sysctl not found")
        if not os.path.exists("/usr/lib/libc.so.7"):
            raise RuntimeError("FreeBSD compatibility layer not found")
        return cls(*args, **kwargs)
    return wrapper

def performance_monitor(func):
    """Decorator for monitoring system performance metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_metrics = _collect_performance_metrics()
        result = func(*args, **kwargs)
        end_metrics = _collect_performance_metrics()
        _log_performance_delta(start_metrics, end_metrics)
        return result
    return wrapper

def logging_catch_exceptions(func):
    """Decorator for comprehensive error handling and logging"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"FreeBSD operation failed: {str(e)}", exc_info=True)
            raise
    return wrapper

def _collect_performance_metrics() -> Dict[str, float]:
    """Collect comprehensive system performance metrics"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "swap_percent": psutil.swap_memory().percent,
        "io_counters": psutil.disk_io_counters()._asdict(),
        "network_counters": psutil.net_io_counters()._asdict()
    }

def _log_performance_delta(start: Dict[str, float], end: Dict[str, float]):
    """Log performance metric changes"""
    for metric in start:
        if isinstance(start[metric], (int, float)):
            delta = end[metric] - start[metric]
            logger.info(f"Performance delta for {metric}: {delta}")

@logging_catch_exceptions
def set_resource_limits(
    memory_limit_bytes: int = DEFAULT_RLIMIT_AS,
    max_file_descriptors: int = DEFAULT_RLIMIT_NOFILE,
    additional_limits: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Sets comprehensive FreeBSD system resource limits with Orbis OS optimizations"""
    
    limits_status = {}
    
    # Set memory limit
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        limits_status["memory_limit"] = {"status": "success", "value": memory_limit_bytes}
    except ValueError as e:
        logger.error(f"Failed to set memory limit: {e}")
        limits_status["memory_limit"] = {"status": "failed", "error": str(e)}
    
    # Set file descriptor limit
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (max_file_descriptors, max_file_descriptors))
        limits_status["file_descriptors"] = {"status": "success", "value": max_file_descriptors}
    except ValueError as e:
        logger.error(f"Failed to set file descriptor limit: {e}")
        limits_status["file_descriptors"] = {"status": "failed", "error": str(e)}
    
    # Apply Orbis OS specific limits
    orbis_limits = {
        "orbis.process.max_threads": 1024,
        "orbis.memory.max_locked": int(memory_limit_bytes * 0.75)
    }
    
    for limit_name, limit_value in orbis_limits.items():
        try:
            subprocess.run(["sysctl", f"{limit_name}={limit_value}"], check=True)
            limits_status[limit_name] = {"status": "success", "value": limit_value}
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set Orbis OS limit {limit_name}: {e}")
            limits_status[limit_name] = {"status": "failed", "error": str(e)}
    
    # Apply additional custom limits if specified
    if additional_limits:
        for limit_name, limit_value in additional_limits.items():
            try:
                subprocess.run(["sysctl", f"{limit_name}={limit_value}"], check=True)
                limits_status[limit_name] = {"status": "success", "value": limit_value}
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to set additional limit {limit_name}: {e}")
                limits_status[limit_name] = {"status": "failed", "error": str(e)}
    
    return limits_status

@performance_monitor
def optimize_system(optimization_flags: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Applies comprehensive FreeBSD and Orbis OS specific system optimizations"""
    
    optimization_status = {}
    flags = {**FREEBSD_OPTIMIZATION_FLAGS, **(optimization_flags or {})}
    
    # Apply FreeBSD optimizations
    for flag, value in flags.items():
        try:
            subprocess.run(["sysctl", f"{flag}={value}"], check=True)
            optimization_status[flag] = {"status": "success", "value": value}
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set FreeBSD optimization {flag}: {e}")
            optimization_status[flag] = {"status": "failed", "error": str(e)}
    
    # Apply Orbis OS optimizations
    for flag, value in ORBIS_OPTIMIZATION_FLAGS.items():
        try:
            subprocess.run(["sysctl", f"{flag}={value}"], check=True)
            optimization_status[flag] = {"status": "success", "value": value}
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set Orbis OS optimization {flag}: {e}")
            optimization_status[flag] = {"status": "failed", "error": str(e)}
    
    return optimization_status

@system_compatible
class FreeBSDManager:
    """Comprehensive manager for FreeBSD system operations with Orbis OS support"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FreeBSD system manager with Orbis OS support"""
        self._system_info = {}
        self._optimization_state = {}
        self._resource_limits = {}
        self._performance_metrics = {}
        self._orbis_state = {}
        
        # Load configuration
        self.config = config or {}
        self.freebsd_settings = settings.freebsd_settings
        
        # Initialize monitoring
        self._init_monitoring()
    
    def _init_monitoring(self):
        """Initialize system monitoring"""
        self._system_info = {
            "os_version": self._get_os_version(),
            "cpu_info": self._get_cpu_info(),
            "memory_info": self._get_memory_info(),
            "storage_info": self._get_storage_info()
        }
    
    def _get_os_version(self) -> str:
        """Get FreeBSD OS version"""
        try:
            return subprocess.check_output(["uname", "-r"]).decode().strip()
        except subprocess.CalledProcessError:
            return "unknown"
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information"""
        return {
            "count": psutil.cpu_count(),
            "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "stats": psutil.cpu_stats()._asdict()
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory information"""
        return {
            "virtual": psutil.virtual_memory()._asdict(),
            "swap": psutil.swap_memory()._asdict()
        }
    
    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage information"""
        return {
            "partitions": [p._asdict() for p in psutil.disk_partitions()],
            "usage": {m.mountpoint: psutil.disk_usage(m.mountpoint)._asdict() 
                     for m in psutil.disk_partitions()}
        }
    
    @logging_catch_exceptions
    def initialize(self) -> Dict[str, Any]:
        """Initialize FreeBSD system environment with Orbis OS optimizations"""
        
        # Set resource limits
        self._resource_limits = set_resource_limits(
            memory_limit_bytes=self.config.get("memory_limit", DEFAULT_RLIMIT_AS),
            max_file_descriptors=self.config.get("max_file_descriptors", DEFAULT_RLIMIT_NOFILE)
        )
        
        # Apply system optimizations
        self._optimization_state = optimize_system(
            optimization_flags=self.config.get("optimization_flags")
        )
        
        # Collect initial metrics
        self._performance_metrics = _collect_performance_metrics()
        
        return {
            "system_info": self._system_info,
            "resource_limits": self._resource_limits,
            "optimization_state": self._optimization_state,
            "performance_metrics": self._performance_metrics
        }

# Export FreeBSDManager and utility functions
__all__ = ["FreeBSDManager", "set_resource_limits", "optimize_system"]