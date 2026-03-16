"""
Prometheus Metrics Module for LLM Servant
Provides CPU, RAM, and other system metrics in Prometheus format.
"""

import threading
import time
from typing import Optional, Callable

import psutil

try:
    from prometheus_client import (
        Gauge,
        Counter,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Provide dummy classes for when prometheus_client is not installed
    Gauge = None
    Counter = None
    REGISTRY = None


# ============================================================
# Metrics Registry and Definitions
# ============================================================

# Custom registry for LLM Servant metrics
# Using default REGISTRY for compatibility with existing exporters

if PROMETHEUS_AVAILABLE:
    # System Resource Metrics
    ram_usage_bytes = Gauge(
        'llm_servant_ram_usage_bytes',
        'Current RAM usage in bytes',
        ['type']  # type: total, used, available, cached, buffers
    )
    
    ram_usage_percent = Gauge(
        'llm_servant_ram_usage_percent',
        'Current RAM usage as percentage'
    )
    
    cpu_usage_percent = Gauge(
        'llm_servant_cpu_usage_percent',
        'Current CPU usage as percentage',
        ['cpu']  # cpu: total, or core number (0, 1, 2, ...)
    )
    
    cpu_count = Gauge(
        'llm_servant_cpu_count',
        'Number of CPU cores',
        ['type']  # type: physical, logical
    )
    
    # Process-specific metrics
    process_ram_bytes = Gauge(
        'llm_servant_process_ram_bytes',
        'RAM usage of the LLM Servant process in bytes'
    )
    
    process_cpu_percent = Gauge(
        'llm_servant_process_cpu_percent',
        'CPU usage of the LLM Servant process as percentage'
    )
    
    process_threads = Gauge(
        'llm_servant_process_threads',
        'Number of threads in the LLM Servant process'
    )
    
    # Application metrics
    request_count = Counter(
        'llm_servant_requests_total',
        'Total number of requests',
        ['endpoint', 'method', 'status']
    )
    
    active_sessions = Gauge(
        'llm_servant_active_sessions',
        'Number of active chat sessions'
    )
    
    knowledge_memory_size_bytes = Gauge(
        'llm_servant_knowledge_memory_size_bytes',
        'Size of the knowledge memory in bytes'
    )
    
    pdf_documents_processed = Gauge(
        'llm_servant_pdf_documents_processed',
        'Total number of PDF documents processed'
    )
    
    insights_count = Gauge(
        'llm_servant_insights_count',
        'Total number of insights in knowledge memory'
    )


# ============================================================
# Metrics Collector
# ============================================================

class MetricsCollector:
    """
    Background collector that periodically updates Prometheus metrics.
    
    Collects system metrics (CPU, RAM) at configurable intervals.
    """
    
    def __init__(self, collection_interval: float = 15.0):
        """
        Initialize the metrics collector.
        
        Args:
            collection_interval: How often to collect metrics in seconds (default: 15)
        """
        self.collection_interval = collection_interval
        self._stop_event = threading.Event()
        self._collector_thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
    
    def start(self) -> None:
        """Start the background metrics collection thread."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        if self._collector_thread is not None and self._collector_thread.is_alive():
            return  # Already running
        
        self._stop_event.clear()
        self._collector_thread = threading.Thread(
            target=self._collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        self._collector_thread.start()
    
    def stop(self) -> None:
        """Stop the background metrics collection thread."""
        self._stop_event.set()
        if self._collector_thread is not None:
            self._collector_thread.join(timeout=5.0)
            self._collector_thread = None
    
    def _collection_loop(self) -> None:
        """Main collection loop that runs in background thread."""
        while not self._stop_event.is_set():
            try:
                self.collect_all_metrics()
            except Exception:
                pass  # Silently ignore collection errors
            
            # Wait for next collection interval
            self._stop_event.wait(timeout=self.collection_interval)
    
    def collect_all_metrics(self) -> None:
        """Collect all system and process metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self._collect_ram_metrics()
        self._collect_cpu_metrics()
        self._collect_process_metrics()
    
    def _collect_ram_metrics(self) -> None:
        """Collect RAM usage metrics."""
        try:
            mem = psutil.virtual_memory()
            
            ram_usage_bytes.labels(type='total').set(mem.total)
            ram_usage_bytes.labels(type='used').set(mem.used)
            ram_usage_bytes.labels(type='available').set(mem.available)
            
            # These may not exist on all platforms
            if hasattr(mem, 'cached'):
                ram_usage_bytes.labels(type='cached').set(mem.cached)
            if hasattr(mem, 'buffers'):
                ram_usage_bytes.labels(type='buffers').set(mem.buffers)
            
            ram_usage_percent.set(mem.percent)
        except Exception:
            pass
    
    def _collect_cpu_metrics(self) -> None:
        """Collect CPU usage metrics."""
        try:
            # Total CPU usage
            total_cpu = psutil.cpu_percent(interval=None)
            cpu_usage_percent.labels(cpu='total').set(total_cpu)
            
            # Per-core CPU usage
            per_cpu = psutil.cpu_percent(interval=None, percpu=True)
            for i, cpu_pct in enumerate(per_cpu):
                cpu_usage_percent.labels(cpu=str(i)).set(cpu_pct)
            
            # CPU counts
            cpu_count.labels(type='physical').set(psutil.cpu_count(logical=False) or 0)
            cpu_count.labels(type='logical').set(psutil.cpu_count(logical=True) or 0)
        except Exception:
            pass
    
    def _collect_process_metrics(self) -> None:
        """Collect process-specific metrics."""
        try:
            # Refresh process info
            self._process = psutil.Process()
            
            # Memory usage
            mem_info = self._process.memory_info()
            process_ram_bytes.set(mem_info.rss)
            
            # CPU usage
            cpu_pct = self._process.cpu_percent()
            process_cpu_percent.set(cpu_pct)
            
            # Thread count
            num_threads = self._process.num_threads()
            process_threads.set(num_threads)
        except Exception:
            pass


# ============================================================
# Metrics Helper Functions
# ============================================================

def get_metrics_response() -> tuple:
    """
    Generate Prometheus metrics response.
    
    Returns:
        Tuple of (metrics_data, content_type) suitable for HTTP response
        
    Example:
        >>> data, content_type = get_metrics_response()
        >>> return Response(data, mimetype=content_type)
    """
    if not PROMETHEUS_AVAILABLE:
        return (
            "# prometheus_client not installed\n",
            "text/plain; charset=utf-8"
        )
    
    return (
        generate_latest(REGISTRY),
        CONTENT_TYPE_LATEST
    )


def update_knowledge_metrics(
    size_bytes: int,
    pdf_count: int,
    insights: int
) -> None:
    """
    Update knowledge memory related metrics.
    
    Args:
        size_bytes: Current size of knowledge memory in bytes
        pdf_count: Number of PDFs processed
        insights: Number of insights extracted
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    knowledge_memory_size_bytes.set(size_bytes)
    pdf_documents_processed.set(pdf_count)
    insights_count.set(insights)


def update_session_count(count: int) -> None:
    """
    Update the active sessions metric.
    
    Args:
        count: Current number of active sessions
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    active_sessions.set(count)


def record_request(endpoint: str, method: str, status: int) -> None:
    """
    Record a request for metrics.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        status: HTTP status code
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    request_count.labels(
        endpoint=endpoint,
        method=method,
        status=str(status)
    ).inc()


def is_prometheus_available() -> bool:
    """
    Check if prometheus_client is available.
    
    Returns:
        True if prometheus_client is installed and available
    """
    return PROMETHEUS_AVAILABLE


# ============================================================
# Global Collector Instance
# ============================================================

# Create a default collector instance
_default_collector: Optional[MetricsCollector] = None


def start_metrics_collection(interval: float = 15.0) -> MetricsCollector:
    """
    Start the global metrics collector.
    
    Args:
        interval: Collection interval in seconds
        
    Returns:
        The MetricsCollector instance
    """
    global _default_collector
    
    if _default_collector is None:
        _default_collector = MetricsCollector(collection_interval=interval)
    
    _default_collector.start()
    return _default_collector


def stop_metrics_collection() -> None:
    """Stop the global metrics collector."""
    global _default_collector
    
    if _default_collector is not None:
        _default_collector.stop()
        _default_collector = None


def get_system_stats_dict() -> dict:
    """
    Get current system stats as a dictionary.
    
    This is a non-Prometheus alternative for getting system stats.
    
    Returns:
        Dictionary with CPU and RAM statistics
    """
    result = {
        "cpu": {
            "usage_percent": 0.0,
            "per_cpu_percent": [],
            "physical_cores": psutil.cpu_count(logical=False) or 0,
            "logical_cores": psutil.cpu_count(logical=True) or 0,
        },
        "ram": {
            "total_bytes": 0,
            "used_bytes": 0,
            "available_bytes": 0,
            "percent": 0.0,
        },
        "process": {
            "ram_bytes": 0,
            "cpu_percent": 0.0,
            "threads": 0,
        }
    }
    
    try:
        # CPU stats
        result["cpu"]["usage_percent"] = psutil.cpu_percent(interval=None)
        result["cpu"]["per_cpu_percent"] = psutil.cpu_percent(interval=None, percpu=True)
        
        # RAM stats
        mem = psutil.virtual_memory()
        result["ram"]["total_bytes"] = mem.total
        result["ram"]["used_bytes"] = mem.used
        result["ram"]["available_bytes"] = mem.available
        result["ram"]["percent"] = mem.percent
        
        # Process stats
        process = psutil.Process()
        mem_info = process.memory_info()
        result["process"]["ram_bytes"] = mem_info.rss
        result["process"]["cpu_percent"] = process.cpu_percent()
        result["process"]["threads"] = process.num_threads()
    except Exception:
        pass
    
    return result
