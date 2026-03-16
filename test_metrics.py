"""
Tests for Prometheus Metrics Module
"""

import time
import threading
import unittest
from unittest.mock import patch, MagicMock

from metrics import (
    MetricsCollector,
    get_system_stats_dict,
    is_prometheus_available,
    start_metrics_collection,
    stop_metrics_collection,
    update_knowledge_metrics,
    update_session_count,
    record_request,
    get_metrics_response,
)


class TestMetricsModule(unittest.TestCase):
    """Test cases for metrics module."""
    
    def test_is_prometheus_available(self):
        """Test that prometheus availability check works."""
        result = is_prometheus_available()
        # Should be True since we installed prometheus_client
        self.assertIsInstance(result, bool)
    
    def test_get_system_stats_dict(self):
        """Test getting system stats as a dictionary."""
        stats = get_system_stats_dict()
        
        # Verify CPU stats
        self.assertIn("cpu", stats)
        self.assertIn("usage_percent", stats["cpu"])
        self.assertIn("per_cpu_percent", stats["cpu"])
        self.assertIn("physical_cores", stats["cpu"])
        self.assertIn("logical_cores", stats["cpu"])
        
        # Verify RAM stats
        self.assertIn("ram", stats)
        self.assertIn("total_bytes", stats["ram"])
        self.assertIn("used_bytes", stats["ram"])
        self.assertIn("available_bytes", stats["ram"])
        self.assertIn("percent", stats["ram"])
        
        # Verify process stats
        self.assertIn("process", stats)
        self.assertIn("ram_bytes", stats["process"])
        self.assertIn("cpu_percent", stats["process"])
        self.assertIn("threads", stats["process"])
    
    def test_get_system_stats_values(self):
        """Test that system stats have reasonable values."""
        stats = get_system_stats_dict()
        
        # CPU percent should be between 0 and 100
        self.assertGreaterEqual(stats["cpu"]["usage_percent"], 0)
        self.assertLessEqual(stats["cpu"]["usage_percent"], 100)
        
        # RAM values should be positive
        self.assertGreater(stats["ram"]["total_bytes"], 0)
        self.assertGreater(stats["ram"]["used_bytes"], 0)
        
        # RAM percent should be between 0 and 100
        self.assertGreaterEqual(stats["ram"]["percent"], 0)
        self.assertLessEqual(stats["ram"]["percent"], 100)
        
        # At least 1 logical core
        self.assertGreaterEqual(stats["cpu"]["logical_cores"], 1)


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector class."""
    
    def test_collector_initialization(self):
        """Test MetricsCollector can be initialized."""
        collector = MetricsCollector(collection_interval=1.0)
        
        self.assertEqual(collector.collection_interval, 1.0)
    
    def test_collector_start_stop(self):
        """Test starting and stopping the collector."""
        collector = MetricsCollector(collection_interval=0.5)
        
        collector.start()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        if collector._collector_thread:
            self.assertTrue(collector._collector_thread.is_alive())
        
        collector.stop()
        
        # After stopping, thread should not be alive
        if collector._collector_thread:
            self.assertFalse(collector._collector_thread.is_alive())
    
    def test_collector_collects_metrics(self):
        """Test that the collector actually collects metrics."""
        collector = MetricsCollector(collection_interval=0.1)
        
        # Just test that collect_all_metrics doesn't raise
        collector.collect_all_metrics()


class TestPrometheusMetricsFunctions(unittest.TestCase):
    """Test cases for Prometheus metrics helper functions."""
    
    def test_get_metrics_response(self):
        """Test getting Prometheus metrics response."""
        data, content_type = get_metrics_response()
        
        self.assertIsInstance(data, (str, bytes))
        self.assertIn("text", content_type)
    
    @unittest.skipUnless(is_prometheus_available(), "Prometheus not available")
    def test_update_knowledge_metrics(self):
        """Test updating knowledge memory metrics."""
        # Should not raise
        update_knowledge_metrics(
            size_bytes=1024000,
            pdf_count=5,
            insights=100
        )
    
    @unittest.skipUnless(is_prometheus_available(), "Prometheus not available")
    def test_update_session_count(self):
        """Test updating session count metric."""
        # Should not raise
        update_session_count(5)
    
    @unittest.skipUnless(is_prometheus_available(), "Prometheus not available")
    def test_record_request(self):
        """Test recording a request metric."""
        # Should not raise
        record_request(
            endpoint="/api/chat",
            method="POST",
            status=200
        )


class TestGlobalMetricsCollection(unittest.TestCase):
    """Test cases for global metrics collection functions."""
    
    def test_start_stop_metrics_collection(self):
        """Test starting and stopping global metrics collection."""
        collector = start_metrics_collection(interval=1.0)
        
        self.assertIsNotNone(collector)
        
        stop_metrics_collection()


class TestMetricsWithoutPrometheus(unittest.TestCase):
    """Test that metrics work gracefully without prometheus_client."""
    
    def test_functions_work_without_prometheus(self):
        """Test that helper functions don't raise when prometheus is not available."""
        # These should not raise even if prometheus_client is unavailable
        # (they check is_prometheus_available() internally)
        
        # Note: Since we have prometheus_client installed, we can't fully test
        # the "not available" case without mocking, but we can verify the
        # functions handle the available case correctly
        
        try:
            update_knowledge_metrics(1024, 1, 10)
            update_session_count(1)
            record_request("/test", "GET", 200)
        except Exception as e:
            self.fail(f"Metrics functions raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
