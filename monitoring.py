"""
Monitoring and Logging Module for Advanced Medical Imaging Diagnosis Agent
Provides system health monitoring, performance metrics, and observability
"""

import logging
import time
import psutil
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import threading
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
import redis
import sqlite3

from config import config

logger = logging.getLogger(__name__)

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.monitoring_enabled = config.ENABLE_MONITORING
        self.metrics_port = config.METRICS_PORT
        self.monitoring_interval = 30  # seconds
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # Metrics storage
        self.cpu_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.disk_history = deque(maxlen=1000)
        self.network_history = deque(maxlen=1000)
        
        # Performance metrics
        self.api_response_times = deque(maxlen=1000)
        self.image_processing_times = deque(maxlen=1000)
        self.ai_analysis_times = deque(maxlen=1000)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=100)
        
        # Initialize Prometheus metrics
        if self.monitoring_enabled:
            self._setup_prometheus_metrics()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        try:
            # System metrics
            self.cpu_gauge = Gauge('system_cpu_percent', 'CPU usage percentage')
            self.memory_gauge = Gauge('system_memory_percent', 'Memory usage percentage')
            self.disk_gauge = Gauge('system_disk_percent', 'Disk usage percentage')
            
            # Application metrics
            self.api_requests_total = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
            self.api_response_time = Histogram('api_response_time_seconds', 'API response time')
            self.image_processing_time = Histogram('image_processing_time_seconds', 'Image processing time')
            self.ai_analysis_time = Histogram('ai_analysis_time_seconds', 'AI analysis time')
            
            # Error metrics
            self.errors_total = Counter('errors_total', 'Total errors', ['error_type'])
            
            # Business metrics
            self.cases_processed = Counter('cases_processed_total', 'Total cases processed')
            self.images_analyzed = Counter('images_analyzed_total', 'Total images analyzed')
            self.reports_generated = Counter('reports_generated_total', 'Total reports generated')
            
            logger.info("Prometheus metrics setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up Prometheus metrics: {str(e)}")
    
    def start_monitoring(self):
        """Start system monitoring"""
        if not self.monitoring_enabled:
            logger.info("Monitoring is disabled")
            return
        
        try:
            # Start Prometheus HTTP server
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("System monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    def stop_monitoring_service(self):
        """Stop system monitoring"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_history.append((datetime.now(), cpu_percent))
            if self.monitoring_enabled:
                self.cpu_gauge.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_history.append((datetime.now(), memory_percent))
            if self.monitoring_enabled:
                self.memory_gauge.set(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_history.append((datetime.now(), disk_percent))
            if self.monitoring_enabled:
                self.disk_gauge.set(disk_percent)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            self.network_history.append((datetime.now(), network_data))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    def record_api_request(self, endpoint: str, method: str, response_time: float):
        """Record API request metrics"""
        try:
            self.api_response_times.append((datetime.now(), response_time))
            
            if self.monitoring_enabled:
                self.api_requests_total.labels(endpoint=endpoint, method=method).inc()
                self.api_response_time.observe(response_time)
            
        except Exception as e:
            logger.error(f"Error recording API request: {str(e)}")
    
    def record_image_processing(self, processing_time: float):
        """Record image processing metrics"""
        try:
            self.image_processing_times.append((datetime.now(), processing_time))
            
            if self.monitoring_enabled:
                self.image_processing_time.observe(processing_time)
            
        except Exception as e:
            logger.error(f"Error recording image processing: {str(e)}")
    
    def record_ai_analysis(self, analysis_time: float):
        """Record AI analysis metrics"""
        try:
            self.ai_analysis_times.append((datetime.now(), analysis_time))
            
            if self.monitoring_enabled:
                self.ai_analysis_time.observe(analysis_time)
            
        except Exception as e:
            logger.error(f"Error recording AI analysis: {str(e)}")
    
    def record_error(self, error_type: str, error_message: str):
        """Record error metrics"""
        try:
            self.error_counts[error_type] += 1
            self.error_history.append((datetime.now(), error_type, error_message))
            
            if self.monitoring_enabled:
                self.errors_total.labels(error_type=error_type).inc()
            
        except Exception as e:
            logger.error(f"Error recording error: {str(e)}")
    
    def record_business_metric(self, metric_name: str, value: int = 1):
        """Record business metrics"""
        try:
            if self.monitoring_enabled:
                if metric_name == "cases_processed":
                    self.cases_processed.inc(value)
                elif metric_name == "images_analyzed":
                    self.images_analyzed.inc(value)
                elif metric_name == "reports_generated":
                    self.reports_generated.inc(value)
            
        except Exception as e:
            logger.error(f"Error recording business metric: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            # Current metrics
            current_cpu = psutil.cpu_percent(interval=1)
            current_memory = psutil.virtual_memory().percent
            current_disk = (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100
            
            # Calculate averages
            cpu_avg = np.mean([m[1] for m in self.cpu_history]) if self.cpu_history else 0
            memory_avg = np.mean([m[1] for m in self.memory_history]) if self.memory_history else 0
            disk_avg = np.mean([m[1] for m in self.disk_history]) if self.disk_history else 0
            
            # Performance metrics
            avg_api_response = np.mean([t[1] for t in self.api_response_times]) if self.api_response_times else 0
            avg_image_processing = np.mean([t[1] for t in self.image_processing_times]) if self.image_processing_times else 0
            avg_ai_analysis = np.mean([t[1] for t in self.ai_analysis_times]) if self.ai_analysis_times else 0
            
            # Error summary
            total_errors = sum(self.error_counts.values())
            recent_errors = len([e for e in self.error_history if e[0] > datetime.now() - timedelta(hours=1)])
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'system_health': {
                    'cpu_current': current_cpu,
                    'cpu_average': cpu_avg,
                    'memory_current': current_memory,
                    'memory_average': memory_avg,
                    'disk_current': current_disk,
                    'disk_average': disk_avg
                },
                'performance_metrics': {
                    'api_response_time_avg': avg_api_response,
                    'image_processing_time_avg': avg_image_processing,
                    'ai_analysis_time_avg': avg_ai_analysis
                },
                'error_summary': {
                    'total_errors': total_errors,
                    'recent_errors_1h': recent_errors,
                    'error_breakdown': dict(self.error_counts)
                },
                'monitoring_status': {
                    'enabled': self.monitoring_enabled,
                    'metrics_port': self.metrics_port,
                    'data_points': len(self.cpu_history)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}
    
    def generate_health_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        try:
            # Get current status
            current_status = self.get_system_status()
            
            # Historical analysis
            if self.cpu_history:
                cpu_trend = self._calculate_trend([m[1] for m in self.cpu_history])
                memory_trend = self._calculate_trend([m[1] for m in self.memory_history])
                disk_trend = self._calculate_trend([m[1] for m in self.disk_history])
            else:
                cpu_trend = memory_trend = disk_trend = "insufficient_data"
            
            # Performance analysis
            if self.api_response_times:
                api_performance = self._analyze_performance([t[1] for t in self.api_response_times])
                image_performance = self._analyze_performance([t[1] for t in self.image_processing_times])
                ai_performance = self._analyze_performance([t[1] for t in self.ai_analysis_times])
            else:
                api_performance = image_performance = ai_performance = {}
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(current_status)
            
            report = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'monitoring_period': f"{self.monitoring_interval}s intervals",
                    'data_points': len(self.cpu_history)
                },
                'current_status': current_status,
                'trend_analysis': {
                    'cpu_trend': cpu_trend,
                    'memory_trend': memory_trend,
                    'disk_trend': disk_trend
                },
                'performance_analysis': {
                    'api_performance': api_performance,
                    'image_processing_performance': image_performance,
                    'ai_analysis_performance': ai_performance
                },
                'recommendations': recommendations
            }
            
            # Save report if path provided
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Health report saved to {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating health report: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return "insufficient_data"
        
        recent = values[-10:] if len(values) >= 10 else values
        older = values[:len(values)-len(recent)] if len(values) > len(recent) else values
        
        if len(older) == 0:
            return "stable"
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_performance(self, times: List[float]) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not times:
            return {}
        
        return {
            'count': len(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }
    
    def _generate_health_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Generate health recommendations"""
        recommendations = []
        
        # CPU recommendations
        if status['system_health']['cpu_current'] > 80:
            recommendations.append("High CPU usage detected. Consider scaling up resources or optimizing code.")
        
        # Memory recommendations
        if status['system_health']['memory_current'] > 85:
            recommendations.append("High memory usage detected. Check for memory leaks or consider increasing RAM.")
        
        # Disk recommendations
        if status['system_health']['disk_current'] > 90:
            recommendations.append("High disk usage detected. Consider cleanup or expanding storage.")
        
        # Performance recommendations
        if status['performance_metrics']['api_response_time_avg'] > 2.0:
            recommendations.append("Slow API response times. Consider optimizing database queries or caching.")
        
        if status['performance_metrics']['image_processing_time_avg'] > 5.0:
            recommendations.append("Slow image processing. Consider using GPU acceleration or optimizing algorithms.")
        
        # Error recommendations
        if status['error_summary']['recent_errors_1h'] > 10:
            recommendations.append("High error rate detected. Review error logs and fix underlying issues.")
        
        if not recommendations:
            recommendations.append("System is performing well. Continue monitoring for any changes.")
        
        return recommendations
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot monitoring metrics"""
        try:
            if not self.cpu_history:
                logger.warning("No metrics data available for plotting")
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Extract timestamps and values
            timestamps = [m[0] for m in self.cpu_history]
            cpu_values = [m[1] for m in self.cpu_history]
            memory_values = [m[1] for m in self.memory_history]
            disk_values = [m[1] for m in self.disk_history]
            
            # CPU usage over time
            ax1.plot(timestamps, cpu_values, 'b-', label='CPU %')
            ax1.set_title('CPU Usage Over Time')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.legend()
            ax1.grid(True)
            
            # Memory usage over time
            ax2.plot(timestamps, memory_values, 'r-', label='Memory %')
            ax2.set_title('Memory Usage Over Time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Memory Usage (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Disk usage over time
            ax3.plot(timestamps, disk_values, 'g-', label='Disk %')
            ax3.set_title('Disk Usage Over Time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Disk Usage (%)')
            ax3.legend()
            ax3.grid(True)
            
            # Performance metrics
            if self.api_response_times:
                api_times = [t[1] for t in self.api_response_times]
                ax4.hist(api_times, bins=20, alpha=0.7, label='API Response Time')
                ax4.set_title('API Response Time Distribution')
                ax4.set_xlabel('Response Time (seconds)')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Metrics plot saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")

class DatabaseMonitor:
    """Database performance monitoring"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.connection_stats = deque(maxlen=100)
        self.query_times = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
    
    def monitor_connection(self):
        """Monitor database connection health"""
        try:
            start_time = time.time()
            
            if 'sqlite' in self.db_url:
                conn = sqlite3.connect(self.db_url.replace('sqlite:///', ''))
                conn.close()
            elif 'mongodb' in self.db_url:
                # MongoDB connection check would go here
                pass
            
            connection_time = time.time() - start_time
            self.connection_stats.append((datetime.now(), connection_time))
            
            return connection_time
            
        except Exception as e:
            self.error_counts['connection_error'] += 1
            logger.error(f"Database connection error: {str(e)}")
            return None
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get database status"""
        try:
            # Test connection
            connection_time = self.monitor_connection()
            
            # Calculate metrics
            avg_connection_time = np.mean([s[1] for s in self.connection_stats]) if self.connection_stats else 0
            total_errors = sum(self.error_counts.values())
            
            status = {
                'connection_healthy': connection_time is not None,
                'connection_time': connection_time,
                'avg_connection_time': avg_connection_time,
                'total_connections': len(self.connection_stats),
                'error_summary': dict(self.error_counts),
                'total_errors': total_errors
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting database status: {str(e)}")
            return {'error': str(e)}

class RedisMonitor:
    """Redis performance monitoring"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.performance_metrics = deque(maxlen=100)
        
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()  # Test connection
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
    
    def get_redis_status(self) -> Dict[str, Any]:
        """Get Redis status"""
        try:
            if not self.redis_client:
                return {'status': 'not_connected', 'error': 'Redis client not available'}
            
            # Get Redis info
            info = self.redis_client.info()
            
            # Get performance metrics
            start_time = time.time()
            self.redis_client.ping()
            ping_time = time.time() - start_time
            
            self.performance_metrics.append((datetime.now(), ping_time))
            avg_ping_time = np.mean([m[1] for m in self.performance_metrics]) if self.performance_metrics else 0
            
            status = {
                'status': 'connected',
                'version': info.get('redis_version', 'unknown'),
                'uptime': info.get('uptime_in_seconds', 0),
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', 'unknown'),
                'ping_time': ping_time,
                'avg_ping_time': avg_ping_time,
                'total_commands': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting Redis status: {str(e)}")
            return {'status': 'error', 'error': str(e)}

# Global monitoring instance
system_monitor = SystemMonitor()
