"""
Monitoring and Logging Module

This module provides comprehensive monitoring and logging capabilities for the customer segmentation pipeline.
Includes performance monitoring, error tracking, and system health checks.
"""

import logging
import structlog
import time
import psutil
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import json
import os
from pathlib import Path


class PerformanceMonitor:
    """Monitor performance of functions and operations."""
    
    def __init__(self, log_file: str = None):
        """
        Initialize performance monitor.
        
        Args:
            log_file (str): Path to performance log file
        """
        self.log_file = log_file or '../logs/performance.log'
        self.metrics = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Setup structured logging."""
        # Ensure log directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
    
    def timing_decorator(self, operation_name: str = None):
        """
        Decorator to measure function execution time.
        
        Args:
            operation_name (str): Name of operation for logging
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    self.logger.info(
                        "operation_completed",
                        operation=name,
                        execution_time=execution_time,
                        status="success"
                    )
                    
                    # Store metric
                    if name not in self.metrics:
                        self.metrics[name] = []
                    self.metrics[name].append(execution_time)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    self.logger.error(
                        "operation_failed",
                        operation=name,
                        execution_time=execution_time,
                        status="error",
                        error=str(e)
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def log_system_metrics(self):
        """Log current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.used / disk.total * 100
            
            self.logger.info(
                "system_metrics",
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                memory_available_gb=memory.available / (1024**3),
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(
                "system_metrics_error",
                error=str(e)
            )
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times),
                    'p95_time': np.percentile(times, 95),
                    'p99_time': np.percentile(times, 99)
                }
        
        return summary


class ModelMonitor:
    """Monitor model performance and drift."""
    
    def __init__(self, log_file: str = None):
        """
        Initialize model monitor.
        
        Args:
            log_file (str): Path to model monitoring log file
        """
        self.log_file = log_file or '../logs/model_monitoring.log'
        self.logger = structlog.get_logger()
        self.baseline_metrics = {}
        self.current_metrics = {}
    
    def log_prediction(self, customer_id: str, cluster: int, 
                     confidence: float = None, features: Dict = None):
        """
        Log individual prediction.
        
        Args:
            customer_id (str): Customer identifier
            cluster (int): Predicted cluster
            confidence (float): Prediction confidence
            features (Dict): Input features
        """
        self.logger.info(
            "prediction_made",
            customer_id=customer_id,
            cluster=cluster,
            confidence=confidence,
            feature_count=len(features) if features else 0,
            timestamp=datetime.now().isoformat()
        )
    
    def log_batch_prediction(self, total_customers: int, processing_time: float,
                          accuracy_metrics: Dict = None):
        """
        Log batch prediction metrics.
        
        Args:
            total_customers (int): Number of customers processed
            processing_time (float): Total processing time
            accuracy_metrics (Dict): Accuracy metrics if available
        """
        throughput = total_customers / processing_time if processing_time > 0 else 0
        
        self.logger.info(
            "batch_prediction_completed",
            total_customers=total_customers,
            processing_time=processing_time,
            throughput=throughput,
            accuracy_metrics=accuracy_metrics,
            timestamp=datetime.now().isoformat()
        )
    
    def detect_drift(self, current_data: pd.DataFrame, 
                     reference_data: pd.DataFrame = None) -> Dict:
        """
        Detect data drift in customer features.
        
        Args:
            current_data (pd.DataFrame): Current customer data
            reference_data (pd.DataFrame): Reference/historical data
            
        Returns:
            Dict: Drift detection results
        """
        if reference_data is None:
            return {"error": "No reference data provided for drift detection"}
        
        drift_results = {}
        
        # Compare distributions for key features
        key_features = ['Recency', 'Frequency', 'Monetary']
        
        for feature in key_features:
            if feature in current_data.columns and feature in reference_data.columns:
                current_stats = current_data[feature].describe()
                ref_stats = reference_data[feature].describe()
                
                # Calculate drift metrics
                mean_drift = abs(current_stats['mean'] - ref_stats['mean']) / ref_stats['mean']
                std_drift = abs(current_stats['std'] - ref_stats['std']) / ref_stats['std']
                
                drift_results[feature] = {
                    'mean_drift': mean_drift,
                    'std_drift': std_drift,
                    'current_mean': current_stats['mean'],
                    'reference_mean': ref_stats['mean'],
                    'current_std': current_stats['std'],
                    'reference_std': ref_stats['std']
                }
        
        # Overall drift assessment
        drift_threshold = 0.2  # 20% change threshold
        significant_drift = [
            feature for feature, metrics in drift_results.items()
            if metrics['mean_drift'] > drift_threshold
        ]
        
        drift_results['overall'] = {
            'significant_drift_features': significant_drift,
            'drift_detected': len(significant_drift) > 0,
            'drift_threshold': drift_threshold
        }
        
        self.logger.info(
            "drift_detection_completed",
            drift_results=drift_results,
            timestamp=datetime.now().isoformat()
        )
        
        return drift_results


class HealthChecker:
    """Monitor system and application health."""
    
    def __init__(self):
        """Initialize health checker."""
        self.logger = structlog.get_logger()
        self.health_checks = {}
    
    def register_health_check(self, name: str, check_func: Callable):
        """
        Register a health check function.
        
        Args:
            name (str): Name of health check
            check_func (Callable): Function that returns health status
        """
        self.health_checks[name] = check_func
    
    def check_model_health(self) -> Dict:
        """Check if models are loaded and functional."""
        try:
            from config import get_file_path
            import joblib
            
            # Check model files exist
            model_path = get_file_path('kmeans_model')
            scaler_path = get_file_path('scaler_model')
            
            model_exists = os.path.exists(model_path)
            scaler_exists = os.path.exists(scaler_path)
            
            # Try to load models
            model_loaded = False
            if model_exists:
                try:
                    joblib.load(model_path)
                    model_loaded = True
                except:
                    pass
            
            health_status = {
                'status': 'healthy' if model_exists and scaler_exists and model_loaded else 'unhealthy',
                'model_file_exists': model_exists,
                'scaler_file_exists': scaler_exists,
                'model_loadable': model_loaded,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("model_health_check", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.error("model_health_check_error", **error_status)
            return error_status
    
    def check_data_health(self) -> Dict:
        """Check data availability and quality."""
        try:
            from config import get_file_path
            import pandas as pd
            
            # Check key data files
            data_files = {
                'raw_data': get_file_path('raw_data'),
                'processed_data': get_file_path('processed_data'),
                'customer_features': get_file_path('features'),
                'customer_segments': get_file_path('segments')
            }
            
            file_status = {}
            for name, path in data_files.items():
                if os.path.exists(path):
                    try:
                        df = pd.read_csv(path)
                        file_status[name] = {
                            'exists': True,
                            'readable': True,
                            'rows': len(df),
                            'columns': len(df.columns),
                            'size_mb': os.path.getsize(path) / (1024*1024)
                        }
                    except Exception as e:
                        file_status[name] = {
                            'exists': True,
                            'readable': False,
                            'error': str(e)
                        }
                else:
                    file_status[name] = {
                        'exists': False,
                        'readable': False
                    }
            
            # Overall health
            all_exist = all(status['exists'] for status in file_status.values())
            all_readable = all(status.get('readable', False) for status in file_status.values())
            
            health_status = {
                'status': 'healthy' if all_exist and all_readable else 'unhealthy',
                'files': file_status,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("data_health_check", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.error("data_health_check_error", **error_status)
            return error_status
    
    def check_system_health(self) -> Dict:
        """Check system resources."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk space
            disk = psutil.disk_usage('/')
            
            # Determine health status
            cpu_healthy = cpu_percent < 80
            memory_healthy = memory.percent < 80
            disk_healthy = (disk.used / disk.total) < 80
            
            overall_healthy = cpu_healthy and memory_healthy and disk_healthy
            
            health_status = {
                'status': 'healthy' if overall_healthy else 'degraded',
                'cpu': {
                    'percent_used': cpu_percent,
                    'healthy': cpu_healthy
                },
                'memory': {
                    'percent_used': memory.percent,
                    'available_gb': memory.available / (1024**3),
                    'healthy': memory_healthy
                },
                'disk': {
                    'percent_used': (disk.used / disk.total) * 100,
                    'available_gb': disk.free / (1024**3),
                    'healthy': disk_healthy
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("system_health_check", **health_status)
            return health_status
            
        except Exception as e:
            error_status = {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.logger.error("system_health_check_error", **error_status)
            return error_status
    
    def run_all_checks(self) -> Dict:
        """Run all registered health checks."""
        results = {
            'overall_status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Built-in checks
        results['checks']['model'] = self.check_model_health()
        results['checks']['data'] = self.check_data_health()
        results['checks']['system'] = self.check_system_health()
        
        # Custom checks
        for name, check_func in self.health_checks.items():
            try:
                results['checks'][name] = check_func()
            except Exception as e:
                results['checks'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
        
        # Determine overall status
        unhealthy_checks = [
            name for name, check in results['checks'].items()
            if check.get('status') == 'unhealthy'
        ]
        
        if unhealthy_checks:
            results['overall_status'] = 'unhealthy'
            results['unhealthy_checks'] = unhealthy_checks
        else:
            degraded_checks = [
                name for name, check in results['checks'].items()
                if check.get('status') == 'degraded'
            ]
            if degraded_checks:
                results['overall_status'] = 'degraded'
                results['degraded_checks'] = degraded_checks
        
        self.logger.info("health_check_completed", **results)
        return results


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, config_file: str = None):
        """
        Initialize alert manager.
        
        Args:
            config_file (str): Path to alert configuration
        """
        self.logger = structlog.get_logger()
        self.alert_rules = self.load_alert_config(config_file)
        self.alert_history = []
    
    def load_alert_config(self, config_file: str = None) -> Dict:
        """Load alert configuration."""
        default_config = {
            'performance': {
                'max_execution_time': 300,  # 5 minutes
                'max_memory_usage': 80,  # 80%
                'max_cpu_usage': 80  # 80%
            },
            'model': {
                'min_confidence_threshold': 0.5,
                'max_drift_threshold': 0.2
            },
            'data': {
                'min_data_quality_score': 0.8,
                'max_missing_percentage': 10
            }
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for category, rules in user_config.items():
                    if category in default_config:
                        default_config[category].update(rules)
            except Exception as e:
                self.logger.error("alert_config_load_error", error=str(e))
        
        return default_config
    
    def check_performance_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for performance-related alerts."""
        alerts = []
        rules = self.alert_rules.get('performance', {})
        
        # Execution time alerts
        for operation, stats in metrics.items():
            if stats['avg_time'] > rules.get('max_execution_time', 300):
                alerts.append({
                    'type': 'performance',
                    'severity': 'warning',
                    'message': f"High execution time for {operation}: {stats['avg_time']:.2f}s",
                    'metric': 'execution_time',
                    'value': stats['avg_time'],
                    'threshold': rules.get('max_execution_time'),
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def send_alert(self, alert: Dict):
        """Send alert notification."""
        self.alert_history.append(alert)
        
        self.logger.warning(
            "alert_triggered",
            alert_type=alert['type'],
            severity=alert['severity'],
            message=alert['message'],
            timestamp=alert['timestamp']
        )
        
        # Here you could add email, Slack, or other notification methods
        print(f"ALERT: {alert['severity'].upper()} - {alert['message']}")


# Initialize monitoring components
performance_monitor = PerformanceMonitor()
model_monitor = ModelMonitor()
health_checker = HealthChecker()
alert_manager = AlertManager()

# Decorator for easy performance monitoring
def monitor_performance(operation_name: str = None):
    """Decorator for performance monitoring."""
    return performance_monitor.timing_decorator(operation_name)


if __name__ == "__main__":
    # Example usage
    print("Monitoring Module")
    print("=" * 30)
    
    # Setup logging
    monitor = PerformanceMonitor()
    
    # Example with decorator
    @monitor.timing_decorator("example_operation")
    def example_function():
        time.sleep(2)
        return "completed"
    
    # Run example
    result = example_function()
    
    # Log system metrics
    monitor.log_system_metrics()
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print("Performance Summary:")
    print(json.dumps(summary, indent=2))
    
    # Health check
    health = health_checker.run_all_checks()
    print("\nHealth Check:")
    print(json.dumps(health, indent=2))
