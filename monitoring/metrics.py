"""
监控指标模块
提供实时指标收集、分析和异常检测功能
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics
import threading
import time

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'tags': self.tags,
        }


class Metric:
    """
    指标类

    存储和管理单个指标的历史数据
    """

    def __init__(
        self,
        name: str,
        max_points: int = 1000,
        aggregation_window: Optional[int] = None,
    ):
        """
        Args:
            name: 指标名称
            max_points: 最大数据点数
            aggregation_window: 聚合窗口大小
        """
        self.name = name
        self.max_points = max_points
        self.aggregation_window = aggregation_window

        self.data_points: deque = deque(maxlen=max_points)
        self.lock = threading.Lock()

    def add(self, value: float, tags: Optional[Dict[str, str]] = None):
        """
        添加数据点

        Args:
            value: 指标值
            tags: 标签
        """
        with self.lock:
            point = MetricPoint(
                timestamp=datetime.now(),
                value=value,
                tags=tags or {},
            )
            self.data_points.append(point)

    def get_latest(self) -> Optional[float]:
        """获取最新值"""
        with self.lock:
            if not self.data_points:
                return None
            return self.data_points[-1].value

    def get_values(self, n: Optional[int] = None) -> List[float]:
        """
        获取最近的值

        Args:
            n: 数量，None表示全部

        Returns:
            值列表
        """
        with self.lock:
            values = [p.value for p in self.data_points]
            if n is not None:
                return values[-n:]
            return values

    def get_mean(self, n: Optional[int] = None) -> Optional[float]:
        """获取均值"""
        values = self.get_values(n)
        if not values:
            return None
        return statistics.mean(values)

    def get_std(self, n: Optional[int] = None) -> Optional[float]:
        """获取标准差"""
        values = self.get_values(n)
        if len(values) < 2:
            return None
        return statistics.stdev(values)

    def get_percentile(self, percentile: float, n: Optional[int] = None) -> Optional[float]:
        """
        获取百分位数

        Args:
            percentile: 百分位数 (0-100)
            n: 数据点数量

        Returns:
            百分位数值
        """
        values = self.get_values(n)
        if not values:
            return None
        return statistics.quantiles(values, n=100)[int(percentile)] if len(values) > 1 else values[0]

    def get_rate(self, window: int = 60) -> Optional[float]:
        """
        获取变化率

        Args:
            window: 时间窗口（秒）

        Returns:
            变化率（每秒）
        """
        with self.lock:
            if len(self.data_points) < 2:
                return None

            # 获取窗口内的数据
            now = datetime.now()
            cutoff = now - timedelta(seconds=window)

            window_points = [p for p in self.data_points if p.timestamp >= cutoff]

            if len(window_points) < 2:
                return None

            first_value = window_points[0].value
            last_value = window_points[-1].value

            time_diff = (window_points[-1].timestamp - window_points[0].timestamp).total_seconds()

            if time_diff == 0:
                return None

            return (last_value - first_value) / time_diff

    def clear(self):
        """清空数据"""
        with self.lock:
            self.data_points.clear()


class MetricsCollector:
    """
    指标收集器

    收集和管理多个指标
    """

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.Lock()

    def get_or_create_metric(
        self,
        name: str,
        max_points: int = 1000,
    ) -> Metric:
        """获取或创建指标"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name, max_points)
                logger.debug(f"创建指标: {name}")
            return self.metrics[name]

    def record(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        记录指标

        Args:
            name: 指标名称
            value: 指标值
            tags: 标签
        """
        metric = self.get_or_create_metric(name)
        metric.add(value, tags)

    def increment(self, name: str, delta: float = 1.0):
        """
        增加计数器

        Args:
            name: 指标名称
            delta: 增量
        """
        metric = self.get_or_create_metric(name)
        current = metric.get_latest() or 0
        metric.add(current + delta)

    def get_metric(self, name: str) -> Optional[Metric]:
        """获取指标"""
        with self.lock:
            return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """获取所有指标"""
        with self.lock:
            return self.metrics.copy()


class AnomalyDetector:
    """
    异常检测器

    检测指标异常值
    """

    def __init__(
        self,
        std_threshold: float = 3.0,
        percentile_threshold: float = 95.0,
        min_samples: int = 30,
    ):
        """
        Args:
            std_threshold: 标准差阈值
            percentile_threshold: 百分位阈值
            min_samples: 最小样本数
        """
        self.std_threshold = std_threshold
        self.percentile_threshold = percentile_threshold
        self.min_samples = min_samples

    def detect_std(self, metric: Metric) -> Optional[bool]:
        """
        使用标准差检测异常

        Args:
            metric: 指标对象

        Returns:
            是否异常，None表示数据不足
        """
        values = metric.get_values()
        if len(values) < self.min_samples:
            return None

        mean = metric.get_mean()
        std = metric.get_std()

        if std is None or std == 0:
            return None

        latest = metric.get_latest()

        # 检查是否超出阈值
        z_score = abs(latest - mean) / std
        return z_score > self.std_threshold

    def detect_percentile(self, metric: Metric) -> Optional[bool]:
        """
        使用百分位数检测异常

        Args:
            metric: 指标对象

        Returns:
            是否异常，None表示数据不足
        """
        values = metric.get_values()
        if len(values) < self.min_samples:
            return None

        if NUMPY_AVAILABLE:
            threshold = np.percentile(values[:-1], self.percentile_threshold)
        else:
            threshold = statistics.quantiles(values[:-1], n=100)[int(self.percentile_threshold)]

        latest = metric.get_latest()

        return latest > threshold

    def detect_change(self, metric: Metric, change_threshold: float = 0.5) -> Optional[bool]:
        """
        检测剧烈变化

        Args:
            metric: 指标对象
            change_threshold: 变化阈值（比例）

        Returns:
            是否异常
        """
        values = metric.get_values()
        if len(values) < 2:
            return None

        latest = values[-1]
        previous = values[-2]

        if previous == 0:
            return None

        change = abs(latest - previous) / abs(previous)
        return change > change_threshold


class PerformanceMonitor:
    """
    性能监控器

    监控策略和系统性能指标
    """

    def __init__(self):
        self.collector = MetricsCollector()
        self.detector = AnomalyDetector()
        self.alert_callbacks: List[Callable] = []

    def record_return(self, strategy_name: str, return_value: float):
        """记录收益率"""
        self.collector.record(
            f"strategy.{strategy_name}.return",
            return_value,
            tags={'strategy': strategy_name},
        )

    def record_drawdown(self, strategy_name: str, drawdown: float):
        """记录回撤"""
        self.collector.record(
            f"strategy.{strategy_name}.drawdown",
            drawdown,
            tags={'strategy': strategy_name},
        )

    def record_position_value(self, strategy_name: str, value: float):
        """记录持仓价值"""
        self.collector.record(
            f"strategy.{strategy_name}.position_value",
            value,
            tags={'strategy': strategy_name},
        )

    def record_order_latency(self, latency_ms: float):
        """记录订单延迟"""
        self.collector.record(
            "system.order_latency",
            latency_ms,
        )

    def record_system_cpu(self, cpu_percent: float):
        """记录CPU使用率"""
        self.collector.record(
            "system.cpu",
            cpu_percent,
        )

    def record_system_memory(self, memory_percent: float):
        """记录内存使用率"""
        self.collector.record(
            "system.memory",
            memory_percent,
        )

    def check_anomalies(self) -> List[Dict[str, Any]]:
        """
        检查所有指标的异常

        Returns:
            异常列表
        """
        anomalies = []

        for metric_name, metric in self.collector.get_all_metrics().items():
            # 标准差检测
            is_anomaly = self.detector.detect_std(metric)
            if is_anomaly:
                anomalies.append({
                    'metric': metric_name,
                    'type': 'std',
                    'value': metric.get_latest(),
                    'mean': metric.get_mean(),
                    'std': metric.get_std(),
                    'timestamp': datetime.now().isoformat(),
                })

            # 百分位数检测
            is_anomaly = self.detector.detect_percentile(metric)
            if is_anomaly:
                anomalies.append({
                    'metric': metric_name,
                    'type': 'percentile',
                    'value': metric.get_latest(),
                    'threshold': self.detector.percentile_threshold,
                    'timestamp': datetime.now().isoformat(),
                })

        # 触发告警回调
        for anomaly in anomalies:
            for callback in self.alert_callbacks:
                try:
                    callback(anomaly)
                except Exception as e:
                    logger.error(f"异常告警回调失败: {e}")

        return anomalies

    def add_alert_callback(self, callback: Callable):
        """添加异常告警回调"""
        self.alert_callbacks.append(callback)

    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指标摘要

        Args:
            metric_name: 指标名称

        Returns:
            摘要字典
        """
        metric = self.collector.get_metric(metric_name)
        if metric is None:
            return None

        return {
            'name': metric_name,
            'latest': metric.get_latest(),
            'mean': metric.get_mean(),
            'std': metric.get_std(),
            'count': len(metric.get_values()),
        }

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """获取所有指标摘要"""
        summaries = {}
        for metric_name in self.collector.get_all_metrics().keys():
            summary = self.get_metric_summary(metric_name)
            if summary:
                summaries[metric_name] = summary
        return summaries


# 全局性能监控器实例
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


__all__ = [
    'MetricPoint',
    'Metric',
    'MetricsCollector',
    'AnomalyDetector',
    'PerformanceMonitor',
    'get_performance_monitor',
]
