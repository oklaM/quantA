"""
实时监控面板

提供实时监控、数据展示、告警查看等功能
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import threading
import time

from live.monitoring.alerting import AlertManager, Alert
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SystemStatus:
    """系统状态"""
    status: str = "running"  # running, stopped, error
    uptime_seconds: float = 0
    cpu_percent: float = 0
    memory_percent: float = 0
    disk_percent: float = 0
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'uptime_seconds': self.uptime_seconds,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'disk_percent': self.disk_percent,
            'last_update': self.last_update.isoformat(),
        }


@dataclass
class TradingStatus:
    """交易状态"""
    is_trading: bool = False
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    pending_orders: int = 0
    last_order_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_trading': self.is_trading,
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'rejected_orders': self.rejected_orders,
            'pending_orders': self.pending_orders,
            'fill_rate': self.filled_orders / self.total_orders if self.total_orders > 0 else 0,
            'last_order_time': self.last_order_time.isoformat() if self.last_order_time else None,
        }


@dataclass
class PerformanceStatus:
    """绩效状态"""
    total_value: float = 0
    initial_capital: float = 0
    daily_pnl: float = 0
    daily_pnl_ratio: float = 0
    total_pnl: float = 0
    total_pnl_ratio: float = 0
    max_drawdown: float = 0
    sharpe_ratio: float = 0
    win_rate: float = 0
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_value': self.total_value,
            'initial_capital': self.initial_capital,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_ratio': self.daily_pnl_ratio,
            'total_pnl': self.total_pnl,
            'total_pnl_ratio': self.total_pnl_ratio,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'last_update': self.last_update.isoformat(),
        }


class Monitor:
    """监控器"""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.system_status = SystemStatus()
        self.trading_status = TradingStatus()
        self.performance_status = PerformanceStatus()
        self.start_time = datetime.now()

        self.running = False
        self.update_thread = None

    def start(self, update_interval: int = 5):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.start_time = datetime.now()

        self.update_thread = threading.Thread(
            target=self._update_loop,
            args=(update_interval,),
            daemon=True,
        )
        self.update_thread.start()

        logger.info("监控器已启动")

    def stop(self):
        """停止监控"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=10)

        logger.info("监控器已停止")

    def _update_loop(self, interval: int):
        """更新循环"""
        while self.running:
            self._update_system_status()
            time.sleep(interval)

    def _update_system_status(self):
        """更新系统状态"""
        import psutil

        # 运行时间
        uptime = (datetime.now() - self.start_time).total_seconds()
        self.system_status.uptime_seconds = uptime

        # 系统资源
        self.system_status.cpu_percent = psutil.cpu_percent(interval=1)
        self.system_status.memory_percent = psutil.virtual_memory().percent
        self.system_status.disk_percent = psutil.disk_usage('/').percent

        self.system_status.last_update = datetime.now()

    def update_trading_status(
        self,
        is_trading: bool,
        total_orders: int,
        filled_orders: int,
        rejected_orders: int,
        pending_orders: int,
    ):
        """更新交易状态"""
        self.trading_status.is_trading = is_trading
        self.trading_status.total_orders = total_orders
        self.trading_status.filled_orders = filled_orders
        self.trading_status.rejected_orders = rejected_orders
        self.trading_status.pending_orders = pending_orders

    def update_performance_status(
        self,
        total_value: float,
        initial_capital: float,
        daily_pnl: float,
        daily_pnl_ratio: float,
        total_pnl: float,
        total_pnl_ratio: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
    ):
        """更新绩效状态"""
        self.performance_status.total_value = total_value
        self.performance_status.initial_capital = initial_capital
        self.performance_status.daily_pnl = daily_pnl
        self.performance_status.daily_pnl_ratio = daily_pnl_ratio
        self.performance_status.total_pnl = total_pnl
        self.performance_status.total_pnl_ratio = total_pnl_ratio
        self.performance_status.max_drawdown = max_drawdown
        self.performance_status.sharpe_ratio = sharpe_ratio
        self.performance_status.win_rate = win_rate
        self.performance_status.last_update = datetime.now()

    def get_status(self) -> Dict[str, Any]:
        """获取完整状态"""
        return {
            'system': self.system_status.to_dict(),
            'trading': self.trading_status.to_dict(),
            'performance': self.performance_status.to_dict(),
            'alerts': {
                'total': len(self.alert_manager.alert_history),
                'unresolved': len([a for a in self.alert_manager.alert_history if not a.resolved]),
                'recent': [
                    {
                        'type': a.alert_type.value,
                        'severity': a.severity.value,
                        'title': a.title,
                        'message': a.message,
                        'timestamp': a.timestamp.isoformat(),
                    }
                    for a in self.alert_manager.get_alert_history(limit=10)
                ],
            },
            'timestamp': datetime.now().isoformat(),
        }

    def get_summary(self) -> str:
        """获取状态摘要"""
        status = self.get_status()

        summary = f"""
========================================
quantA 监控面板
========================================

系统状态:
  状态: {status['system']['status']}
  运行时间: {status['system']['uptime_seconds'] / 3600:.1f} 小时
  CPU: {status['system']['cpu_percent']:.1f}%
  内存: {status['system']['memory_percent']:.1f}%
  磁盘: {status['system']['disk_percent']:.1f}%

交易状态:
  交易中: {status['trading']['is_trading']}
  总订单: {status['trading']['total_orders']}
  成交: {status['trading']['filled_orders']}
  拒绝: {status['trading']['rejected_orders']}
  成交率: {status['trading']['fill_rate']:.1%}

绩效状态:
  总资产: ¥{status['performance']['total_value']:,.2f}
  今日盈亏: ¥{status['performance']['daily_pnl']:,.2f} ({status['performance']['daily_pnl_ratio']:.2%})
  总盈亏: ¥{status['performance']['total_pnl']:,.2f} ({status['performance']['total_pnl_ratio']:.2%})
  最大回撤: {status['performance']['max_drawdown']:.2%}
  夏普比率: {status['performance']['sharpe_ratio']:.2f}
  胜率: {status['performance']['win_rate']:.1%}

告警状态:
  总告警: {status['alerts']['total']}
  未解决: {status['alerts']['unresolved']}

最后更新: {status['timestamp']}
========================================
        """

        return summary.strip()


class MetricsCollector:
    """指标收集器（用于监控）"""

    def __init__(self, monitor: Monitor):
        self.monitor = monitor
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def collect(self):
        """收集当前指标"""
        status = self.monitor.get_status()

        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'system': status['system'],
            'trading': status['trading'],
            'performance': status['performance'],
        }

        self.metrics_history.append(snapshot)

        # 限制历史大小
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history // 2:]

        return snapshot

    def get_metrics_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """获取历史指标"""
        metrics = self.metrics_history

        if start_time:
            metrics = [m for m in metrics if datetime.fromisoformat(m['timestamp']) >= start_time]

        if end_time:
            metrics = [m for m in metrics if datetime.fromisoformat(m['timestamp']) <= end_time]

        return metrics

    def export_metrics(self, file_path: str):
        """导出指标到文件"""
        import json

        with open(file_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"指标已导出到: {file_path}")


def create_monitor(alert_manager: AlertManager) -> Monitor:
    """创建监控器"""
    return Monitor(alert_manager)


__all__ = [
    'SystemStatus',
    'TradingStatus',
    'PerformanceStatus',
    'Monitor',
    'MetricsCollector',
    'create_monitor',
]
