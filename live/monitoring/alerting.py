"""
监控告警系统

提供实时监控、指标收集、告警触发等功能
"""

import json
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    # 系统相关
    SYSTEM_ERROR = "system_error"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"

    # 交易相关
    ORDER_REJECTED = "order_rejected"
    ORDER_FAILED = "order_failed"
    EXECUTION_DELAY = "execution_delay"

    # 风险相关
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"

    # 性能相关
    LOW_RETURN = "low_return"
    HIGH_VOLATILITY = "high_volatility"

    # 数据相关
    DATA_DELAY = "data_delay"
    DATA_MISSING = "data_missing"
    DATA_ERROR = "data_error"

    # 其他
    CUSTOM = "custom"


@dataclass
class Alert:
    """告警"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }

    def resolve(self):
        """解决告警"""
        self.resolved = True
        self.resolved_at = datetime.now()


class AlertChannel:
    """告警渠道基类"""

    def send(self, alert: Alert):
        """发送告警"""
        raise NotImplementedError


class ConsoleAlertChannel(AlertChannel):
    """控制台告警渠道"""

    def send(self, alert: Alert):
        """输出到控制台"""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        severity_icon = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }.get(alert.severity, "")

        logger.info(
            f"{severity_icon} [{alert.severity.value.upper()}] "
            f"{alert.title}: {alert.message}"
        )


class FileAlertChannel(AlertChannel):
    """文件告警渠道"""

    def __init__(self, file_path: str = "logs/alerts.jsonl"):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def send(self, alert: Alert):
        """写入文件"""
        with self.lock:
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(alert.to_dict(), ensure_ascii=False) + '\n')


class EmailAlertChannel(AlertChannel):
    """邮件告警渠道（占位）"""

    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
        logger.info("邮件告警渠道已初始化（未实现）")

    def send(self, alert: Alert):
        """发送邮件"""
        # TODO: 实现邮件发送
        logger.warning(f"邮件告警未实现: {alert.title}")


class WebhookAlertChannel(AlertChannel):
    """Webhook告警渠道"""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, alert: Alert):
        """发送Webhook"""
        try:
            import requests

            payload = {
                'alert': alert.to_dict(),
                'timestamp': datetime.now().isoformat(),
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
            )
            response.raise_for_status()

            logger.debug(f"Webhook发送成功: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Webhook发送失败: {e}")


class AlertRule:
    """告警规则"""

    def __init__(
        self,
        rule_id: str,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        alert_type: AlertType,
        severity: AlertSeverity,
        message_template: str,
        cooldown_seconds: int = 300,  # 默认5分钟冷却
    ):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.alert_type = alert_type
        self.severity = severity
        self.message_template = message_template
        self.cooldown_seconds = cooldown_seconds
        self.last_triggered = None

    def check(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        """检查规则"""
        # 检查冷却时间
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return None

        # 检查条件
        if self.condition(metrics):
            self.last_triggered = datetime.now()

            # 生成告警
            return Alert(
                alert_id=f"{self.rule_id}_{int(datetime.now().timestamp())}",
                alert_type=self.alert_type,
                severity=self.severity,
                title=self.name,
                message=self.message_template.format(**metrics),
                metadata={'rule_id': self.rule_id, 'metrics': metrics},
            )

        return None


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.rules: List[AlertRule] = []
        self.alert_history: List[Alert] = []
        self.alert_queue = queue.Queue()
        self.running = False
        self.worker_thread = None

        # 默认添加控制台渠道
        self.add_channel(ConsoleAlertChannel())

    def add_channel(self, channel: AlertChannel):
        """添加告警渠道"""
        self.channels.append(channel)
        logger.info(f"添加告警渠道: {channel.__class__.__name__}")

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")

    def trigger_alert(self, alert: Alert):
        """触发告警"""
        self.alert_history.append(alert)
        self.alert_queue.put(alert)

        logger.info(f"告警触发: [{alert.severity.value}] {alert.title}")

    def check_rules(self, metrics: Dict[str, Any]):
        """检查所有规则"""
        for rule in self.rules:
            alert = rule.check(metrics)
            if alert:
                self.trigger_alert(alert)

    def start(self):
        """启动告警处理"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.worker_thread.start()

        logger.info("告警管理器已启动")

    def stop(self):
        """停止告警处理"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

        logger.info("告警管理器已停止")

    def _process_alerts(self):
        """处理告警队列"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_alert(alert)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理告警失败: {e}")

    def _send_alert(self, alert: Alert):
        """发送告警到所有渠道"""
        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as e:
                logger.error(f"发送告警失败 ({channel.__class__.__name__}): {e}")

    def get_alert_history(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """获取告警历史"""
        alerts = self.alert_history

        # 过滤
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]

        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]

        # 排序和限制
        alerts = sorted(alerts, key=lambda a: a.timestamp, reverse=True)
        return alerts[:limit]


class MetricsCollector:
    """指标收集器"""

    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
        self.metrics: Dict[str, Any] = {}
        self.metrics_history: List[Dict[str, Any]] = []
        self.running = False
        self.worker_thread = None

    def update_metric(self, key: str, value: Any):
        """更新指标"""
        self.metrics[key] = value

    def update_metrics(self, metrics: Dict[str, Any]):
        """批量更新指标"""
        self.metrics.update(metrics)

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return self.metrics.copy()

    def start(self, interval_seconds: int = 60):
        """启动指标收集"""
        if self.running:
            return

        self.running = True
        self.worker_thread = threading.Thread(
            target=self._collect_loop,
            args=(interval_seconds,),
            daemon=True,
        )
        self.worker_thread.start()

        logger.info("指标收集器已启动")

    def stop(self):
        """停止指标收集"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

        logger.info("指标收集器已停止")

    def _collect_loop(self, interval_seconds: int):
        """收集循环"""
        import time

        while self.running:
            # 保存历史
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'metrics': self.metrics.copy(),
            }
            self.metrics_history.append(snapshot)

            # 限制历史大小
            if len(self.metrics_history) > 10000:
                self.metrics_history = self.metrics_history[-5000:]

            # 检查告警规则
            self.alert_manager.check_rules(self.metrics)

            time.sleep(interval_seconds)


def create_default_alert_manager() -> AlertManager:
    """创建默认告警管理器（包含常用规则）"""
    manager = AlertManager()

    # 添加文件渠道
    manager.add_channel(FileAlertChannel())

    # 添加常用告警规则

    # 1. 亏损告警
    manager.add_rule(AlertRule(
        rule_id="loss_limit",
        name="亏损限制告警",
        condition=lambda m: m.get('daily_pnl_ratio', 0) < -0.05,  # 单日亏损超过5%
        alert_type=AlertType.LOSS_LIMIT,
        severity=AlertSeverity.WARNING,
        message_template="单日亏损超过5%: {daily_pnl_ratio:.2%}",
        cooldown_seconds=3600,  # 1小时冷却
    ))

    # 2. 回撤告警
    manager.add_rule(AlertRule(
        rule_id="drawdown_limit",
        name="回撤限制告警",
        condition=lambda m: m.get('max_drawdown', 0) < -0.10,  # 最大回撤超过10%
        alert_type=AlertType.DRAWDOWN_LIMIT,
        severity=AlertSeverity.ERROR,
        message_template="最大回撤超过10%: {max_drawdown:.2%}",
        cooldown_seconds=1800,  # 30分钟冷却
    ))

    # 3. 持仓集中度告警
    manager.add_rule(AlertRule(
        rule_id="position_concentration",
        name="持仓集中度告警",
        condition=lambda m: m.get('max_position_ratio', 0) > 0.30,  # 单一持仓超过30%
        alert_type=AlertType.POSITION_LIMIT,
        severity=AlertSeverity.WARNING,
        message_template="单一持仓比例过高: {max_position_ratio:.2%}",
        cooldown_seconds=1800,
    ))

    # 4. 数据延迟告警
    manager.add_rule(AlertRule(
        rule_id="data_delay",
        name="数据延迟告警",
        condition=lambda m: m.get('data_delay_seconds', 0) > 300,  # 数据延迟超过5分钟
        alert_type=AlertType.DATA_DELAY,
        severity=AlertSeverity.ERROR,
        message_template="数据延迟: {data_delay_seconds}秒",
        cooldown_seconds=600,
    ))

    return manager


__all__ = [
    'AlertSeverity',
    'AlertType',
    'Alert',
    'AlertChannel',
    'ConsoleAlertChannel',
    'FileAlertChannel',
    'EmailAlertChannel',
    'WebhookAlertChannel',
    'AlertRule',
    'AlertManager',
    'MetricsCollector',
    'create_default_alert_manager',
]
