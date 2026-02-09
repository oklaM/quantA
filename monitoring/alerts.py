"""
监控告警模块
提供实时监控、异常检测和多渠道告警功能
"""

import json
import queue
import smtplib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from utils.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    # 策略相关
    STRATEGY_ERROR = "strategy_error"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_PERFORMANCE = "strategy_performance"

    # 风险相关
    HIGH_DRAWDOWN = "high_drawdown"
    POSITION_LIMIT = "position_limit"
    LOSS_LIMIT = "loss_limit"
    VOLATILITY_SPIKE = "volatility_spike"

    # 系统相关
    SYSTEM_ERROR = "system_error"
    DATA_DELAY = "data_delay"
    CONNECTION_ERROR = "connection_error"
    RESOURCE_HIGH = "resource_high"

    # 交易相关
    ORDER_REJECTED = "order_rejected"
    ORDER_FAILED = "order_failed"
    EXECUTION_DELAY = "execution_delay"


@dataclass
class Alert:
    """告警对象"""
    alert_id: str
    alert_type: AlertType
    level: AlertLevel
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
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
        }


class AlertChannel:
    """告警渠道基类"""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True

    def send(self, alert: Alert) -> bool:
        """
        发送告警

        Args:
            alert: 告警对象

        Returns:
            是否发送成功
        """
        raise NotImplementedError("子类必须实现send方法")

    def enable(self):
        """启用渠道"""
        self.enabled = True
        logger.info(f"告警渠道已启用: {self.name}")

    def disable(self):
        """禁用渠道"""
        self.enabled = False
        logger.info(f"告警渠道已禁用: {self.name}")


class LogChannel(AlertChannel):
    """日志告警渠道"""

    def __init__(self):
        super().__init__("log")

    def send(self, alert: Alert) -> bool:
        """记录到日志"""
        log_msg = f"[告警] {alert.title}: {alert.message}"

        if alert.level == AlertLevel.INFO:
            logger.info(log_msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_msg)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_msg)
        elif alert.level == AlertLevel.CRITICAL:
            logger.critical(log_msg)

        return True


class EmailChannel(AlertChannel):
    """邮件告警渠道"""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
    ):
        """
        Args:
            smtp_host: SMTP服务器地址
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            from_addr: 发件人地址
            to_addrs: 收件人地址列表
        """
        super().__init__("email")
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send(self, alert: Alert) -> bool:
        """发送邮件"""
        if not self.enabled:
            return False

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"

            # 邮件正文
            body = f"""
告警时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
告警级别: {alert.level.value.upper()}
告警类型: {alert.alert_type.value}

{alert.message}

详细信息:
{json.dumps(alert.metadata, indent=2, ensure_ascii=False)}
"""
            msg.attach(MIMEText(body, 'plain'))

            # 发送邮件
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"邮件告警已发送: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Webhook告警渠道"""

    def __init__(self, url: str, method: str = "POST"):
        """
        Args:
            url: Webhook URL
            method: HTTP方法
        """
        super().__init__("webhook")
        self.url = url
        self.method = method

    def send(self, alert: Alert) -> bool:
        """发送Webhook"""
        if not self.enabled or not REQUESTS_AVAILABLE:
            return False

        try:
            payload = {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata,
            }

            response = requests.request(
                method=self.method,
                url=self.url,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"Webhook告警已发送: {alert.title}")
                return True
            else:
                logger.warning(f"Webhook返回错误: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
            return False


class DingTalkChannel(AlertChannel):
    """钉钉告警渠道"""

    def __init__(self, webhook_url: str):
        """
        Args:
            webhook_url: 钉钉机器人Webhook URL
        """
        super().__init__("dingtalk")
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> bool:
        """发送钉钉消息"""
        if not self.enabled or not REQUESTS_AVAILABLE:
            return False

        try:
            # 构建消息
            emoji_map = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
            }

            emoji = emoji_map.get(alert.level, "⚠️")

            text = f"""
{emoji} **{alert.title}**

**告警级别**: {alert.level.value.upper()}
**告警类型**: {alert.alert_type.value}
**告警时间**: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.message}
"""

            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": alert.title,
                    "text": text,
                },
            }

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') == 0:
                    logger.info(f"钉钉告警已发送: {alert.title}")
                    return True
                else:
                    logger.warning(f"钉钉返回错误: {result.get('errmsg')}")
                    return False
            else:
                logger.warning(f"钉钉返回错误: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"发送钉钉告警失败: {e}")
            return False


class SlackChannel(AlertChannel):
    """Slack告警渠道"""

    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        """
        Args:
            webhook_url: Slack Webhook URL
            channel: 频道名称（可选）
        """
        super().__init__("slack")
        self.webhook_url = webhook_url
        self.channel = channel

    def send(self, alert: Alert) -> bool:
        """发送Slack消息"""
        if not self.enabled or not REQUESTS_AVAILABLE:
            return False

        try:
            color_map = {
                AlertLevel.INFO: "#36a64f",
                AlertLevel.WARNING: "#ff9900",
                AlertLevel.ERROR: "#ff0000",
                AlertLevel.CRITICAL: "#990000",
            }

            color = color_map.get(alert.level, "#ff9900")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {
                                "title": "级别",
                                "value": alert.level.value.upper(),
                                "short": True,
                            },
                            {
                                "title": "类型",
                                "value": alert.alert_type.value,
                                "short": True,
                            },
                            {
                                "title": "时间",
                                "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                "short": False,
                            },
                        ],
                        "footer": "quantA Trading System",
                        "ts": int(alert.alert.timestamp()),
                    }
                ]
            }

            if self.channel:
                payload["channel"] = self.channel

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info(f"Slack告警已发送: {alert.title}")
                return True
            else:
                logger.warning(f"Slack返回错误: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"发送Slack告警失败: {e}")
            return False


class AlertManager:
    """
    告警管理器

    管理告警规则、分发告警到各渠道
    """

    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, Callable] = {}
        self._alert_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # 默认添加日志渠道
        self.add_channel(LogChannel())

        logger.info("告警管理器初始化完成")

    def add_channel(self, channel: AlertChannel):
        """添加告警渠道"""
        self.channels.append(channel)
        logger.info(f"添加告警渠道: {channel.name}")

    def remove_channel(self, channel_name: str):
        """移除告警渠道"""
        self.channels = [c for c in self.channels if c.name != channel_name]
        logger.info(f"移除告警渠道: {channel_name}")

    def add_rule(self, rule_name: str, rule_func: Callable):
        """
        添加告警规则

        Args:
            rule_name: 规则名称
            rule_func: 规则函数，返回Alert对象或None
        """
        self.alert_rules[rule_name] = rule_func
        logger.info(f"添加告警规则: {rule_name}")

    def remove_rule(self, rule_name: str):
        """移除告警规则"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"移除告警规则: {rule_name}")

    def check_rules(self, context: Dict[str, Any]):
        """
        检查所有告警规则

        Args:
            context: 上下文数据
        """
        for rule_name, rule_func in self.alert_rules.items():
            try:
                alert = rule_func(context)
                if alert is not None:
                    self.send_alert(alert)
            except Exception as e:
                logger.error(f"检查告警规则失败 {rule_name}: {e}")

    def send_alert(self, alert: Alert):
        """
        发送告警

        Args:
            alert: 告警对象
        """
        # 添加到历史
        self.alert_history.append(alert)

        # 添加到队列（异步发送）
        self._alert_queue.put(alert)

    def start(self):
        """启动告警处理线程"""
        if self._running:
            logger.warning("告警处理线程已在运行")
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self._worker_thread.start()
        logger.info("告警处理线程已启动")

    def stop(self):
        """停止告警处理线程"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("告警处理线程已停止")

    def _process_alerts(self):
        """处理告警队列（后台线程）"""
        while self._running:
            try:
                alert = self._alert_queue.get(timeout=1)

                # 发送到所有启用的渠道
                for channel in self.channels:
                    if channel.enabled:
                        try:
                            channel.send(alert)
                        except Exception as e:
                            logger.error(f"渠道 {channel.name} 发送失败: {e}")

                self._alert_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理告警失败: {e}")

    def get_alert_history(
        self,
        alert_type: Optional[AlertType] = None,
        level: Optional[AlertLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        获取告警历史

        Args:
            alert_type: 告警类型过滤
            level: 告警级别过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制

        Returns:
            告警列表
        """
        filtered = self.alert_history

        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]

        if level:
            filtered = [a for a in filtered if a.level == level]

        if start_time:
            filtered = [a for a in filtered if a.timestamp >= start_time]

        if end_time:
            filtered = [a for a in filtered if a.timestamp <= end_time]

        # 按时间倒序
        filtered = sorted(filtered, key=lambda a: a.timestamp, reverse=True)

        return filtered[:limit]


# 全局告警管理器实例
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
        _global_alert_manager.start()
    return _global_alert_manager


__all__ = [
    'AlertLevel',
    'AlertType',
    'Alert',
    'AlertChannel',
    'LogChannel',
    'EmailChannel',
    'WebhookChannel',
    'DingTalkChannel',
    'SlackChannel',
    'AlertManager',
    'get_alert_manager',
]
