"""
监控告警系统示例
展示如何使用监控告警功能
"""

import time
import random
from datetime import datetime, timedelta

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from monitoring import (
    AlertManager,
    Alert,
    AlertLevel,
    AlertType,
    PerformanceMonitor,
    get_performance_monitor,
    LogChannel,
    DingTalkChannel,
    SlackChannel,
)
from utils.logging import get_logger

logger = get_logger(__name__)


def example_basic_alerting():
    """示例1：基本告警功能"""
    print("\n" + "="*70)
    print("示例1：基本告警功能")
    print("="*70)

    # 创建告警管理器
    alert_manager = AlertManager()

    # 创建并发送告警
    alert = Alert(
        alert_id="alert_001",
        alert_type=AlertType.STRATEGY_ERROR,
        level=AlertLevel.ERROR,
        title="策略运行错误",
        message="策略Strategy_A在执行过程中遇到错误",
        metadata={
            'strategy': 'Strategy_A',
            'error_code': 'E001',
            'timestamp': datetime.now().isoformat(),
        },
    )

    alert_manager.send_alert(alert)

    # 等待异步处理
    time.sleep(1)

    # 查看告警历史
    history = alert_manager.get_alert_history(limit=5)
    print(f"\n告警历史（共{len(history)}条）:")
    for h in history:
        print(f"  [{h.level.value.upper()}] {h.title}: {h.message}")


def example_email_alerting():
    """示例2：邮件告警"""
    print("\n" + "="*70)
    print("示例2：邮件告警")
    print("="*70)

    print("""
配置邮件告警：

from monitoring import EmailChannel, AlertManager

# 创建邮件渠道
email_channel = EmailChannel(
    smtp_host='smtp.gmail.com',
    smtp_port=587,
    username='your_email@gmail.com',
    password='your_app_password',
    from_addr='your_email@gmail.com',
    to_addrs=['recipient@example.com'],
)

# 添加到告警管理器
alert_manager = AlertManager()
alert_manager.add_channel(email_channel)

# 发送测试告警
alert = Alert(
    alert_id="test_001",
    alert_type=AlertType.SYSTEM_ERROR,
    level=AlertLevel.WARNING,
    title="测试邮件告警",
    message="这是一封测试邮件",
)
alert_manager.send_alert(alert)

注意：
- 需要在邮箱设置中启用"不够安全的应用访问"或使用应用专用密码
- Gmail需要使用应用专用密码，不能使用账户密码
- 建议使用SMTP over TLS (port 587)
""")


def example_dingtalk_alerting():
    """示例3：钉钉告警"""
    print("\n" + "="*70)
    print("示例3：钉钉告警")
    print("="*70)

    if not REQUESTS_AVAILABLE:
        print("\n无法运行：需要安装 requests")
        print("安装命令: pip install requests")
        return

    print("""
配置钉钉告警：

1. 在钉钉群中添加自定义机器人
   - 群设置 -> 智能群助手 -> 添加机器人 -> 自定义
   - Webhook地址：复制生成的Webhook URL

2. 配置告警渠道

from monitoring import DingTalkChannel, AlertManager

# 创建钉钉渠道
dingtalk_channel = DingTalkChannel(
    webhook_url='https://oapi.dingtalk.com/robot/send?access_token=YOUR_TOKEN'
)

# 添加到告警管理器
alert_manager = AlertManager()
alert_manager.add_channel(dingtalk_channel)

# 发送告警
alert = Alert(
    alert_id="dt_001",
    alert_type=AlertType.HIGH_DRAWDOWN,
    level=AlertLevel.CRITICAL,
    title="⚠️ 高回撤告警",
    message="策略ABC当前回撤超过15%，请注意风险",
    metadata={
        'strategy': 'ABC',
        'current_drawdown': -0.156,
        'threshold': -0.15,
    },
)
alert_manager.send_alert(alert)

3. 安全设置（推荐）
   - 加签验证
   - 关键词验证（如"告警"、"警告"等）
   - IP地址限制
""")


def example_slack_alerting():
    """示例4：Slack告警"""
    print("\n" + "="*70)
    print("示例4：Slack告警")
    print("="*70)

    if not REQUESTS_AVAILABLE:
        print("\n无法运行：需要安装 requests")
        print("安装命令: pip install requests")
        return

    print("""
配置Slack告警：

1. 在Slack中创建Incoming Webhook
   - 访问 https://api.slack.com/apps
   - 创建新应用 -> Incoming Webhooks
   - 启用Webhook并复制Webhook URL

2. 配置告警渠道

from monitoring import SlackChannel, AlertManager

# 创建Slack渠道
slack_channel = SlackChannel(
    webhook_url='https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
    channel='#alerts',  # 可选：指定频道
)

# 添加到告警管理器
alert_manager = AlertManager()
alert_manager.add_channel(slack_channel)

# 发送告警
alert = Alert(
    alert_id="slack_001",
    alert_type=AlertType.STRATEGY_PERFORMANCE,
    level=AlertLevel.INFO,
    title="策略表现更新",
    message="策略XYZ今日收益率为2.3%，超过目标",
    metadata={
        'strategy': 'XYZ',
        'daily_return': 0.023,
        'target': 0.02,
    },
)
alert_manager.send_alert(alert)
""")


def example_performance_monitoring():
    """示例5：性能监控"""
    print("\n" + "="*70)
    print("示例5：性能监控")
    print("="*70)

    # 获取性能监控器
    monitor = get_performance_monitor()

    # 模拟记录一些数据
    print("\n记录指标数据...")

    for i in range(50):
        # 记录收益率
        return_value = random.gauss(0.001, 0.02)
        monitor.record_return('strategy_a', return_value)

        # 记录回撤
        drawdown = max(0, -random.gauss(0.02, 0.01))
        monitor.record_drawdown('strategy_a', drawdown)

        # 记录系统指标
        monitor.record_system_cpu(random.uniform(20, 80))
        monitor.record_system_memory(random.uniform(40, 90))

        time.sleep(0.01)  # 模拟时间间隔

    # 获取指标摘要
    print("\n指标摘要:")
    summaries = monitor.get_all_summaries()

    for metric_name, summary in summaries.items():
        print(f"\n{metric_name}:")
        print(f"  最新值: {summary['latest']:.4f}")
        print(f"  均值: {summary['mean']:.4f}")
        print(f"  标准差: {summary['std']:.4f}")
        print(f"  数据点: {summary['count']}")


def example_anomaly_detection():
    """示例6：异常检测"""
    print("\n" + "="*70)
    print("示例6：异常检测")
    print("="*70)

    monitor = get_performance_monitor()

    # 添加异常告警回调
    def anomaly_callback(anomaly):
        print(f"\n检测到异常！")
        print(f"  指标: {anomaly['metric']}")
        print(f"  类型: {anomaly['type']}")
        print(f"  当前值: {anomaly['value']:.4f}")
        print(f"  时间: {anomaly['timestamp']}")

        # 发送告警
        alert_manager = get_alert_manager()
        alert = Alert(
            alert_id=f"anomaly_{int(time.time())}",
            alert_type=AlertType.STRATEGY_PERFORMANCE,
            level=AlertLevel.WARNING,
            title=f"检测到指标异常: {anomaly['metric']}",
            message=f"指标{anomaly['metric']}在{anomaly['type']}检测中发现异常",
            metadata=anomaly,
        )
        alert_manager.send_alert(alert)

    monitor.add_alert_callback(anomaly_callback)

    # 记录正常数据
    print("\n记录正常数据...")
    for i in range(30):
        monitor.record_return('strategy_b', random.gauss(0.001, 0.01))
        time.sleep(0.01)

    # 记录异常数据（触发告警）
    print("\n记录异常数据...")
    monitor.record_return('strategy_b', 0.5)  # 异常大的收益率
    time.sleep(0.5)  # 等待异步处理


def example_custom_alert_rules():
    """示例7：自定义告警规则"""
    print("\n" + "="*70)
    print("示例7：自定义告警规则")
    print("="*70)

    print("""
自定义告警规则：

from monitoring import AlertManager, Alert, AlertLevel, AlertType

alert_manager = get_alert_manager()

# 规则1: 回撤过高
def check_high_drawdown(context):
    current_drawdown = context.get('drawdown', 0)
    threshold = -0.10  # 10%

    if current_drawdown < threshold:
        return Alert(
            alert_id=f"drawdown_{int(time.time())}",
            alert_type=AlertType.HIGH_DRAWDOWN,
            level=AlertLevel.CRITICAL,
            title="⚠️ 高回撤告警",
            message=f"当前回撤{current_drawdown:.2%}超过阈值{threshold:.2%}",
            metadata={
                'current_drawdown': current_drawdown,
                'threshold': threshold,
            },
        )
    return None

alert_manager.add_rule('high_drawdown', check_high_drawdown)

# 规则2: 交易延迟过高
def check_order_latency(context):
    latency_ms = context.get('order_latency', 0)
    threshold_ms = 1000  # 1秒

    if latency_ms > threshold_ms:
        return Alert(
            alert_id=f"latency_{int(time.time())}",
            alert_type=AlertType.EXECUTION_DELAY,
            level=AlertLevel.WARNING,
            title="订单延迟告警",
            message=f"订单延迟{latency_ms:.0f}ms超过阈值{threshold_ms}ms",
            metadata={
                'latency_ms': latency_ms,
                'threshold_ms': threshold_ms,
            },
        )
    return None

alert_manager.add_rule('high_latency', check_order_latency)

# 定期检查规则
def monitoring_loop():
    while True:
        # 获取当前上下文数据
        context = {
            'drawdown': get_current_drawdown(),
            'order_latency': get_last_order_latency(),
            # ... 其他指标
        }

        # 检查所有规则
        alert_manager.check_rules(context)

        time.sleep(60)  # 每分钟检查一次
""")


def example_webhook_alerting():
    """示例8：Webhook告警"""
    print("\n" + "="*70)
    print("示例8：Webhook告警")
    print("="*70)

    print("""
配置Webhook告警：

from monitoring import WebhookChannel, AlertManager

# 自定义Webhook端点
webhook_channel = WebhookChannel(
    url='https://your-server.com/api/alerts',
    method='POST',  # 或 'PUT', 'PATCH'
)

alert_manager = AlertManager()
alert_manager.add_channel(webhook_channel)

# 发送告警（会POST到指定URL）
alert = Alert(
    alert_id="webhook_001",
    alert_type=AlertType.SYSTEM_ERROR,
    level=AlertLevel.ERROR,
    title="系统错误",
    message="检测到系统异常",
    metadata={'error': 'Connection timeout'},
)
alert_manager.send_alert(alert)

Webhook请求格式：
{
    "alert_id": "webhook_001",
    "type": "system_error",
    "level": "error",
    "title": "系统错误",
    "message": "检测到系统异常",
    "timestamp": "2024-01-01T12:00:00",
    "metadata": {"error": "Connection timeout"}
}
""")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 监控告警系统示例")
    print("="*70)

    try:
        # 启动全局告警管理器
        alert_manager = get_alert_manager()
        alert_manager.start()

        # 示例1：基本告警
        example_basic_alerting()

        # 示例2-4：配置说明
        example_email_alerting()
        example_dingtalk_alerting()
        example_slack_alerting()

        # 示例5-6：性能监控
        example_performance_monitoring()
        example_anomaly_detection()

        # 示例7-8：概念说明
        example_custom_alert_rules()
        example_webhook_alerting()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

        # 停止告警管理器
        alert_manager.stop()

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
