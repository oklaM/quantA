"""
实盘风控系统示例
展示如何使用风控系统进行实盘交易风险控制
"""

from trading.risk import (
    RiskController,
    OrderRequest,
    ActionType,
    RiskRuleType,
    BaseRiskRule,
    RiskCheckResult,
    RiskLevel,
)
from monitoring import AlertManager, Alert, AlertType, AlertLevel
from utils.logging import get_logger

logger = get_logger(__name__)


def example_basic_risk_control():
    """示例1：基础风控"""
    print("\n" + "="*70)
    print("示例1：基础风控控制")
    print("="*70)

    # 创建风控控制器
    controller = RiskController({
        'min_available_cash': 100000,  # 最少保留10万
        'max_single_order_amount': 1000000,  # 单笔最多100万
        'max_daily_volume': 50000000,  # 日交易量最多5000万
        'max_positions': 30,  # 最多持仓30只
        'max_single_position_ratio': 0.25,  # 单一持仓最多25%
        'max_daily_loss_ratio': 0.03,  # 日亏损超过3%停止交易
    })

    # 模拟账户和持仓
    account = {
        'total_asset': 5000000,  # 总资产500万
        'available_cash': 2000000,  # 可用资金200万
    }

    positions = [
        {'symbol': '600000.SH', 'quantity': 10000, 'market_value': 500000},
        {'symbol': '000001.SZ', 'quantity': 20000, 'market_value': 800000},
    ]

    daily_stats = {
        'initial_asset': 5000000,
        'traded_volume': 0,
        'daily_pnl': 0,
    }

    context = {
        'account': account,
        'positions': positions,
        'daily_stats': daily_stats,
    }

    # 测试订单1：正常订单
    print("\n测试订单1: 正常买入")
    allowed, rejects = controller.validate_order(
        symbol='600036.SH',
        action='buy',
        quantity=1000,
        price=10.50,
        context=context,
    )

    if allowed:
        print("  ✓ 订单通过风控检查")
    else:
        print(f"  ✗ 订单被拒绝: {rejects}")

    # 测试订单2：超出单笔限制
    print("\n测试订单2: 超出单笔金额限制")
    allowed, rejects = controller.validate_order(
        symbol='600036.SH',
        action='buy',
        quantity=200000,  # 2000万
        price=100.0,
        context=context,
    )

    if allowed:
        print("  ✓ 订单通过风控检查")
    else:
        print(f"  ✗ 订单被拒绝: {rejects[0]}")

    # 测试订单3：资金不足
    print("\n测试订单3: 资金不足")
    allowed, rejects = controller.validate_order(
        symbol='600036.SH',
        action='buy',
        quantity=300000,
        price=10.0,
        context=context,
    )

    if allowed:
        print("  ✓ 订单通过风控检查")
    else:
        print(f"  ✗ 订单被拒绝: {rejects[0]}")

    # 获取风控统计
    stats = controller.get_statistics()
    print(f"\n风控统计:")
    print(f"  总检查次数: {stats['total_checks']}")
    print(f"  总拒绝次数: {stats['total_rejects']}")
    print(f"  拒绝率: {stats['reject_ratio']:.2%}")
    print(f"  活跃规则数: {stats['active_rules']}")


def example_stock_blacklist():
    """示例2：股票黑名单"""
    print("\n" + "="*70)
    print("示例2：股票黑名单管理")
    print("="*70)

    controller = RiskController({
        'stock_blacklist': ['ST.*', '.*ST'],  # ST股票黑名单
    })

    context = {
        'account': {'total_asset': 1000000, 'available_cash': 500000},
        'positions': [],
        'daily_stats': {},
    }

    # 测试ST股票
    print("\n测试ST股票交易:")
    allowed, rejects = controller.validate_order(
        symbol='ST康美',
        action='buy',
        quantity=1000,
        price=5.0,
        context=context,
    )

    if allowed:
        print("  ✓ 订单通过")
    else:
        print(f"  ✗ 订单被拒绝: {rejects[0]}")


def example_custom_risk_rule():
    """示例3：自定义风控规则"""
    print("\n" + "="*70)
    print("示例3：自定义风控规则")
    print("="*70)

    from trading.risk import BaseRiskRule, RiskCheckResult, RiskRuleType

    class CustomVolatilityRule(BaseRiskRule):
        """自定义波动率限制规则"""

        def __init__(self, max_volatility: float = 0.05):
            super().__init__(
                name="波动率限制",
                rule_type=RiskRuleType.PRICE_LIMIT,
                risk_level=RiskLevel.MEDIUM,
            )
            self.max_volatility = max_volatility

        def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
            """检查股票波动率"""
            market_data = context.get('market_data', {})

            # 获取历史波动率（简化处理）
            # 实际应用中应从context中获取真实的波动率数据
            volatility = market_data.get(order.symbol, {}).get('volatility', 0)

            if volatility > self.max_volatility:
                return RiskCheckResult(
                    passed=False,
                    rule_type=self.rule_type,
                    rule_name=self.name,
                    message=f"股票波动率过高: {volatility:.2%} > {self.max_volatility:.2%}",
                    risk_level=RiskLevel.MEDIUM,
                    metadata={
                        'symbol': order.symbol,
                        'volatility': volatility,
                        'limit': self.max_volatility,
                    },
                )

            return RiskCheckResult(
                passed=True,
                rule_type=self.rule_type,
                rule_name=self.name,
                message="波动率检查通过",
            )

    # 创建风控控制器
    controller = RiskController()

    # 添加自定义规则
    controller.add_custom_rule(CustomVolatilityRule(max_volatility=0.08))

    # 测试
    context = {
        'account': {'total_asset': 1000000, 'available_cash': 500000},
        'positions': [],
        'daily_stats': {},
        'market_data': {
            '600000.SH': {'volatility': 0.06},  # 6%波动率
            '000001.SZ': {'volatility': 0.10},  # 10%波动率
        },
    }

    print("\n测试低波动率股票:")
    allowed, rejects = controller.validate_order(
        symbol='600000.SH',
        action='buy',
        quantity=1000,
        price=10.0,
        context=context,
    )
    print(f"  {'✓ 通过' if allowed else '✗ 拒绝'}: {rejects[0] if not allowed else ''}")

    print("\n测试高波动率股票:")
    allowed, rejects = controller.validate_order(
        symbol='000001.SZ',
        action='buy',
        quantity=1000,
        price=10.0,
        context=context,
    )
    print(f"  {'✓ 通过' if allowed else '✗ 拒绝'}: {rejects[0] if not allowed else ''}")


def example_risk_with_alerting():
    """示例4：风控告警集成"""
    print("\n" + "="*70)
    print("示例4：风控告警集成")
    print("="*70)

    # 创建告警管理器
    alert_manager = AlertManager()
    alert_manager.start()

    controller = RiskController({
        'max_daily_loss_ratio': 0.02,  # 日亏损2%停止
    })

    # 模拟亏损场景
    print("\n模拟日亏损超限场景:")

    account = {
        'total_asset': 1000000,
        'available_cash': 980000,
    }

    daily_stats = {
        'initial_asset': 1000000,
        'daily_pnl': -30000,  # 亏损3万
        'traded_volume': 500000,
    }

    context = {
        'account': account,
        'positions': [],
        'daily_stats': daily_stats,
    }

    # 尝试下单
    allowed, rejects = controller.validate_order(
        symbol='600000.SH',
        action='buy',
        quantity=1000,
        price=10.0,
        context=context,
    )

    if not allowed:
        print(f"  订单被风控拒绝")

        # 发送告警
        alert = Alert(
            alert_id="risk_alert_001",
            alert_type=AlertType.LOSS_LIMIT,
            level=AlertLevel.CRITICAL,
            title="⚠️ 风控警告",
            message=f"日亏损已达{daily_stats['daily_pnl']/daily_stats['initial_asset']:.2%}，超过限制",
            metadata={
                'daily_pnl': daily_stats['daily_pnl'],
                'daily_return': daily_stats['daily_pnl'] / daily_stats['initial_asset'],
            },
        )
        alert_manager.send_alert(alert)

    alert_manager.stop()


def example_dynamic_risk_adjustment():
    """示例5：动态风控调整"""
    print("\n" + "="*70)
    print("示例5：动态风控调整")
    print("="*70)

    controller = RiskController({
        'max_daily_loss_ratio': 0.05,
        'max_single_order_amount': 1000000,
    })

    print("""
动态风控调整策略：

1. 根据账户状态调整风控参数

    def adjust_risk_by_performance(controller, daily_return):
        if daily_return < -0.02:  # 亏损2%
            # 收紧风控：降低单笔金额限制
            controller.config['max_single_order_amount'] = 500000
            logger.info("风控已收紧：单笔限额降至50万")

        elif daily_return > 0.02:  # 盈利2%
            # 放宽风控：提高单笔金额限制
            controller.config['max_single_order_amount'] = 2000000
            logger.info("风控已放宽：单笔限额升至200万")

2. 根据市场波动率调整

    def adjust_risk_by_volatility(controller, market_vol):
        if market_vol > 0.03:  # 高波动
            controller.disable_rule("持仓限制")  # 允许更多持仓分散
            logger.info("高波动市场：启用分散策略")

        else:
            controller.enable_rule("持仓限制")

3. 根据回撤调整

    def adjust_risk_by_drawdown(controller, drawdown):
        if drawdown < -0.10:  # 回撤10%
            # 禁止开新仓
            for rule in controller.manager.rules:
                if rule.rule_type in [RiskRuleType.POSITION_LIMIT]:
                    rule.enabled = False

            logger.warning(f"回撤{drawdown:.2%}：禁止开新仓")

4. 分级风控

    LEVELS = {
        'conservative': {
            'max_single_order_ratio': 0.1,
            'max_positions': 20,
            'max_daily_loss_ratio': 0.02,
        },
        'moderate': {
            'max_single_order_ratio': 0.2,
            'max_positions': 30,
            'max_daily_loss_ratio': 0.05,
        },
        'aggressive': {
            'max_single_order_ratio': 0.3,
            'max_positions': 50,
            'max_daily_loss_ratio': 0.08,
        },
    }

    def set_risk_level(controller, level):
        config = LEVELS[level]
        # 更新风控配置
        for key, value in config.items():
            controller.config[key] = value
        logger.info(f"风控级别已设置为: {level}")
""")


def example_risk_monitoring_dashboard():
    """示例6：风控监控看板"""
    print("\n" + "="*70)
    print("示例6：风控监控看板")
    print("="*70)

    controller = RiskController()

    # 模拟多次检查
    context = {
        'account': {'total_asset': 1000000, 'available_cash': 500000},
        'positions': [],
        'daily_stats': {},
    }

    # 执行多次检查
    for i in range(10):
        controller.validate_order(
            symbol='600000.SH',
            action='buy',
            quantity=1000,
            price=10.0,
            context=context,
        )

    # 获取统计信息
    stats = controller.get_statistics()

    print("\n风控监控看板:")
    print("="*70)
    print(f"总检查次数: {stats['total_checks']}")
    print(f"总拒绝次数: {stats['total_rejects']}")
    print(f"通过率: {(1 - stats['reject_ratio'])*100:.1f}%")
    print(f"活跃规则: {stats['active_rules']}/{stats['total_rules']}")

    if stats['rejects_by_rule']:
        print("\n按规则分类:")
        for rule_name, count in stats['rejects_by_rule'].items():
            print(f"  {rule_name}: {count}次")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 实盘风控系统示例")
    print("="*70)

    try:
        # 示例1：基础风控
        example_basic_risk_control()

        # 示例2：股票黑名单
        # example_stock_blacklist()

        # 示例3：自定义规则
        # example_custom_risk_rule()

        # 示例4：风控告警
        # example_risk_with_alerting()

        # 示例5：动态调整
        # example_dynamic_risk_adjustment()

        # 示例6：监控看板
        example_risk_monitoring_dashboard()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
