"""
实盘风控系统测试
"""

from datetime import datetime, time

import pytest

from trading.risk import (
    ActionType,
    CashLimitRule,
    DailyLossLimitRule,
    DailyVolumeLimitRule,
    OrderRequest,
    PositionLimitRule,
    RiskController,
    RiskLevel,
    RiskManager,
    RiskRuleType,
    SingleOrderLimitRule,
    StockBlacklistRule,
    TradingTimeLimitRule,
)


@pytest.fixture
def sample_account():
    """示例账户"""
    return {
        "total_asset": 1000000,
        "available_cash": 500000,
    }


@pytest.fixture
def sample_context(sample_account):
    """示例上下文"""
    return {
        "account": sample_account,
        "positions": [
            {"symbol": "600000.SH", "quantity": 10000, "market_value": 100000},
            {"symbol": "000001.SZ", "quantity": 20000, "market_value": 200000},
        ],
        "daily_stats": {
            "initial_asset": 1000000,
            "traded_volume": 0,
            "daily_pnl": 0,
        },
    }


@pytest.mark.trading
class TestCashLimitRule:
    """测试资金限制规则"""

    def test_cash_sufficient(self, sample_context):
        """测试资金充足"""
        rule = CashLimitRule(min_available_cash=100000)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=10000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True

    def test_cash_insufficient(self, sample_context):
        """测试资金不足"""
        rule = CashLimitRule(min_available_cash=200000)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=40000,  # 需要40万
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False
        assert "资金不足" in result.message

    def test_sell_no_check(self, sample_context):
        """测试卖出不检查资金"""
        rule = CashLimitRule()

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.SELL,
            quantity=100000,  # 超出可用资金
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True


@pytest.mark.trading
class TestSingleOrderLimitRule:
    """测试单笔订单限制"""

    def test_quantity_limit(self, sample_context):
        """测试数量限制"""
        rule = SingleOrderLimitRule(max_quantity=100000)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=200000,  # 超过限制
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False

    def test_amount_limit(self, sample_context):
        """测试金额限制"""
        rule = SingleOrderLimitRule(max_amount=1000000)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=200000,
            price=10.0,  # 200万
        )

        result = rule.check(order, sample_context)
        assert result.passed is False

    def test_within_limit(self, sample_context):
        """测试在限制内"""
        rule = SingleOrderLimitRule()

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True


@pytest.mark.trading
class TestDailyVolumeLimitRule:
    """测试日交易量限制"""

    def test_volume_limit(self, sample_context):
        """测试交易量限制"""
        # 设置已交易金额
        sample_context["daily_stats"]["traded_volume"] = 45000000

        rule = DailyVolumeLimitRule(max_daily_volume=50000000)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000000,  # 1000万股 = 1000万，超过限制
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False

    def test_within_limit(self, sample_context):
        """测试在限制内"""
        rule = DailyVolumeLimitRule()

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True


@pytest.mark.trading
class TestPositionLimitRule:
    """测试持仓限制"""

    def test_position_count_limit(self, sample_context):
        """测试持仓数量限制"""
        # 模拟已有50个持仓
        sample_context["positions"] = [
            {"symbol": f"{i:06d}.SH", "quantity": 1000, "market_value": 10000} for i in range(50)
        ]

        rule = PositionLimitRule(max_positions=50)

        order = OrderRequest(
            symbol="600001.SH",  # 新股票
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False

    def test_concentration_limit(self, sample_context):
        """测试持仓集中度限制"""
        rule = PositionLimitRule(max_single_position_ratio=0.2)

        # 测试单一持仓比例超限
        order = OrderRequest(
            symbol="600000.SH",  # 已有10万
            action=ActionType.BUY,
            quantity=90000,  # 90万，总价值100万
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False

    def test_sell_no_limit(self, sample_context):
        """测试卖出不受限制"""
        rule = PositionLimitRule(max_positions=1)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.SELL,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True


@pytest.mark.trading
class TestStockBlacklistRule:
    """测试股票黑名单"""

    def test_blacklisted_symbol(self, sample_context):
        """测试黑名单股票"""
        rule = StockBlacklistRule(blacklist=["600000.SH"])

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False
        assert "黑名单" in result.message

    def test_allowed_symbol(self, sample_context):
        """测试非黑名单股票"""
        rule = StockBlacklistRule(blacklist=["600000.SH"])

        order = OrderRequest(
            symbol="000001.SZ",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True

    def test_add_to_blacklist(self):
        """测试添加到黑名单"""
        rule = StockBlacklistRule()

        assert "000001.SZ" not in rule.blacklist
        rule.add_to_blacklist("000001.SZ")
        assert "000001.SZ" in rule.blacklist

    def test_remove_from_blacklist(self):
        """测试从黑名单移除"""
        rule = StockBlacklistRule(blacklist=["000001.SZ"])

        assert "000001.SZ" in rule.blacklist
        rule.remove_from_blacklist("000001.SZ")
        assert "000001.SZ" not in rule.blacklist


@pytest.mark.trading
class TestDailyLossLimitRule:
    """测试日亏损限制"""

    def test_loss_limit(self, sample_context):
        """测试亏损限制"""
        # 设置日亏损5%
        sample_context["daily_stats"]["daily_pnl"] = -60000

        rule = DailyLossLimitRule(max_daily_loss_ratio=0.05)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is False

    def test_within_limit(self, sample_context):
        """测试在限制内"""
        sample_context["daily_stats"]["daily_pnl"] = -20000

        rule = DailyLossLimitRule(max_daily_loss_ratio=0.05)

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        result = rule.check(order, sample_context)
        assert result.passed is True


@pytest.mark.trading
class TestRiskManager:
    """测试风控管理器"""

    def test_add_remove_rule(self):
        """测试添加移除规则"""
        manager = RiskManager()

        rule = CashLimitRule()
        manager.add_rule(rule)

        assert len(manager.rules) == 1
        assert rule in manager.rules

        manager.remove_rule("资金限制")
        assert len(manager.rules) == 0

    def test_check_order(self, sample_context):
        """测试检查订单"""
        manager = RiskManager()
        manager.add_rule(CashLimitRule(min_available_cash=100000))

        order = OrderRequest(
            symbol="600000.SH",
            action=ActionType.BUY,
            quantity=1000,
            price=10.0,
        )

        results = manager.check_order(order, sample_context)
        assert len(results) == 1
        assert results[0].passed is True

    def test_statistics(self, sample_context):
        """测试统计信息"""
        manager = RiskManager()
        manager.add_rule(CashLimitRule(min_available_cash=100000))

        # 执行多次检查
        for i in range(10):
            order = OrderRequest(
                symbol="600000.SH",
                action=ActionType.BUY,
                quantity=1000,
                price=10.0,
            )
            manager.check_order(order, sample_context)

        stats = manager.get_statistics()
        assert stats["total_checks"] == 10
        assert stats["total_rejects"] == 0
        assert stats["active_rules"] == 1


@pytest.mark.trading
class TestRiskController:
    """测试风控控制器"""

    def test_initialization(self):
        """测试初始化"""
        controller = RiskController()

        assert controller.manager is not None
        assert len(controller.manager.rules) > 0

    def test_validate_order(self, sample_context):
        """测试验证订单"""
        controller = RiskController()

        allowed, rejects = controller.validate_order(
            symbol="600000.SH",
            action="buy",
            quantity=1000,
            price=10.0,
            context=sample_context,
        )

        assert isinstance(allowed, bool)
        assert isinstance(rejects, list)

    def test_add_custom_rule(self):
        """测试添加自定义规则"""
        from trading.risk import BaseRiskRule, RiskCheckResult

        class CustomRule(BaseRiskRule):
            def __init__(self):
                super().__init__(
                    name="自定义规则",
                    rule_type=RiskRuleType.CASH_LIMIT,
                )

            def check(self, order, context):
                return RiskCheckResult(
                    passed=True,
                    rule_type=self.rule_type,
                    rule_name=self.name,
                    message="测试通过",
                )

        controller = RiskController()
        initial_count = len(controller.manager.rules)

        controller.add_custom_rule(CustomRule())

        assert len(controller.manager.rules) == initial_count + 1

    def test_enable_disable_rule(self):
        """测试启用禁用规则"""
        controller = RiskController()

        # 禁用规则
        controller.disable_rule("资金限制")

        # 验证已被禁用
        for rule in controller.manager.rules:
            if rule.name == "资金限制":
                assert rule.enabled is False

        # 重新启用
        controller.enable_rule("资金限制")

        for rule in controller.manager.rules:
            if rule.name == "资金限制":
                assert rule.enabled is True
