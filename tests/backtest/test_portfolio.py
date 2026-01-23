"""
测试投资组合管理
"""

from datetime import datetime

import numpy as np
import pytest

from backtest.engine.event_engine import FillEvent
from backtest.engine.portfolio import (
    Account,
    Portfolio,
    Position,
    PositionSide,
)


@pytest.mark.unit
class TestPosition:
    """测试持仓"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(
            symbol="000001.SZ", quantity=1000, avg_price=100.0, side=PositionSide.LONG
        )

        assert position.symbol == "000001.SZ"
        assert position.quantity == 1000
        assert position.avg_price == 100.0
        assert position.side == PositionSide.LONG

    def test_position_default_values(self):
        """测试持仓默认值"""
        position = Position(symbol="000001.SZ")

        assert position.quantity == 0
        assert position.avg_price == 0.0
        assert position.side == PositionSide.LONG
        assert position.available_qty == 0
        assert position.today_bought == 0


@pytest.mark.unit
class TestAccount:
    """测试账户"""

    def test_account_creation(self):
        """测试账户创建"""
        account = Account(initial_cash=1000000.0, cash=1000000.0, total_value=1000000.0)

        assert account.initial_cash == 1000000.0
        assert account.cash == 1000000.0
        assert account.total_value == 1000000.0
        assert account.total_return == 0.0
        assert len(account.positions) == 0

    def test_add_to_equity_curve(self):
        """测试添加净值记录"""
        account = Account(initial_cash=1000000.0)
        dt = datetime.now()

        account.add_to_equity_curve(dt, 1100000.0)

        assert len(account.equity_curve) == 1
        assert account.equity_curve[0]["total_value"] == 1100000.0
        assert account.equity_curve[0]["return"] == 0.1

    def test_update_position_buy(self):
        """测试更新持仓（买入）"""
        account = Account(initial_cash=1000000.0)

        # 买入
        account.update_position("000001.SZ", 1000, 100.0, "buy")

        assert "000001.SZ" in account.positions
        position = account.positions["000001.SZ"]
        assert position.quantity == 1000
        assert position.avg_price == 100.0

    def test_update_position_buy_multiple(self):
        """测试多次买入"""
        account = Account(initial_cash=1000000.0)

        # 第一次买入
        account.update_position("000001.SZ", 1000, 100.0, "buy")
        # 第二次买入
        account.update_position("000001.SZ", 500, 110.0, "buy")

        position = account.positions["000001.SZ"]
        assert position.quantity == 1500
        # 平均成本 = (1000*100 + 500*110) / 1500 = 103.33
        assert abs(position.avg_price - 103.33) < 0.01

    def test_update_position_sell(self):
        """测试更新持仓（卖出）"""
        account = Account(initial_cash=1000000.0)

        # 先买入
        account.update_position("000001.SZ", 1000, 100.0, "buy")
        # 卖出一半
        account.update_position("000001.SZ", 500, 110.0, "sell")

        position = account.positions["000001.SZ"]
        assert position.quantity == 500

    def test_update_position_sell_all(self):
        """测试全部卖出"""
        account = Account(initial_cash=1000000.0)

        # 买入
        account.update_position("000001.SZ", 1000, 100.0, "buy")
        # 全部卖出
        account.update_position("000001.SZ", 1000, 110.0, "sell")

        assert "000001.SZ" not in account.positions

    def test_update_total_value(self):
        """测试更新总资产"""
        account = Account(initial_cash=1000000.0)
        account.update_position("000001.SZ", 1000, 100.0, "buy")

        # 手动扣除现金（update_position只更新持仓，不更新现金）
        # 在实际使用中，现金会在update_fill中扣除
        account.cash -= 1000 * 100.0

        # 更新价格
        prices = {"000001.SZ": 110.0}
        account.update_total_value(prices)

        # 现金 = 1000000 - 1000*100 = 900000
        # 持仓市值 = 1000 * 110 = 110000
        # 总资产 = 900000 + 110000 = 1010000
        assert account.total_value == 1010000.0
        assert account.total_return == 0.01


@pytest.mark.unit
class TestPortfolio:
    """测试投资组合"""

    def test_portfolio_creation(self):
        """测试投资组合创建"""
        portfolio = Portfolio(
            initial_cash=1000000.0,
            commission_rate=0.0003,
        )

        assert portfolio.account.initial_cash == 1000000.0
        assert portfolio.account.cash == 1000000.0
        assert portfolio.commission_rate == 0.0003

    def test_calculate_commission(self):
        """测试佣金计算"""
        portfolio = Portfolio(
            initial_cash=1000000.0,
            commission_rate=0.0003,
        )

        # 1000股 @ 100元 = 100000元
        # 佣金 = 100000 * 0.0003 = 30元
        commission = portfolio._calculate_commission(1000, 100.0)
        assert np.isclose(commission, 30.0)

    def test_calculate_commission_minimum(self):
        """测试最低佣金"""
        portfolio = Portfolio(
            initial_cash=1000000.0,
            commission_rate=0.0003,
            min_commission=5.0,
        )

        # 100股 @ 10元 = 1000元
        # 佣金 = 1000 * 0.0003 = 0.3元，但最低5元
        commission = portfolio._calculate_commission(100, 10.0)
        assert commission == 5.0

    def test_update_current_price(self):
        """测试更新当前价格"""
        portfolio = Portfolio(initial_cash=1000000.0)
        portfolio.update_current_price("000001.SZ", 100.0)

        assert portfolio.get_position("000001.SZ")["current_price"] == 100.0

    def test_update_fill_buy(self):
        """测试处理买入成交"""
        portfolio = Portfolio(
            initial_cash=1000000.0,
            commission_rate=0.0003,
        )

        fill_event = FillEvent(
            datetime=datetime.now(),
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=100.0,
            commission=30.0,
        )

        portfolio.update_fill(fill_event)

        # 检查现金减少
        # 1000 * 100 + 30 = 100030
        assert abs(portfolio.account.cash - (1000000 - 100030)) < 1

        # 检查持仓
        position = portfolio.get_position("000001.SZ")
        assert position["quantity"] == 1000
        assert position["avg_price"] == 100.0

    def test_update_fill_sell(self):
        """测试处理卖出成交"""
        portfolio = Portfolio(
            initial_cash=1000000.0,
            commission_rate=0.0003,
        )

        # 先买入
        portfolio.update_fill(
            FillEvent(
                datetime=datetime.now(),
                order_id="order_1",
                symbol="000001.SZ",
                side="buy",
                quantity=1000,
                price=100.0,
                commission=30.0,
            )
        )

        # 更新价格
        portfolio.update_current_price("000001.SZ", 110.0)

        # 卖出
        portfolio.update_fill(
            FillEvent(
                datetime=datetime.now(),
                order_id="order_2",
                symbol="000001.SZ",
                side="sell",
                quantity=1000,
                price=110.0,
                commission=33.0,
            )
        )

        # 检查持仓
        position = portfolio.get_position("000001.SZ")
        assert position["quantity"] == 0

    def test_get_position(self):
        """测试获取持仓"""
        portfolio = Portfolio(initial_cash=1000000.0)

        # 无持仓时
        position = portfolio.get_position("000001.SZ")
        assert position["quantity"] == 0
        assert position["symbol"] == "000001.SZ"

    def test_get_all_positions(self):
        """测试获取所有持仓"""
        portfolio = Portfolio(initial_cash=1000000.0)

        # 买入两个股票
        portfolio.update_fill(
            FillEvent(
                datetime=datetime.now(),
                order_id="order_1",
                symbol="000001.SZ",
                side="buy",
                quantity=1000,
                price=100.0,
                commission=30.0,
            )
        )

        portfolio.update_fill(
            FillEvent(
                datetime=datetime.now(),
                order_id="order_2",
                symbol="600000.SH",
                side="buy",
                quantity=500,
                price=50.0,
                commission=7.5,
            )
        )

        positions = portfolio.get_all_positions()
        assert len(positions) == 2

    def test_get_account_info(self):
        """测试获取账户信息"""
        portfolio = Portfolio(initial_cash=1000000.0)

        info = portfolio.get_account_info()

        assert info["initial_cash"] == 1000000.0
        assert info["cash"] == 1000000.0
        assert info["total_value"] == 1000000.0
        assert info["total_return"] == 0.0
        assert info["num_positions"] == 0

    def test_equity_curve(self):
        """测试净值曲线"""
        portfolio = Portfolio(initial_cash=1000000.0)

        # 添加净值记录
        dt = datetime.now()
        portfolio.add_to_equity_curve(dt)
        portfolio.update_total_value()

        equity_curve = portfolio.get_equity_curve()
        assert len(equity_curve) == 1
        assert "total_value" in equity_curve.columns
        assert "return" in equity_curve.columns

    def test_reset(self):
        """测试重置投资组合"""
        portfolio = Portfolio(initial_cash=1000000.0)

        # 进行一些交易
        portfolio.update_fill(
            FillEvent(
                datetime=datetime.now(),
                order_id="order_1",
                symbol="000001.SZ",
                side="buy",
                quantity=1000,
                price=100.0,
                commission=30.0,
            )
        )

        # 重置
        portfolio.reset()

        # 检查状态已重置
        assert portfolio.account.cash == 1000000.0
        assert len(portfolio.account.positions) == 0
        assert portfolio._order_id == 0
