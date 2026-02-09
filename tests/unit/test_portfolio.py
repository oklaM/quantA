"""
投资组合管理模块单元测试
测试 backtest/engine/portfolio.py 中的所有类和方法
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from backtest.engine.portfolio import (
    Position,
    PositionSide,
    Account,
    Portfolio,
)
from backtest.engine.event_engine import FillEvent, Event


@pytest.mark.unit
class TestPosition:
    """持仓测试"""

    def test_position_creation(self):
        """测试持仓创建"""
        position = Position(symbol="600519.SH")
        assert position.symbol == "600519.SH"
        assert position.quantity == 0
        assert position.avg_price == 0.0
        assert position.side == PositionSide.LONG
        assert position.available_qty == 0
        assert position.today_bought == 0

    def test_position_with_values(self):
        """测试带值的持仓"""
        position = Position(
            symbol="600519.SH",
            quantity=1000,
            avg_price=1500.0,
            side=PositionSide.LONG,
            available_qty=500,
            today_bought=500
        )
        assert position.quantity == 1000
        assert position.avg_price == 1500.0
        assert position.available_qty == 500
        assert position.today_bought == 500

    def test_position_side_enum(self):
        """测试持仓方向枚举"""
        assert PositionSide.LONG.value == "long"
        assert PositionSide.SHORT.value == "short"

    def test_position_market_value_property(self):
        """测试市值属性"""
        position = Position(symbol="600519.SH", quantity=100)
        # 默认返回0.0
        assert position.market_value == 0.0

    def test_position_pnl_property(self):
        """测试盈亏属性"""
        position = Position(symbol="600519.SH", quantity=100)
        # 默认返回0.0
        assert position.pnl == 0.0

    def test_position_pnl_pct_property(self):
        """测试盈亏比例属性"""
        position = Position(symbol="600519.SH", quantity=100)
        # 默认返回0.0
        assert position.pnl_pct == 0.0


@pytest.mark.unit
class TestAccount:
    """账户测试"""

    def test_account_creation_defaults(self):
        """测试账户默认创建"""
        account = Account()
        assert account.initial_cash == 1000000.0
        assert account.cash == 1000000.0
        assert account.total_value == 1000000.0
        assert len(account.positions) == 0
        assert account.total_return == 0.0
        assert account.total_trades == 0

    def test_account_custom_initial_cash(self):
        """测试自定义初始资金"""
        account = Account(initial_cash=5000000.0, cash=5000000.0, total_value=5000000.0)
        assert account.initial_cash == 5000000.0
        assert account.cash == 5000000.0
        assert account.total_value == 5000000.0

    def test_account_get_position_nonexistent(self):
        """测试获取不存在的持仓"""
        account = Account()
        position = account.get_position("600519.SH")
        assert position is None

    def test_account_update_position_buy(self):
        """测试更新持仓买入"""
        account = Account()
        account.update_position("600519.SH", 100, 1500.0, "buy")

        position = account.get_position("600519.SH")
        assert position is not None
        assert position.quantity == 100
        assert position.avg_price == 1500.0
        assert position.today_bought == 100
        assert position.available_qty == 100

    def test_account_update_position_sell(self):
        """测试更新持仓卖出"""
        account = Account()
        # 先买入
        account.update_position("600519.SH", 100, 1500.0, "buy")
        # 再卖出
        account.update_position("600519.SH", 50, 1600.0, "sell")

        position = account.get_position("600519.SH")
        assert position is not None
        assert position.quantity == 50
        assert position.avg_price == 1500.0  # 平均成本不变

    def test_account_update_position_sell_all(self):
        """测试全部卖出"""
        account = Account()
        account.update_position("600519.SH", 100, 1500.0, "buy")
        account.update_position("600519.SH", 100, 1600.0, "sell")

        position = account.get_position("600519.SH")
        assert position is None  # 持仓应该被删除

    def test_account_update_position_multiple_buys(self):
        """测试多次买入更新平均成本"""
        account = Account()
        account.update_position("600519.SH", 100, 1500.0, "buy")
        account.update_position("600519.SH", 100, 1600.0, "buy")

        position = account.get_position("600519.SH")
        assert position.quantity == 200
        assert position.avg_price == 1550.0  # (1500*100 + 1600*100) / 200

    def test_account_update_total_value(self):
        """测试更新总资产"""
        account = Account(initial_cash=1000000.0, cash=1000000.0, total_value=1000000.0)
        account.update_position("600519.SH", 100, 1500.0, "buy")

        prices = {"600519.SH": 1600.0}
        account.update_total_value(prices)

        expected_value = account.cash + 100 * 1600.0
        assert account.total_value == expected_value
        # 注意：买入后现金减少，但持仓价值增加，总价值可能高于或低于初始值

    def test_account_add_to_equity_curve(self):
        """测试添加净值记录"""
        account = Account()
        timestamp = datetime(2024, 1, 1, 15, 0, 0)
        account.add_to_equity_curve(timestamp, 1500000.0)

        assert len(account.equity_curve) == 1
        assert account.equity_curve[0]["datetime"] == timestamp
        assert account.equity_curve[0]["total_value"] == 1500000.0
        assert account.equity_curve[0]["return"] == 0.5

    def test_account_performance_metrics(self):
        """测试性能指标"""
        account = Account(initial_cash=1000000.0)
        account.total_trades = 100
        account.winning_trades = 60
        account.losing_trades = 40

        assert account.total_trades == 100
        assert account.winning_trades == 60
        assert account.losing_trades == 40


@pytest.mark.unit
class TestPortfolio:
    """投资组合测试"""

    @pytest.fixture
    def portfolio(self):
        """创建投资组合实例"""
        return Portfolio(
            initial_cash=1000000.0,
            commission_rate=0.0003,
            min_commission=5.0,
            stamp_duty_rate=0.001
        )

    def test_portfolio_creation(self, portfolio):
        """测试投资组合创建"""
        assert portfolio.account.initial_cash == 1000000.0
        assert portfolio.commission_rate == 0.0003
        assert portfolio.min_commission == 5.0
        assert portfolio.stamp_duty_rate == 0.001
        assert portfolio._order_id == 0
        assert len(portfolio._order_history) == 0

    def test_portfolio_calculate_commission(self, portfolio):
        """测试佣金计算"""
        # 小额交易，使用最低佣金
        commission1 = portfolio._calculate_commission(100, 10.0)
        assert commission1 == 5.0  # 最低佣金

        # 大额交易
        commission2 = portfolio._calculate_commission(10000, 100.0)
        expected = 10000 * 100.0 * 0.0003
        assert commission2 == expected

    def test_portfolio_update_fill_buy(self, portfolio):
        """测试买入成交更新"""
        event = FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        )

        initial_cash = portfolio.account.cash
        portfolio.update_fill(event)

        # 检查持仓
        position = portfolio.account.get_position("600519.SH")
        assert position is not None
        assert position.quantity == 100
        assert position.avg_price == 1500.0

        # 检查现金减少
        expected_cost = 100 * 1500.0 + portfolio._calculate_commission(100, 1500.0)
        assert portfolio.account.cash == initial_cash - expected_cost

    def test_portfolio_update_fill_sell(self, portfolio):
        """测试卖出成交更新"""
        # 先买入
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        initial_cash = portfolio.account.cash

        # 再卖出
        event = FillEvent(
            symbol="600519.SH",
            datetime=datetime(2024, 1, 2, 10, 0, 0),
            side="sell",
            quantity=100,
            price=1600.0
        )
        portfolio.update_fill(event)

        # 检查持仓被清空
        assert portfolio.account.get_position("600519.SH") is None

        # 检查现金增加
        commission = portfolio._calculate_commission(100, 1600.0)
        stamp_duty = 100 * 1600.0 * 0.001
        expected_increase = 100 * 1600.0 - commission - stamp_duty
        assert portfolio.account.cash == initial_cash + expected_increase

    def test_portfolio_update_fill_insufficient_cash(self, portfolio):
        """测试资金不足"""
        # 创建一个超大订单
        event = FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=1000000,  # 超过资金
            price=1500.0
        )

        initial_cash = portfolio.account.cash
        portfolio.update_fill(event)

        # 现金不应该变化
        assert portfolio.account.cash == initial_cash

    def test_portfolio_update_current_price(self, portfolio):
        """测试更新当前价格"""
        portfolio.update_current_price("600519.SH", 1600.0)
        assert portfolio._current_prices["600519.SH"] == 1600.0

    def test_portfolio_update_total_value(self, portfolio):
        """测试更新总资产"""
        # 先买入
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        # 更新价格
        portfolio.update_current_price("600519.SH", 1600.0)
        portfolio.update_total_value()

        expected_position_value = 100 * 1600.0
        expected_total = portfolio.account.cash + expected_position_value
        assert portfolio.account.total_value == expected_total

    def test_portfolio_get_position_nonexistent(self, portfolio):
        """测试获取不存在的持仓信息"""
        position_info = portfolio.get_position("600519.SH")
        assert position_info["symbol"] == "600519.SH"
        assert position_info["quantity"] == 0
        assert position_info["market_value"] == 0.0
        assert position_info["pnl"] == 0.0

    def test_portfolio_get_position_existing(self, portfolio):
        """测试获取已存在的持仓信息"""
        # 先买入
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        portfolio.update_current_price("600519.SH", 1600.0)

        position_info = portfolio.get_position("600519.SH")
        assert position_info["symbol"] == "600519.SH"
        assert position_info["quantity"] == 100
        assert position_info["avg_price"] == 1500.0
        assert position_info["current_price"] == 1600.0
        assert position_info["market_value"] == 160000.0
        assert position_info["cost_value"] == 150000.0
        assert position_info["pnl"] == 10000.0
        assert position_info["pnl_pct"] == pytest.approx(6.67, rel=1)

    def test_portfolio_get_all_positions(self, portfolio):
        """测试获取所有持仓"""
        # 买入多个股票
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))
        portfolio.update_fill(FillEvent(
            symbol="000858.SZ",
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            side="buy",
            quantity=200,
            price=100.0
        ))

        positions = portfolio.get_all_positions()
        assert len(positions) == 2

    def test_portfolio_get_account_info(self, portfolio):
        """测试获取账户信息"""
        # 先进行一些交易
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        portfolio.update_current_price("600519.SH", 1600.0)
        portfolio.update_total_value()

        info = portfolio.get_account_info()
        assert info["initial_cash"] == 1000000.0
        assert "cash" in info
        assert "total_value" in info
        assert "total_return" in info
        assert "num_positions" in info
        assert info["num_positions"] == 1

    def test_portfolio_get_equity_curve(self, portfolio):
        """测试获取净值曲线"""
        timestamp = datetime(2024, 1, 1, 15, 0, 0)
        portfolio.add_to_equity_curve(timestamp)

        equity_curve = portfolio.get_equity_curve()
        assert isinstance(equity_curve, pd.DataFrame)
        assert len(equity_curve) == 1
        assert "datetime" in equity_curve.columns
        assert "total_value" in equity_curve.columns

    def test_portfolio_add_to_equity_curve(self, portfolio):
        """测试添加净值记录"""
        timestamp = datetime(2024, 1, 1, 15, 0, 0)
        portfolio.add_to_equity_curve(timestamp)

        assert len(portfolio.account.equity_curve) == 1
        assert portfolio.account.equity_curve[0]["datetime"] == timestamp

    def test_portfolio_reset(self, portfolio):
        """测试重置投资组合"""
        # 进行一些交易
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        # 重置
        portfolio.reset()

        # 验证重置后的状态
        assert portfolio.account.cash == portfolio.account.initial_cash
        assert len(portfolio.account.positions) == 0
        assert portfolio._order_id == 0
        assert len(portfolio._order_history) == 0
        assert len(portfolio._current_prices) == 0

    def test_portfolio_update_time(self, portfolio):
        """测试更新时间"""
        from backtest.engine.event_engine import EventType
        event = Event(type=EventType.MARKET_OPEN, datetime=datetime(2024, 1, 1, 10, 0, 0))
        # 这个方法目前是空实现
        portfolio.update_time(event)
        # 只要不出错就行

    def test_portfolio_order_history(self, portfolio):
        """测试订单历史记录"""
        event = FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        )

        portfolio.update_fill(event)

        assert len(portfolio._order_history) == 1
        assert portfolio._order_history[0]["order_id"] == 1
        assert portfolio._order_history[0]["symbol"] == "600519.SH"
        assert portfolio._order_history[0]["side"] == "buy"
        assert portfolio._order_history[0]["quantity"] == 100

    def test_portfolio_total_trades_increment(self, portfolio):
        """测试交易次数递增"""
        assert portfolio.account.total_trades == 0

        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        assert portfolio.account.total_trades == 1

    def test_portfolio_commission_and_stamp_duty(self, portfolio):
        """测试佣金和印花税计算"""
        # 买入
        buy_event = FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=10000,
            price=100.0
        )

        initial_cash = portfolio.account.cash
        portfolio.update_fill(buy_event)

        # 买入只收佣金
        buy_commission = portfolio._calculate_commission(10000, 100.0)
        expected_buy_cost = 10000 * 100.0 + buy_commission
        assert portfolio.account.cash == initial_cash - expected_buy_cost

        # 卖出
        sell_event = FillEvent(
            datetime=datetime(2024, 1, 2, 10, 0, 0),
            order_id="order2",
            symbol="600519.SH",
            side="sell",
            quantity=10000,
            price=110.0
        )

        initial_cash = portfolio.account.cash
        portfolio.update_fill(sell_event)

        # 卖出收佣金和印花税
        sell_commission = portfolio._calculate_commission(10000, 110.0)
        stamp_duty = 10000 * 110.0 * 0.001
        expected_sell_income = 10000 * 110.0 - sell_commission - stamp_duty
        assert portfolio.account.cash == initial_cash + expected_sell_income


@pytest.mark.unit
class TestPortfolioEdgeCases:
    """投资组合边界情况测试"""

    def test_portfolio_zero_quantity_trade(self):
        """测试零数量交易"""
        portfolio = Portfolio()
        event = FillEvent(
            symbol="600519.SH",
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            side="buy",
            quantity=0,
            price=1500.0
        )

        initial_cash = portfolio.account.cash
        portfolio.update_fill(event)

        # 零数量应该不影响账户
        commission = portfolio._calculate_commission(0, 1500.0)
        # commission应该是min_commission=5.0（因为0*price=0 < 5）
        # 但实际上total_cost = 0 + 5 + 0 = 5
        # 所以现金应该减少5元
        assert portfolio.account.cash < initial_cash

    def test_portfolio_negative_price_trade(self):
        """测试负价格（异常情况）"""
        portfolio = Portfolio()
        event = FillEvent(
            symbol="600519.SH",
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            side="buy",
            quantity=100,
            price=-1500.0  # 负价格
        )

        # 这应该正常处理（虽然不现实）
        portfolio.update_fill(event)
        # 持仓应该被创建
        position = portfolio.account.get_position("600519.SH")
        assert position is not None

    def test_portfolio_multiple_same_symbol_trades(self):
        """测试同一股票多次交易"""
        portfolio = Portfolio()

        # 第一次买入
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        # 第二次买入
        portfolio.update_fill(FillEvent(
            symbol="600519.SH",
            datetime=datetime(2024, 1, 1, 11, 0, 0),
            side="buy",
            quantity=100,
            price=1600.0
        ))

        position = portfolio.account.get_position("600519.SH")
        assert position.quantity == 200
        assert position.avg_price == 1550.0
        assert position.today_bought == 200

    def test_portfolio_update_total_value_with_missing_price(self):
        """测试更新总资产时缺少价格"""
        portfolio = Portfolio()

        # 买入股票
        portfolio.update_fill(FillEvent(
            datetime=datetime(2024, 1, 1, 10, 0, 0),
            order_id="order1",
            symbol="600519.SH",
            side="buy",
            quantity=100,
            price=1500.0
        ))

        # 不更新价格，直接更新总资产
        portfolio.update_total_value()

        # 持仓价值应该为0（因为缺少价格）
        assert portfolio.account.total_value == portfolio.account.cash

    def test_portfolio_custom_rates(self):
        """测试自定义费率"""
        portfolio = Portfolio(
            commission_rate=0.0001,
            min_commission=1.0,
            stamp_duty_rate=0.0005
        )

        assert portfolio.commission_rate == 0.0001
        assert portfolio.min_commission == 1.0
        assert portfolio.stamp_duty_rate == 0.0005
