"""
测试回测策略
测试backtest/engine/strategies.py中的各种策略
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backtest.engine.event_engine import BarEvent, EventQueue
from backtest.engine.portfolio import Portfolio
from backtest.engine.strategies import (
    BollingerBandsStrategy,
    BreakoutStrategy,
    BuyAndHoldStrategy,
    DualThrustStrategy,
    GridTradingStrategy,
    MACDStrategy,
    MomentumStrategy,
    RSIStrategy,
)


@pytest.fixture
def sample_data():
    """生成测试数据"""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = 100 + np.cumsum(np.random.randn(100) * 2)

    df = pd.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
            "high": prices * (1 + np.random.uniform(0, 0.02, 100)),
            "low": prices * (1 - np.random.uniform(0, 0.02, 100)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, 100),
        }
    )

    # 确保high >= close >= low
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99

    return df


@pytest.fixture
def event_queue():
    """创建事件队列"""
    return EventQueue()


@pytest.fixture
def portfolio():
    """创建投资组合"""
    return Portfolio(initial_cash=1000000.0)


@pytest.mark.unit
class TestBuyAndHoldStrategy:
    """测试买入持有策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = BuyAndHoldStrategy(symbol="600519.SH", quantity=1000)
        assert strategy.symbol == "600519.SH"
        assert strategy.quantity == 1000
        assert strategy.bought is False

    def test_on_bar_buy(self, event_queue, portfolio):
        """测试买入逻辑"""
        strategy = BuyAndHoldStrategy(symbol="600519.SH", quantity=100)
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        event = BarEvent(
            datetime=datetime.now(),
            symbol="600519.SH",
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000000,
        )

        strategy.on_bar(event)

        # 应该生成买入订单
        order_event = event_queue.get()
        assert order_event is not None
        assert strategy.bought is True


@pytest.mark.unit
class TestBollingerBandsStrategy:
    """测试布林带策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = BollingerBandsStrategy(symbol="600519.SH", period=20, std_dev=2.0, quantity=1000)
        assert strategy.symbol == "600519.SH"
        assert strategy.period == 20
        assert strategy.std_dev == 2.0

    def test_on_bar_insufficient_data(self, event_queue, portfolio):
        """测试数据不足时不会交易"""
        strategy = BollingerBandsStrategy(symbol="600519.SH", period=20)
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        # 数据不足，不应生成信号
        for i in range(10):
            event = BarEvent(
                datetime=datetime.now(),
                symbol="600519.SH",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000,
            )
            strategy.on_bar(event)

        # 队列应该为空
        assert event_queue.empty()


@pytest.mark.unit
class TestMACDStrategy:
    """测试MACD策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = MACDStrategy(
            symbol="600519.SH", fast_period=12, slow_period=26, signal_period=9, quantity=1000
        )
        assert strategy.symbol == "600519.SH"
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26

    def test_macd_calculation(self, event_queue, portfolio):
        """测试MACD计算"""
        strategy = MACDStrategy(symbol="600519.SH")
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        # 模拟数据 - 需要至少 slow_period + signal_period = 35 个数据点
        for i in range(40):
            event = BarEvent(
                datetime=datetime.now(),
                symbol="600519.SH",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000,
            )
            strategy.on_bar(event)

        # MACD历史应该有数据
        assert len(strategy.macd_history) > 0
        assert len(strategy.signal_history) > 0


@pytest.mark.unit
class TestRSIStrategy:
    """测试RSI策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = RSIStrategy(
            symbol="600519.SH", period=14, oversold=30.0, overbought=70.0, quantity=1000
        )
        assert strategy.period == 14
        assert strategy.oversold == 30.0
        assert strategy.overbought == 70.0

    def test_rsi_calculation(self, event_queue, portfolio):
        """测试RSI计算"""
        strategy = RSIStrategy(symbol="600519.SH", period=14)
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        # 模拟价格数据
        for i in range(20):
            event = BarEvent(
                datetime=datetime.now(),
                symbol="600519.SH",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000000,
            )
            strategy.on_bar(event)

        # 价格历史应该有数据
        assert len(strategy.price_history) > 0


@pytest.mark.unit
class TestBreakoutStrategy:
    """测试突破策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = BreakoutStrategy(symbol="600519.SH", period=20, quantity=1000)
        assert strategy.period == 20
        assert strategy.entry_price is None
        assert strategy.position_side is None

    def test_breakout_logic(self, event_queue, portfolio):
        """测试突破逻辑"""
        strategy = BreakoutStrategy(symbol="600519.SH", period=5)
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        # 模拟突破
        for i in range(10):
            event = BarEvent(
                datetime=datetime.now(),
                symbol="600519.SH",
                open=100.0 + i * 0.5,
                high=105.0 + i,
                low=99.0 + i,
                close=100.0 + i,
                volume=1000000,
            )
            strategy.on_bar(event)

        assert len(strategy.price_history) > 0


@pytest.mark.unit
class TestDualThrustStrategy:
    """测试Dual Thrust策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = DualThrustStrategy(symbol="600519.SH", period=10, k1=0.5, k2=0.5, quantity=1000)
        assert strategy.period == 10
        assert strategy.k1 == 0.5
        assert strategy.k2 == 0.5

    def test_dual_thrust_calculation(self, event_queue, portfolio):
        """测试Dual Thrust计算"""
        strategy = DualThrustStrategy(symbol="600519.SH", period=5)
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        # 模拟数据
        for i in range(15):
            event = BarEvent(
                datetime=datetime.now(),
                symbol="600519.SH",
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=100.0 + i,
                volume=1000000,
            )
            strategy.on_bar(event)

        assert len(strategy.bar_history) > 0


@pytest.mark.unit
class TestGridTradingStrategy:
    """测试网格交易策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = GridTradingStrategy(
            symbol="600519.SH",
            base_price=100.0,  # Fixed: uses base_price instead of lower_price/upper_price
            grid_count=5,
            grid_spacing=0.02,  # Fixed: explicit grid_spacing parameter
            grid_quantity=100,
        )
        assert strategy.base_price == 100.0
        assert strategy.grid_count == 5
        assert strategy.grid_spacing == 0.02
        assert len(strategy.buy_grids) == 5
        assert len(strategy.sell_grids) == 5

    def test_grid_setup(self):
        """测试网格设置"""
        strategy = GridTradingStrategy(
            symbol="600519.SH",
            base_price=100.0,  # Fixed: uses base_price
            grid_count=5,
            grid_spacing=0.02,
        )

        # 验证网格设置
        # Buy grids: base_price * (1 - i * spacing) for i in 0..4
        # [100.0, 98.0, 96.0, 94.0, 92.0]
        assert len(strategy.buy_grids) == 5
        assert strategy.buy_grids[0] == 100.0
        assert strategy.buy_grids[4] == 100.0 * (1 - 4 * 0.02)

        # Sell grids: base_price * (1 + i * spacing) for i in 1..5
        # [102.0, 104.0, 106.0, 108.0, 110.0]
        assert len(strategy.sell_grids) == 5
        assert strategy.sell_grids[0] == 100.0 * (1 + 1 * 0.02)


@pytest.mark.unit
class TestMomentumStrategy:
    """测试动量策略"""

    def test_initialization(self):
        """测试初始化"""
        strategy = MomentumStrategy(
            symbol="600519.SH", lookback=10, momentum_threshold=0.03, quantity=1000
        )
        assert strategy.lookback == 10
        assert strategy.momentum_threshold == 0.03

    def test_momentum_calculation(self, event_queue, portfolio):
        """测试动量计算"""
        strategy = MomentumStrategy(symbol="600519.SH", lookback=5)
        strategy.set_event_queue(event_queue)
        strategy.set_portfolio(portfolio)

        # 模拟上涨趋势
        for i in range(20):
            event = BarEvent(
                datetime=datetime.now(),
                symbol="600519.SH",
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.0 + i * 1.01,
                volume=1000000,
            )
            strategy.on_bar(event)

        assert len(strategy.price_history) > 0
