"""
测试策略基类和具体策略
"""

from datetime import datetime

import pandas as pd
import pytest

from backtest.engine.event_engine import BarEvent, EventQueue
from backtest.engine.strategy import (
    BuyAndHoldStrategy,
    MeanReversionStrategy,
    MovingAverageCrossStrategy,
    Strategy,
)


@pytest.mark.unit
class TestStrategy:
    """测试策略基类"""

    def test_strategy_creation(self):
        """测试策略创建"""
        strategy = Strategy()
        assert strategy is not None
        assert strategy.event_queue is None
        assert strategy.data_handler is None
        assert strategy.portfolio is None

    def test_generate_order_id(self):
        """测试生成订单ID"""
        strategy = Strategy()

        id1 = strategy.generate_order_id()
        id2 = strategy.generate_order_id()
        id3 = strategy.generate_order_id()

        assert id1 == "order_1"
        assert id2 == "order_2"
        assert id3 == "order_3"

    def test_set_event_queue(self):
        """测试设置事件队列"""
        strategy = Strategy()
        queue = EventQueue()

        strategy.set_event_queue(queue)
        assert strategy.event_queue == queue

    def test_buy_no_queue(self):
        """测试没有事件队列时买入"""
        strategy = Strategy()

        result = strategy.buy("000001.SZ", 1000, 100.0)
        assert result is None

    def test_buy_with_queue(self):
        """测试有事件队列时买入"""
        strategy = Strategy()
        queue = EventQueue()
        strategy.set_event_queue(queue)

        order_id = strategy.buy("000001.SZ", 1000, 100.0)

        assert order_id == "order_1"
        assert not queue.empty()

        # 取出订单检查
        event = queue.get()
        assert event.symbol == "000001.SZ"
        assert event.side == "buy"
        assert event.quantity == 1000
        assert event.price == 100.0

    def test_sell_with_queue(self):
        """测试卖出"""
        strategy = Strategy()
        queue = EventQueue()
        strategy.set_event_queue(queue)

        order_id = strategy.sell("600000.SH", 500, 50.0)

        assert order_id == "order_1"
        assert not queue.empty()

        event = queue.get()
        assert event.symbol == "600000.SH"
        assert event.side == "sell"
        assert event.quantity == 500

    def test_market_order(self):
        """测试市价单"""
        strategy = Strategy()
        queue = EventQueue()
        strategy.set_event_queue(queue)

        order_id = strategy.buy("000001.SZ", 1000, None, "market")

        event = queue.get()
        assert event.price is None
        assert event.order_type == "market"


@pytest.mark.unit
class TestBuyAndHoldStrategy:
    """测试买入持有策略"""

    def test_strategy_creation(self):
        """测试策略创建"""
        strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=1000)

        assert strategy.symbol == "000001.SZ"
        assert strategy.quantity == 1000
        assert strategy._bought is False

    def test_on_bar_first_bar(self):
        """测试第一根K线"""
        strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=1000)
        queue = EventQueue()
        strategy.set_event_queue(queue)

        # 第一根K线
        bar = BarEvent(
            datetime=datetime.now(),
            symbol="000001.SZ",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
        )

        strategy.on_bar(bar)

        # 应该发送买入订单
        assert strategy._bought is True
        assert not queue.empty()

        order = queue.get()
        assert order.symbol == "000001.SZ"
        assert order.side == "buy"
        assert order.quantity == 1000

    def test_on_bar_subsequent_bars(self):
        """测试后续K线"""
        strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=1000)
        queue = EventQueue()
        strategy.set_event_queue(queue)

        # 第一根K线
        bar1 = BarEvent(
            datetime=datetime.now(),
            symbol="000001.SZ",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
        )
        strategy.on_bar(bar1)

        # 清空队列
        while not queue.empty():
            queue.get()

        # 第二根K线
        bar2 = BarEvent(
            datetime=datetime.now(),
            symbol="000001.SZ",
            open=102.0,
            high=107.0,
            low=97.0,
            close=104.0,
            volume=1000000,
        )
        strategy.on_bar(bar2)

        # 不应该有新订单
        assert queue.empty()

    def test_on_bar_wrong_symbol(self):
        """测试错误的股票代码"""
        strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=1000)
        queue = EventQueue()
        strategy.set_event_queue(queue)

        # 不同股票的K线
        bar = BarEvent(
            datetime=datetime.now(),
            symbol="600000.SH",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
        )

        strategy.on_bar(bar)

        # 不应该有订单
        assert queue.empty()
        assert strategy._bought is False


@pytest.mark.unit
class TestMovingAverageCrossStrategy:
    """测试双均线策略"""

    def test_strategy_creation(self):
        """测试策略创建"""
        strategy = MovingAverageCrossStrategy(
            symbol="000001.SZ", fast_period=5, slow_period=20, quantity=1000
        )

        assert strategy.symbol == "000001.SZ"
        assert strategy.fast_period == 5
        assert strategy.slow_period == 20
        assert strategy.quantity == 1000
        assert len(strategy.price_history) == 0

    def test_on_bar_insufficient_data(self):
        """测试数据不足"""
        strategy = MovingAverageCrossStrategy(
            symbol="000001.SZ", fast_period=5, slow_period=20, quantity=1000
        )
        queue = EventQueue()
        strategy.set_event_queue(queue)

        # 只提供10根K线，不足20根
        for i in range(10):
            bar = BarEvent(
                datetime=datetime.now(),
                symbol="000001.SZ",
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=100.0 + i,
                volume=1000000,
            )
            strategy.on_bar(bar)

        # 不应该有订单
        assert queue.empty()

    def test_on_bar_golden_cross(self):
        """测试金叉"""
        strategy = MovingAverageCrossStrategy(
            symbol="000001.SZ", fast_period=3, slow_period=5, quantity=1000
        )
        queue = EventQueue()
        strategy.set_event_queue(queue)

        # 创建价格序列，前5根K线慢线在上（下跌趋势），然后快速上涨形成金叉
        # 价格序列：[110, 100, 90, 80, 70, 90, 110]
        # Bar 5 (index 5): price=90, fast_ma=80, slow_ma=86, fast < slow
        # Bar 6 (index 6): price=110, fast_ma=90, slow_ma=88, fast > slow (金叉!)
        prices = [110, 100, 90, 80, 70, 90, 110]

        for i, price in enumerate(prices):
            bar = BarEvent(
                datetime=datetime.now(),
                symbol="000001.SZ",
                open=price,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000,
            )
            strategy.on_bar(bar)

        # 应该有买入订单
        assert not queue.empty()

        order = queue.get()
        assert order.side == "buy"
        assert order.quantity == 1000


@pytest.mark.unit
class TestMeanReversionStrategy:
    """测试均值回归策略"""

    def test_strategy_creation(self):
        """测试策略创建"""
        strategy = MeanReversionStrategy(
            symbol="000001.SZ", period=20, std_threshold=2.0, quantity=1000
        )

        assert strategy.symbol == "000001.SZ"
        assert strategy.period == 20
        assert strategy.std_threshold == 2.0
        assert strategy.quantity == 1000

    def test_on_bar_insufficient_data(self):
        """测试数据不足"""
        strategy = MeanReversionStrategy(
            symbol="000001.SZ", period=20, std_threshold=2.0, quantity=1000
        )
        queue = EventQueue()
        strategy.set_event_queue(queue)

        # 只提供10根K线
        for i in range(10):
            bar = BarEvent(
                datetime=datetime.now(),
                symbol="000001.SZ",
                open=100.0,
                high=105.0,
                low=95.0,
                close=100.0,
                volume=1000000,
            )
            strategy.on_bar(bar)

        # 不应该有订单
        assert queue.empty()
