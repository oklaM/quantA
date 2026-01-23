"""
测试订单执行处理器
"""

from datetime import datetime

import numpy as np
import pytest

from backtest.engine.a_share_rules import Order, Position
from backtest.engine.event_engine import FillEvent, OrderEvent
from backtest.engine.execution import (
    ExecutionHandler,
    SimulationExecutionHandler,
    SlippageModel,
    TransactionCost,
)


@pytest.mark.unit
class TestSlippageModel:
    """测试滑点模型"""

    def test_slippage_model_creation(self):
        """测试滑点模型创建"""
        model = SlippageModel(enabled=True, mode="percentage", rate=0.0001, max_slippage=0.001)

        assert model.enabled is True
        assert model.mode == "percentage"
        assert model.rate == 0.0001
        assert model.max_slippage == 0.001

    def test_slippage_model_default(self):
        """测试滑点模型默认值"""
        model = SlippageModel()

        assert model.enabled is True
        assert model.mode == "linear"
        assert model.rate == 0.0001


@pytest.mark.unit
class TestTransactionCost:
    """测试交易成本"""

    def test_transaction_cost_creation(self):
        """测试交易成本创建"""
        cost = TransactionCost(commission_rate=0.0003, min_commission=5.0, stamp_duty_rate=0.001)

        assert cost.commission_rate == 0.0003
        assert cost.min_commission == 5.0
        assert cost.stamp_duty_rate == 0.001


@pytest.mark.unit
class TestExecutionHandler:
    """测试执行处理器"""

    def test_execution_handler_creation(self):
        """测试执行处理器创建"""
        handler = ExecutionHandler()
        assert handler is not None
        assert handler._current_bars == {}

    def test_update_bar(self):
        """测试更新K线数据"""
        handler = ExecutionHandler()

        bar_data = {
            "symbol": "000001.SZ",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
            "prev_close": 101.0,
        }

        handler.update_bar(bar_data)

        assert "000001.SZ" in handler._current_bars
        assert handler._current_bars["000001.SZ"]["close"] == 102.0

    def test_calculate_commission(self):
        """测试佣金计算"""
        cost = TransactionCost(commission_rate=0.0003, min_commission=5.0)
        handler = ExecutionHandler(transaction_cost=cost)

        # 1000股 @ 100元 = 100000元
        # 佣金 = 100000 * 0.0003 = 30元
        commission = handler._calculate_commission(1000, 100.0)
        assert np.isclose(commission, 30.0)

    def test_calculate_commission_minimum(self):
        """测试最低佣金"""
        cost = TransactionCost(commission_rate=0.0003, min_commission=5.0)
        handler = ExecutionHandler(transaction_cost=cost)

        # 100股 @ 10元 = 1000元
        # 佣金 = 1000 * 0.0003 = 0.3元，但最低5元
        commission = handler._calculate_commission(100, 10.0)
        assert commission == 5.0

    def test_calculate_stamp_duty_buy(self):
        """测试买入印花税（应该为0）"""
        handler = ExecutionHandler()

        stamp_duty = handler.calculate_stamp_duty(1000, 100.0, "buy")
        assert stamp_duty == 0.0

    def test_calculate_stamp_duty_sell(self):
        """测试卖出印花税"""
        handler = ExecutionHandler()

        # 1000股 @ 100元 * 0.001 = 100元
        stamp_duty = handler.calculate_stamp_duty(1000, 100.0, "sell")
        assert stamp_duty == 100.0

    def test_execute_order_no_data(self):
        """测试没有市场数据时的订单执行"""
        handler = ExecutionHandler()

        order_event = OrderEvent(
            datetime=datetime.now(),
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=100.0,
        )

        result = handler.execute_order(order_event)
        assert result is None

    def test_update_today_bought(self):
        """测试更新今日买入"""
        handler = ExecutionHandler()

        handler.update_today_bought("000001.SZ", 1000)
        handler.update_today_bought("000001.SZ", 500)

        assert handler._today_bought["000001.SZ"] == 1500

    def test_reset_today_bought(self):
        """测试重置今日买入"""
        handler = ExecutionHandler()

        handler.update_today_bought("000001.SZ", 1000)
        handler.reset_today_bought()

        assert len(handler._today_bought) == 0


@pytest.mark.unit
class TestSimulationExecutionHandler:
    """测试模拟执行处理器"""

    def test_creation(self):
        """测试创建"""
        handler = SimulationExecutionHandler(
            commission_rate=0.0003,
            slippage_rate=0.0001,
        )

        assert handler is not None
        assert handler.slippage_model.enabled is True
        assert handler.slippage_model.mode == "percentage"
        assert handler.fill_at_close is True

    def test_execute_order_success(self):
        """测试成功执行订单"""
        handler = SimulationExecutionHandler(
            commission_rate=0.0003,
            slippage_rate=0.0,  # 关闭滑点
        )

        # 更新市场数据
        handler.update_bar(
            {
                "symbol": "000001.SZ",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 102.0,
                "volume": 1000000,
                "prev_close": 101.0,
            }
        )

        order_event = OrderEvent(
            datetime=datetime.now(),
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=100.0,
        )

        fill_event = handler.execute_order(order_event)

        assert fill_event is not None
        assert fill_event.symbol == "000001.SZ"
        assert fill_event.side == "buy"
        assert fill_event.quantity == 1000
        # 应该以收盘价成交
        assert fill_event.price == 102.0
        # 佣金 = 1000 * 102 * 0.0003 = 30.6
        assert np.isclose(fill_event.commission, 30.6)

    def test_execute_order_with_slippage(self):
        """测试带滑点的订单执行"""
        handler = SimulationExecutionHandler(
            commission_rate=0.0003,
            slippage_rate=0.001,  # 0.1%滑点
        )

        handler.update_bar(
            {
                "symbol": "000001.SZ",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000000,
                "prev_close": 100.0,
            }
        )

        order_event = OrderEvent(
            datetime=datetime.now(),
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=100.0,
        )

        fill_event = handler.execute_order(order_event)

        # 买入应该有滑点，价格应该高于100
        assert fill_event.price > 100.0
        # 滑点后应该在合理范围内
        assert fill_event.price <= 105.0  # 不超过最高价

    def test_execute_order_sell_with_slippage(self):
        """测试卖出带滑点"""
        handler = SimulationExecutionHandler(
            commission_rate=0.0003,
            slippage_rate=0.001,
        )

        handler.update_bar(
            {
                "symbol": "000001.SZ",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000000,
                "prev_close": 100.0,
            }
        )

        order_event = OrderEvent(
            datetime=datetime.now(),
            order_id="order_1",
            symbol="000001.SZ",
            side="sell",
            quantity=1000,
            price=100.0,
        )

        fill_event = handler.execute_order(order_event)

        # 卖出应该有滑点，价格应该低于100
        assert fill_event.price < 100.0
        # 滑点后应该在合理范围内
        assert fill_event.price >= 95.0  # 不低于最低价
