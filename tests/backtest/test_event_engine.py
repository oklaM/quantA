"""
测试事件引擎
"""

from datetime import datetime, timedelta

import pytest

from backtest.engine.event_engine import (
    BarEvent,
    Event,
    EventQueue,
    EventType,
    FillEvent,
    OrderEvent,
)


@pytest.mark.unit
class TestEventType:
    """测试事件类型枚举"""

    def test_event_type_values(self):
        """测试事件类型值"""
        assert EventType.MARKET_OPEN.value == "market_open"
        assert EventType.BAR.value == "bar"
        assert EventType.ORDER_FILLED.value == "order_filled"


@pytest.mark.unit
class TestEvent:
    """测试事件基类"""

    def test_event_creation(self):
        """测试事件创建"""
        dt = datetime.now()
        event = Event(type=EventType.BAR, datetime=dt, data={"test": "value"})
        assert event.type == EventType.BAR
        assert event.datetime == dt
        assert event.data == {"test": "value"}

    def test_event_repr(self):
        """测试事件字符串表示"""
        dt = datetime.now()
        event = Event(type=EventType.BAR, datetime=dt)
        repr_str = repr(event)
        assert "BAR" in repr_str


@pytest.mark.unit
class TestBarEvent:
    """测试K线事件"""

    def test_bar_event_creation(self):
        """测试K线事件创建"""
        dt = datetime.now()
        bar = BarEvent(
            datetime=dt,
            symbol="000001.SZ",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
        )

        assert bar.symbol == "000001.SZ"
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 95.0
        assert bar.close == 102.0
        assert bar.volume == 1000000
        assert bar.type == EventType.BAR

    def test_bar_event_data(self):
        """测试K线事件数据"""
        dt = datetime.now()
        bar = BarEvent(
            datetime=dt,
            symbol="000001.SZ",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
        )

        expected_data = {
            "symbol": "000001.SZ",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
        }
        assert bar.data == expected_data


@pytest.mark.unit
class TestOrderEvent:
    """测试订单事件"""

    def test_order_event_creation(self):
        """测试订单事件创建"""
        dt = datetime.now()
        order = OrderEvent(
            datetime=dt,
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=100.0,
            order_type="limit",
        )

        assert order.order_id == "order_1"
        assert order.symbol == "000001.SZ"
        assert order.side == "buy"
        assert order.quantity == 1000
        assert order.price == 100.0
        assert order.order_type == "limit"
        assert order.type == EventType.ORDER_SUBMITTED

    def test_market_order(self):
        """测试市价单"""
        dt = datetime.now()
        order = OrderEvent(
            datetime=dt,
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=None,
            order_type="market",
        )

        assert order.price is None
        assert order.order_type == "market"


@pytest.mark.unit
class TestFillEvent:
    """测试成交事件"""

    def test_fill_event_creation(self):
        """测试成交事件创建"""
        dt = datetime.now()
        fill = FillEvent(
            datetime=dt,
            order_id="order_1",
            symbol="000001.SZ",
            side="buy",
            quantity=1000,
            price=100.5,
            commission=5.0,
        )

        assert fill.order_id == "order_1"
        assert fill.symbol == "000001.SZ"
        assert fill.side == "buy"
        assert fill.quantity == 1000
        assert fill.price == 100.5
        assert fill.commission == 5.0
        assert fill.type == EventType.ORDER_FILLED


@pytest.mark.unit
class TestEventQueue:
    """测试事件队列"""

    def test_event_queue_creation(self):
        """测试事件队列创建"""
        queue = EventQueue()
        assert queue.empty()
        assert queue.size() == 0

    def test_event_queue_put_get(self):
        """测试事件队列放入和获取"""
        queue = EventQueue()
        dt = datetime.now()

        event = Event(
            type=EventType.BAR,
            datetime=dt,
        )

        queue.put(event)
        assert not queue.empty()
        assert queue.size() == 1

        retrieved_event = queue.get()
        assert retrieved_event is not None
        assert retrieved_event.type == EventType.BAR
        assert queue.empty()

    def test_event_queue_priority(self):
        """测试事件队列优先级（按时间排序）"""
        queue = EventQueue()

        dt1 = datetime.now()
        dt2 = dt1 + timedelta(seconds=1)
        dt3 = dt1 + timedelta(seconds=2)

        event3 = Event(type=EventType.BAR, datetime=dt3)
        event1 = Event(type=EventType.BAR, datetime=dt1)
        event2 = Event(type=EventType.BAR, datetime=dt2)

        # 乱序放入
        queue.put(event3)
        queue.put(event1)
        queue.put(event2)

        # 应该按时间顺序取出
        assert queue.get().datetime == dt1
        assert queue.get().datetime == dt2
        assert queue.get().datetime == dt3
        assert queue.empty()

    def test_event_queue_get_empty(self):
        """测试从空队列获取"""
        queue = EventQueue()
        event = queue.get()
        assert event is None

    def test_event_queue_multiple_operations(self):
        """测试多次操作"""
        queue = EventQueue()
        dt = datetime.now()

        # 放入多个事件
        for i in range(5):
            event = Event(
                type=EventType.BAR,
                datetime=dt + timedelta(seconds=i),
            )
            queue.put(event)

        assert queue.size() == 5

        # 取出所有事件
        count = 0
        while not queue.empty():
            queue.get()
            count += 1

        assert count == 5
        assert queue.empty()
