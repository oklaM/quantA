"""
事件驱动回测引擎核心架构
实现基于事件的量化回测系统
"""

import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """事件类型"""

    # 市场事件
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    BAR = "bar"  # K线数据

    # 订单事件
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"

    # 其他事件
    TIMER = "timer"
    CUSTOM = "custom"


class Event:
    """事件基类"""

    def __init__(self, type: EventType, datetime: datetime, data: Optional[Dict[str, Any]] = None):
        self.type = type
        self.datetime = datetime
        self.data = data or {}

    def __repr__(self):
        return f"Event({self.type.name}, {self.datetime})"


class BarEvent(Event):
    """K线事件"""

    def __init__(
        self,
        datetime: datetime,
        symbol: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        type: EventType = EventType.BAR,
        data: Optional[Dict[str, Any]] = None,
    ):
        # 如果没有提供data，则用OHLCV数据初始化
        if data is None:
            data = {
                "symbol": symbol,
                "open": open,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        super().__init__(type=type, datetime=datetime, data=data)
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def __repr__(self):
        return f"BarEvent({self.symbol}, {self.datetime}, close={self.close})"


class OrderEvent(Event):
    """订单事件"""

    def __init__(
        self,
        datetime: datetime,
        order_id: str,
        symbol: str,
        side: str,  # 'buy' 或 'sell'
        quantity: int,
        order_type: str = "limit",
        price: Optional[float] = None,
        status: str = "submitted",
        type: EventType = EventType.ORDER_SUBMITTED,
        data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(type=type, datetime=datetime, data=data)
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.status = status

        # 填充data字段
        if not self.data:
            self.data = {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side,
                "quantity": self.quantity,
                "price": self.price,
                "order_type": self.order_type,
                "status": self.status,
            }

    def __repr__(self):
        return f"OrderEvent({self.order_id}, {self.symbol}, {self.side}, {self.quantity})"


class FillEvent(Event):
    """成交事件"""

    def __init__(
        self,
        datetime: datetime,
        order_id: str,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        commission: float = 0.0,
        type: EventType = EventType.ORDER_FILLED,
        data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(type=type, datetime=datetime, data=data)
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.commission = commission

        # 填充data字段
        if not self.data:
            self.data = {
                "order_id": self.order_id,
                "symbol": self.symbol,
                "side": self.side,
                "quantity": self.quantity,
                "price": self.price,
                "commission": self.commission,
            }

    def __repr__(self):
        return f"FillEvent({self.order_id}, {self.symbol}, {self.side}, {self.quantity} @ {self.price})"


class EventQueue:
    """事件队列"""

    def __init__(self):
        self._queue = queue.PriorityQueue()

    def put(self, event: Event):
        """放入事件"""
        # 使用时间戳作为优先级
        if event.datetime is None:
            event.datetime = datetime.now()
        priority = int(event.datetime.timestamp() * 1000000)
        self._queue.put((priority, event))

    def get(self) -> Optional[Event]:
        """获取事件"""
        try:
            _, event = self._queue.get_nowait()
            return event
        except queue.Empty:
            return None

    def empty(self) -> bool:
        """是否为空"""
        return self._queue.empty()

    def size(self) -> int:
        """队列大小"""
        return self._queue.qsize()


class Strategy:
    """策略基类"""

    def __init__(self):
        self.event_queue: Optional[EventQueue] = None
        self.data_handler: Optional["DataHandler"] = None
        self.portfolio: Optional["Portfolio"] = None

    def set_event_queue(self, event_queue: EventQueue):
        """设置事件队列"""
        self.event_queue = event_queue

    def set_data_handler(self, data_handler: "DataHandler"):
        """设置数据处理器"""
        self.data_handler = data_handler

    def set_portfolio(self, portfolio: "Portfolio"):
        """设置投资组合"""
        self.portfolio = portfolio

    def on_bar(self, event: BarEvent):
        """
        处理K线事件（默认实现，子类可覆盖）

        Args:
            event: K线事件
        """
        pass

    def calculate_signals(self, event: BarEvent):
        """
        计算交易信号（由子类实现）

        Args:
            event: K线事件
        """
        pass


class DataHandler(ABC):
    """数据处理器基类"""

    @abstractmethod
    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取下一根K线"""
        pass

    @abstractmethod
    def get_current_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取当前K线"""
        pass

    @abstractmethod
    def reset(self):
        """重置数据处理器"""
        pass


class Portfolio(ABC):
    """投资组合基类"""

    @abstractmethod
    def update_fill(self, event: FillEvent):
        """更新成交"""
        pass

    @abstractmethod
    def update_time(self, event: Event):
        """更新时间"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """获取持仓"""
        pass


class ExecutionHandler(ABC):
    """执行处理器基类"""

    @abstractmethod
    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        """执行订单"""
        pass


class BacktestEngine:
    """
    回测引擎核心
    事件驱动架构
    """

    def __init__(
        self,
        strategy: Strategy,
        data_handler: DataHandler,
        execution_handler: ExecutionHandler,
        portfolio: Portfolio,
    ):
        """
        Args:
            strategy: 策略实例
            data_handler: 数据处理器
            execution_handler: 执行处理器
            portfolio: 投资组合
        """
        self.strategy = strategy
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio

        # 事件队列
        self.event_queue = EventQueue()

        # 连接组件
        self.strategy.set_event_queue(self.event_queue)
        self.strategy.set_data_handler(self.data_handler)
        self.strategy.set_portfolio(self.portfolio)

        # 统计信息
        self._stats = {
            "total_bars": 0,
            "total_orders": 0,
            "total_fills": 0,
            "rejected_orders": 0,
        }

    def run(self) -> Dict[str, Any]:
        """
        运行回测

        Returns:
            回测结果统计
        """
        logger.info("开始回测...")

        try:
            while True:
                # 1. 更新数据
                if not self._update_data():
                    break

                # 2. 处理事件
                self._process_events()

        except KeyboardInterrupt:
            logger.info("回测被中断")
        except Exception as e:
            logger.error(f"回测错误: {e}", exc_info=True)

        logger.info("回测完成")
        return self._stats

    def _update_data(self) -> bool:
        """更新数据"""
        # 从数据处理器获取新数据
        # 这里需要根据具体的数据源实现
        # 暂时返回False表示没有更多数据
        return True

    def _process_events(self):
        """处理事件队列"""
        while not self.event_queue.empty():
            event = self.event_queue.get()

            if event is None:
                break

            # 路由事件到对应的处理器
            if isinstance(event, BarEvent):
                self.strategy.on_bar(event)

            elif isinstance(event, OrderEvent):
                self._process_order_event(event)

            elif isinstance(event, FillEvent):
                self.portfolio.update_fill(event)

    def _process_order_event(self, event: OrderEvent):
        """处理订单事件"""
        self._stats["total_orders"] += 1

        # 执行订单
        fill_event = self.execution_handler.execute_order(event)

        if fill_event:
            self.event_queue.put(fill_event)
            self._stats["total_fills"] += 1
        else:
            self._stats["rejected_orders"] += 1
            logger.warning(f"订单被拒绝: {event.order_id}")


class SimpleDataHandler(DataHandler):
    """简单数据处理器实现"""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Args:
            data: {symbol: DataFrame} 格式的数据
        """
        self.data = data
        self.symbols = list(data.keys())
        self._iterators: Dict[str, Iterator] = {}
        self._current_bars: Dict[str, BarEvent] = {}
        self.reset()

    def reset(self):
        """重置数据处理器"""
        self._iterators = {}
        for symbol in self.symbols:
            self._iterators[symbol] = iter(self.data[symbol].to_dict("records"))
        self._current_bars = {}

    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取下一根K线"""
        try:
            bar_data = next(self._iterators[symbol])
            event = BarEvent(
                datetime=pd.to_datetime(bar_data["date"]) if "date" in bar_data else datetime.now(),
                symbol=symbol,
                open=bar_data["open"],
                high=bar_data["high"],
                low=bar_data["low"],
                close=bar_data["close"],
                volume=int(bar_data["volume"]),
            )
            self._current_bars[symbol] = event
            return event
        except (StopIteration, KeyError):
            return None

    def get_current_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取当前K线"""
        return self._current_bars.get(symbol)


# 需要添加导入
from collections.abc import Iterator

__all__ = [
    "EventType",
    "Event",
    "BarEvent",
    "OrderEvent",
    "FillEvent",
    "EventQueue",
    "Strategy",
    "DataHandler",
    "Portfolio",
    "ExecutionHandler",
    "BacktestEngine",
    "SimpleDataHandler",
]
