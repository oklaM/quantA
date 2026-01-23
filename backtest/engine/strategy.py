"""
策略基类和常用策略
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.engine.data_handler import DataHandler
from backtest.engine.event_engine import (
    BarEvent,
    EventQueue,
    OrderEvent,
)
from backtest.engine.event_engine import Strategy as BaseStrategy
from backtest.engine.portfolio import Portfolio
from utils.logging import get_logger

logger = get_logger(__name__)


class Strategy(BaseStrategy):
    """
    策略基类
    提供常用的交易方法
    """

    def __init__(self):
        super().__init__()
        self._order_id = 0

    def generate_order_id(self) -> str:
        """生成订单ID"""
        self._order_id += 1
        return f"order_{self._order_id}"

    def buy(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "limit",
    ) -> Optional[str]:
        """
        发送买入订单

        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格（None表示市价单）
            order_type: 订单类型

        Returns:
            订单ID
        """
        if self.event_queue is None:
            logger.error("事件队列未设置")
            return None

        order_id = self.generate_order_id()

        order = OrderEvent(
            datetime=None,  # 会在放入队列时更新
            order_id=order_id,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=price,
            order_type=order_type,
        )

        self.event_queue.put(order)
        logger.debug(f"发送买入订单: {symbol} {quantity}股 @{price}")

        return order_id

    def sell(
        self,
        symbol: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "limit",
    ) -> Optional[str]:
        """
        发送卖出订单

        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格（None表示市价单）
            order_type: 订单类型

        Returns:
            订单ID
        """
        if self.event_queue is None:
            logger.error("事件队列未设置")
            return None

        order_id = self.generate_order_id()

        order = OrderEvent(
            datetime=None,
            order_id=order_id,
            symbol=symbol,
            side="sell",
            quantity=quantity,
            price=price,
            order_type=order_type,
        )

        self.event_queue.put(order)
        logger.debug(f"发送卖出订单: {symbol} {quantity}股 @{price}")

        return order_id

    def close_position(self, symbol: str, price: Optional[float] = None):
        """平仓"""
        if self.portfolio is None:
            return

        position = self.portfolio.get_position(symbol)
        if position and position["quantity"] > 0:
            self.sell(symbol, position["quantity"], price)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """获取持仓"""
        if self.portfolio is None:
            return {"quantity": 0}
        return self.portfolio.get_position(symbol)

    def get_cash(self) -> float:
        """获取可用资金"""
        if self.portfolio is None:
            return 0.0
        return self.portfolio.get_account_info()["cash"]

    def get_total_value(self) -> float:
        """获取总资产"""
        if self.portfolio is None:
            return 0.0
        return self.portfolio.get_account_info()["total_value"]


class BuyAndHoldStrategy(Strategy):
    """买入持有策略"""

    def __init__(self, symbol: str, quantity: int = 1000):
        """
        Args:
            symbol: 股票代码
            quantity: 买入数量
        """
        super().__init__()
        self.symbol = symbol
        self.quantity = quantity
        self._bought = False

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        # 第一根K线买入
        if not self._bought and event.symbol == self.symbol:
            self.buy(self.symbol, self.quantity, order_type="market")
            self._bought = True


class MovingAverageCrossStrategy(Strategy):
    """
    双均线交叉策略
    金叉买入，死叉卖出
    """

    def __init__(
        self,
        symbol: str,
        fast_period: int = 5,
        slow_period: int = 20,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            fast_period: 快线周期
            slow_period: 慢线周期
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity

        # 历史价格
        self.price_history: List[float] = []

        # 当前持仓方向
        self.position_side = None  # 'long' or None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        # 添加价格历史
        self.price_history.append(event.close)

        # 数据不足
        if len(self.price_history) < self.slow_period:
            return

        # 计算均线
        prices = pd.Series(self.price_history)
        fast_ma = prices.rolling(window=self.fast_period).mean().iloc[-1]
        slow_ma = prices.rolling(window=self.slow_period).mean().iloc[-1]

        # 计算前一根K线的均线
        if len(self.price_history) > self.slow_period:
            prev_fast_ma = prices.rolling(window=self.fast_period).mean().iloc[-2]
            prev_slow_ma = prices.rolling(window=self.slow_period).mean().iloc[-2]
        else:
            prev_fast_ma = fast_ma
            prev_slow_ma = slow_ma

        # 金叉：快线上穿慢线
        if (prev_fast_ma <= prev_slow_ma) and (fast_ma > slow_ma):
            if self.position_side is None:
                self.buy(self.symbol, self.quantity, order_type="market")
                self.position_side = "long"
                logger.info(
                    f"金叉买入: MA{self.fast_period}={fast_ma:.2f} MA{self.slow_period}={slow_ma:.2f}"
                )

        # 死叉：快线下穿慢线
        elif (prev_fast_ma >= prev_slow_ma) and (fast_ma < slow_ma):
            if self.position_side == "long":
                position = self.get_position(self.symbol)
                if position["quantity"] > 0:
                    self.sell(self.symbol, position["quantity"], order_type="market")
                    self.position_side = None
                    logger.info(
                        f"死叉卖出: MA{self.fast_period}={fast_ma:.2f} MA{self.slow_period}={slow_ma:.2f}"
                    )


class MeanReversionStrategy(Strategy):
    """
    均值回归策略
    价格偏离均值过多时反向交易
    """

    def __init__(
        self,
        symbol: str,
        period: int = 20,
        std_threshold: float = 2.0,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            period: 均值周期
            std_threshold: 标准差阈值
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.period = period
        self.std_threshold = std_threshold
        self.quantity = quantity

        self.price_history: List[float] = []
        self.position_side = None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.price_history.append(event.close)

        if len(self.price_history) < self.period:
            return

        # 计算均值和标准差
        prices = pd.Series(self.price_history)
        mean = prices.rolling(window=self.period).iloc[-1]
        std = prices.rolling(window=self.period).std().iloc[-1]

        # 计算Z-score
        zscore = (event.close - mean) / std if std > 0 else 0

        # 价格过低（买入信号）
        if zscore < -self.std_threshold and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            logger.info(f"均值回归买入: 价格={event.close:.2f}, Z-score={zscore:.2f}")

        # 价格过高（卖出信号）
        elif zscore > self.std_threshold and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                logger.info(f"均值回归卖出: 价格={event.close:.2f}, Z-score={zscore:.2f}")


__all__ = [
    "Strategy",
    "BuyAndHoldStrategy",
    "MovingAverageCrossStrategy",
    "MeanReversionStrategy",
]
