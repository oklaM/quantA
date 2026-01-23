"""
高级回测策略集合
包含多种实用的量化交易策略
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine.event_engine import BarEvent
from backtest.engine.strategy import Strategy
from utils.logging import get_logger

logger = get_logger(__name__)


class BollingerBandsStrategy(Strategy):
    """
    布林带策略

    逻辑：
    - 价格触及下轨时买入
    - 价格触及上轨时卖出
    """

    def __init__(
        self,
        symbol: str,
        period: int = 20,
        std_dev: float = 2.0,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            period: 均线周期
            std_dev: 标准差倍数
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.period = period
        self.std_dev = std_dev
        self.quantity = quantity

        # 价格历史
        self.price_history: List[float] = []
        self.position_side = None  # 'long' or None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.price_history.append(event.close)

        if len(self.price_history) < self.period:
            return

        # 计算布林带
        prices = pd.Series(self.price_history)
        middle = prices.rolling(window=self.period).iloc[-1]
        std = prices.rolling(window=self.period).std().iloc[-1]

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        # 买入信号：价格触及下轨
        if event.close <= lower and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            logger.info(
                f"布林带买入: 价格={event.close:.2f}, " f"下轨={lower:.2f}, 中轨={middle:.2f}"
            )

        # 卖出信号：价格触及上轨或回到中轨
        elif event.close >= upper and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                logger.info(
                    f"布林带卖出: 价格={event.close:.2f}, " f"上轨={upper:.2f}, 中轨={middle:.2f}"
                )


class MACDStrategy(Strategy):
    """
    MACD策略

    逻辑：
    - MACD金叉（快线上穿慢线）时买入
    - MACD死叉（快线下穿慢线）时卖出
    """

    def __init__(
        self,
        symbol: str,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.quantity = quantity

        # 价格历史
        self.price_history: List[float] = []
        self.position_side = None

        # MACD历史
        self.macd_history: List[float] = []
        self.signal_history: List[float] = []

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.price_history.append(event.close)

        if len(self.price_history) < self.slow_period + self.signal_period:
            return

        # 计算MACD
        prices = pd.Series(self.price_history)
        ema_fast = prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        self.macd_history.append(macd_line.iloc[-1])
        self.signal_history.append(signal_line.iloc[-1])

        if len(self.macd_history) < 2:
            return

        # 金叉：MACD上穿信号线
        prev_diff = self.macd_history[-2] - self.signal_history[-2]
        curr_diff = self.macd_history[-1] - self.signal_history[-1]

        if prev_diff <= 0 and curr_diff > 0 and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            logger.info(
                f"MACD金叉买入: MACD={self.macd_history[-1]:.2f}, "
                f"Signal={self.signal_history[-1]:.2f}"
            )

        # 死叉：MACD下穿信号线
        elif prev_diff >= 0 and curr_diff < 0 and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                logger.info(
                    f"MACD死叉卖出: MACD={self.macd_history[-1]:.2f}, "
                    f"Signal={self.signal_history[-1]:.2f}"
                )


class RSIStrategy(Strategy):
    """
    RSI策略

    逻辑：
    - RSI超卖（<30）时买入
    - RSI超买（>70）时卖出
    """

    def __init__(
        self,
        symbol: str,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            period: RSI周期
            oversold: 超卖阈值
            overbought: 超买阈值
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.quantity = quantity

        self.price_history: List[float] = []
        self.position_side = None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.price_history.append(event.close)

        if len(self.price_history) < self.period + 1:
            return

        # 计算RSI
        prices = pd.Series(self.price_history)
        delta = prices.diff()

        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        # 超卖买入
        if current_rsi < self.oversold and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            logger.info(f"RSI超卖买入: RSI={current_rsi:.2f}")

        # 超买卖出
        elif current_rsi > self.overbought and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                logger.info(f"RSI超买卖出: RSI={current_rsi:.2f}")


class BreakoutStrategy(Strategy):
    """
    突破策略

    逻辑：
    - 价格突破N日最高价时买入
    - 价格跌破N日最低价时卖出
    """

    def __init__(
        self,
        symbol: str,
        period: int = 20,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            period: 突破周期（天数）
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.period = period
        self.quantity = quantity

        self.price_history: List[Dict] = []  # 存储 {'high': x, 'low': y}
        self.position_side = None
        self.entry_price = None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.price_history.append({"high": event.high, "low": event.low, "close": event.close})

        if len(self.price_history) < self.period:
            return

        # 计算N日最高价和最低价
        recent_bars = self.price_history[-self.period :]
        highest_high = max(bar["high"] for bar in recent_bars)
        lowest_low = min(bar["low"] for bar in recent_bars)

        # 向上突破
        if event.close > highest_high and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            self.entry_price = event.close
            logger.info(
                f"突破买入: 价格={event.close:.2f}, " f"{self.period}日最高={highest_high:.2f}"
            )

        # 向下突破（止损）
        elif event.close < lowest_low and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                self.entry_price = None
                logger.info(
                    f"突破卖出: 价格={event.close:.2f}, " f"{self.period}日最低={lowest_low:.2f}"
                )

        # 移动止损（最高点回撤10%）
        elif self.position_side == "long" and self.entry_price:
            if event.close < self.entry_price * 0.9:
                position = self.get_position(self.symbol)
                if position["quantity"] > 0:
                    self.sell(self.symbol, position["quantity"], order_type="market")
                    self.position_side = None
                    self.entry_price = None
                    logger.info(
                        f"移动止损: 价格={event.close:.2f}, " f"入场价={self.entry_price:.2f}"
                    )


class DualThrustStrategy(Strategy):
    """
    Dual Thrust策略

    逻辑：
    - 计算上下轨
    - 突破上轨买入，跌破下轨卖出
    """

    def __init__(
        self,
        symbol: str,
        period: int = 5,
        k1: float = 0.5,
        k2: float = 0.5,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            period: 计算周期
            k1: 上轨系数
            k2: 下轨系数
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.period = period
        self.k1 = k1
        self.k2 = k2
        self.quantity = quantity

        self.bar_history: List[BarEvent] = []
        self.position_side = None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.bar_history.append(event)

        if len(self.bar_history) < self.period + 1:
            return

        # 计算Dual Thrust范围
        recent_bars = self.bar_history[-self.period - 1 : -1]

        highs = [bar.high for bar in recent_bars]
        lows = [bar.low for bar in recent_bars]
        closes = [bar.close for bar in recent_bars]

        hh = max(highs)
        ll = min(lows)
        # hc和lc的计算：取前一根bar的最高/最低价与收盘价的比较
        hc = max(highs[-2] if len(highs) > 1 else highs[-1], closes[-2] if len(closes) > 1 else closes[-1])
        lc = min(lows[-2] if len(lows) > 1 else lows[-1], closes[-2] if len(closes) > 1 else closes[-1])

        # 计算上下轨
        range_value = hh - ll
        upper = event.open + self.k1 * range_value
        lower = event.open - self.k2 * range_value

        # 突破上轨买入
        if event.close > upper and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            logger.info(f"Dual Thrust买入: 价格={event.close:.2f}, " f"上轨={upper:.2f}")

        # 跌破下轨卖出
        elif event.close < lower and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                logger.info(f"Dual Thrust卖出: 价格={event.close:.2f}, " f"下轨={lower:.2f}")


class GridTradingStrategy(Strategy):
    """
    网格交易策略

    逻辑：
    - 在价格区间内设置网格
    - 价格下跌时分批买入
    - 价格上涨时分批卖出
    """

    def __init__(
        self,
        symbol: str,
        base_price: float,
        grid_count: int = 10,
        grid_spacing: float = 0.01,  # 1%
        grid_quantity: int = 100,
    ):
        """
        Args:
            symbol: 股票代码
            base_price: 基准价格
            grid_count: 网格数量
            grid_spacing: 网格间距（百分比）
            grid_quantity: 每格交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.base_price = base_price
        self.grid_count = grid_count
        self.grid_spacing = grid_spacing
        self.grid_quantity = grid_quantity

        # 计算网格价位
        self.buy_grids = []
        self.sell_grids = []

        for i in range(grid_count):
            price = base_price * (1 - i * grid_spacing)
            self.buy_grids.append(price)

        for i in range(1, grid_count + 1):
            price = base_price * (1 + i * grid_spacing)
            self.sell_grids.append(price)

        # 已执行的网格
        self.executed_grids: List[float] = []

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        current_price = event.close

        # 检查买入网格
        for buy_price in self.buy_grids:
            if current_price <= buy_price and buy_price not in self.executed_grids:
                # 检查资金
                cash = self.get_cash()
                required = buy_price * self.grid_quantity
                if cash >= required:
                    self.buy(self.symbol, self.grid_quantity, price=buy_price)
                    self.executed_grids.append(buy_price)
                    logger.info(f"网格买入: 价格={current_price:.2f}, " f"网格价={buy_price:.2f}")

        # 检查卖出网格
        position = self.get_position(self.symbol)
        current_qty = position.get("quantity", 0)

        for sell_price in self.sell_grids:
            if current_price >= sell_price and current_qty >= self.grid_quantity:
                # 检查是否在该价位买入过
                bought_price = sell_price / (1 + self.grid_spacing)
                if bought_price in self.executed_grids:
                    self.sell(self.symbol, self.grid_quantity, price=sell_price)
                    self.executed_grids.remove(bought_price)
                    logger.info(f"网格卖出: 价格={current_price:.2f}, " f"网格价={sell_price:.2f}")
                    break


class MomentumStrategy(Strategy):
    """
    动量策略

    逻辑：
    - 计算价格动量
    - 动量为正且强劲时买入
    - 动量转负时卖出
    """

    def __init__(
        self,
        symbol: str,
        lookback: int = 20,
        momentum_threshold: float = 0.02,  # 2%
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            lookback: 回看周期
            momentum_threshold: 动量阈值
            quantity: 交易数量
        """
        super().__init__()
        self.symbol = symbol
        self.lookback = lookback
        self.momentum_threshold = momentum_threshold
        self.quantity = quantity

        self.price_history: List[float] = []
        self.position_side = None

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        self.price_history.append(event.close)

        if len(self.price_history) < self.lookback + 1:
            return

        # 计算动量
        momentum = (event.close - self.price_history[-self.lookback - 1]) / self.price_history[
            -self.lookback - 1
        ]

        # 强劲动量买入
        if momentum > self.momentum_threshold and self.position_side is None:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.position_side = "long"
            logger.info(f"动量买入: 动量={momentum*100:.2f}%")

        # 动量转负卖出
        elif momentum < -self.momentum_threshold and self.position_side == "long":
            position = self.get_position(self.symbol)
            if position["quantity"] > 0:
                self.sell(self.symbol, position["quantity"], order_type="market")
                self.position_side = None
                logger.info(f"动量卖出: 动量={momentum*100:.2f}%")


class BuyAndHoldStrategy(Strategy):
    """
    买入持有策略

    逻辑：
    - 在第一个Bar买入并持有
    - 用于作为基准比较
    """

    def __init__(
        self,
        symbol: str,
        quantity: int = 1000,
    ):
        """
        Args:
            symbol: 股票代码
            quantity: 买入数量
        """
        super().__init__()
        self.symbol = symbol
        self.quantity = quantity
        self.bought = False

    def on_bar(self, event: BarEvent):
        """处理K线事件"""
        if event.symbol != self.symbol:
            return

        # 第一次买入
        if not self.bought:
            self.buy(self.symbol, self.quantity, order_type="market")
            self.bought = True
            logger.info(f"买入持有策略: 买入{self.quantity}股{self.symbol}")


__all__ = [
    "BuyAndHoldStrategy",
    "BollingerBandsStrategy",
    "MACDStrategy",
    "RSIStrategy",
    "BreakoutStrategy",
    "DualThrustStrategy",
    "GridTradingStrategy",
    "MomentumStrategy",
]
