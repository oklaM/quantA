"""
投资组合管理
实现持仓管理、资金管理、绩效计算
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.engine.event_engine import Event, FillEvent
from utils.logging import get_logger

logger = get_logger(__name__)


class PositionSide(Enum):
    """持仓方向"""

    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    quantity: int = 0  # 持仓数量
    avg_price: float = 0.0  # 平均成本
    side: PositionSide = PositionSide.LONG
    available_qty: int = 0  # 可用数量（考虑T+1）
    today_bought: int = 0  # 今日买入（T+1冻结）

    @property
    def market_value(self) -> float:
        """市值（需要更新价格）"""
        # 这个属性需要在外部更新时设置
        return 0.0

    @property
    def pnl(self) -> float:
        """浮动盈亏"""
        return 0.0

    @property
    def pnl_pct(self) -> float:
        """盈亏比例"""
        return 0.0


@dataclass
class Account:
    """账户信息"""

    initial_cash: float = 1000000.0  # 初始资金
    cash: float = 1000000.0  # 可用现金
    total_value: float = 1000000.0  # 总资产
    positions: Dict[str, Position] = field(default_factory=dict)

    # 性能指标
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    # 历史记录
    equity_curve: List[Dict] = field(default_factory=list)

    def add_to_equity_curve(self, timestamp: datetime, total_value: float):
        """添加到净值曲线"""
        self.equity_curve.append(
            {
                "datetime": timestamp,
                "total_value": total_value,
                "cash": self.cash,
                "return": (total_value - self.initial_cash) / self.initial_cash,
            }
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)

    def update_position(self, symbol: str, quantity: int, price: float, side: str):
        """更新持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        position = self.positions[symbol]

        if side == "buy":
            # 买入
            total_cost = position.avg_price * position.quantity + price * quantity
            total_qty = position.quantity + quantity
            position.avg_price = total_cost / total_qty if total_qty > 0 else 0
            position.quantity = total_qty
            position.today_bought += quantity
        else:
            # 卖出
            position.quantity -= quantity

        position.available_qty = position.quantity

        # 清空持仓
        if position.quantity == 0:
            del self.positions[symbol]

    def update_total_value(self, prices: Dict[str, float]):
        """更新总资产"""
        position_value = 0.0

        for symbol, position in self.positions.items():
            price = prices.get(symbol, 0)
            if price > 0:
                position_value += position.quantity * price

        self.total_value = self.cash + position_value
        self.total_return = (self.total_value - self.initial_cash) / self.initial_cash


class Portfolio:
    """
    投资组合管理
    """

    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
    ):
        """
        Args:
            initial_cash: 初始资金
            commission_rate: 佣金费率
            min_commission: 最低佣金
            stamp_duty_rate: 印花税率（仅卖出）
        """
        self.account = Account(initial_cash=initial_cash)
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate

        # 订单ID计数
        self._order_id = 0
        self._order_history: List[Dict] = []

        # 当前价格
        self._current_prices: Dict[str, float] = {}

    def update_time(self, event: Event):
        """更新时间"""
        # 可以在这里更新时间相关的状态
        pass

    def update_fill(self, event: FillEvent):
        """处理成交事件"""
        self._order_id += 1

        # 计算费用
        commission = self._calculate_commission(event.quantity, event.price)
        stamp_duty = 0.0

        if event.side == "sell":
            stamp_duty = event.quantity * event.price * self.stamp_duty_rate

        total_cost = event.quantity * event.price + commission + stamp_duty

        # 更新账户
        if event.side == "buy":
            # 买入：减少现金
            if total_cost > self.account.cash:
                logger.error(f"资金不足: 需要{total_cost:.2f}, 可用{self.account.cash:.2f}")
                return

            self.account.cash -= total_cost
            self.account.update_position(event.symbol, event.quantity, event.price, "buy")

        else:
            # 卖出：增加现金
            self.account.cash += event.quantity * event.price - commission - stamp_duty
            self.account.update_position(event.symbol, event.quantity, event.price, "sell")

        # 更新统计
        self.account.total_trades += 1

        # 记录订单历史
        self._order_history.append(
            {
                "order_id": self._order_id,
                "datetime": event.datetime,
                "symbol": event.symbol,
                "side": event.side,
                "quantity": event.quantity,
                "price": event.price,
                "commission": commission,
                "stamp_duty": stamp_duty,
                "total_cost": total_cost,
            }
        )

        logger.debug(
            f"成交: {event.side} {event.symbol} "
            f"{event.quantity}股 @{event.price:.2f} "
            f"费用:{commission:.2f}"
        )

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """计算佣金"""
        commission = quantity * price * self.commission_rate
        return max(commission, self.min_commission)

    def update_current_price(self, symbol: str, price: float):
        """更新当前价格"""
        self._current_prices[symbol] = price

    def update_total_value(self):
        """更新总资产"""
        self.account.update_total_value(self._current_prices)

    def add_to_equity_curve(self, timestamp: datetime):
        """添加净值记录"""
        self.account.add_to_equity_curve(timestamp, self.account.total_value)

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """获取持仓信息"""
        position = self.account.get_position(symbol)

        current_price = self._current_prices.get(symbol, 0.0)

        if position is None:
            return {
                "symbol": symbol,
                "quantity": 0,
                "available_qty": 0,
                "avg_price": 0.0,
                "current_price": current_price,
                "market_value": 0.0,
                "cost_value": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "today_bought": 0,
            }

        market_value = position.quantity * current_price
        cost_value = position.quantity * position.avg_price
        pnl = market_value - cost_value
        pnl_pct = (pnl / cost_value * 100) if cost_value > 0 else 0.0

        return {
            "symbol": symbol,
            "quantity": position.quantity,
            "available_qty": position.available_qty,
            "avg_price": position.avg_price,
            "current_price": current_price,
            "market_value": market_value,
            "cost_value": cost_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "today_bought": position.today_bought,
        }

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """获取所有持仓"""
        positions = []
        for symbol in self.account.positions.keys():
            positions.append(self.get_position(symbol))
        return positions

    def get_account_info(self) -> Dict[str, Any]:
        """获取账户信息"""
        return {
            "initial_cash": self.account.initial_cash,
            "cash": self.account.cash,
            "total_value": self.account.total_value,
            "position_value": self.account.total_value - self.account.cash,
            "total_return": self.account.total_return,
            "total_return_pct": self.account.total_return * 100,
            "total_trades": self.account.total_trades,
            "num_positions": len(self.account.positions),
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """获取净值曲线"""
        if not self.account.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame(self.account.equity_curve)

    def reset(self):
        """重置投资组合"""
        self.account = Account(initial_cash=self.account.initial_cash)
        self._order_id = 0
        self._order_history = []
        self._current_prices = {}


__all__ = [
    "Position",
    "PositionSide",
    "Account",
    "Portfolio",
]
