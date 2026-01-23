"""
订单执行处理器
处理订单的执行逻辑，包括滑点、成交等
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from backtest.engine.a_share_rules import AShareRulesEngine, Order, Position
from backtest.engine.event_engine import (
    Event,
    EventType,
    FillEvent,
    OrderEvent,
)
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SlippageModel:
    """滑点模型"""

    enabled: bool = True
    mode: str = "linear"  # linear, percentage, none
    rate: float = 0.0001  # 滑点率
    max_slippage: float = 0.001  # 最大滑点


@dataclass
class TransactionCost:
    """交易成本模型"""

    commission_rate: float = 0.0003  # 佣金率
    min_commission: float = 5.0  # 最低佣金
    stamp_duty_rate: float = 0.001  # 印花税率（卖出）


class ExecutionHandler:
    """
    订单执行处理器
    负责验证订单、计算成交价格和滑点
    """

    def __init__(
        self,
        rules_engine: Optional[AShareRulesEngine] = None,
        slippage_model: Optional[SlippageModel] = None,
        transaction_cost: Optional[TransactionCost] = None,
        fill_at_close: bool = False,  # 是否以收盘价成交
    ):
        """
        Args:
            rules_engine: A股规则引擎
            slippage_model: 滑点模型
            transaction_cost: 交易成本
            fill_at_close: 是否以收盘价成交
        """
        self.rules_engine = rules_engine or AShareRulesEngine()
        self.slippage_model = slippage_model or SlippageModel()
        self.transaction_cost = transaction_cost or TransactionCost()
        self.fill_at_close = fill_at_close

        # 当前市场数据
        self._current_bars: Dict[str, Dict] = {}

        # 持仓数据
        self._positions: Dict[str, Position] = {}

        # 今日买入记录
        self._today_bought: Dict[str, int] = {}

    def update_bar(self, bar_data: Dict):
        """更新当前K线数据"""
        symbol = bar_data["symbol"]
        self._current_bars[symbol] = bar_data

    def update_position(self, position: Position):
        """更新持仓"""
        self._positions[position.symbol] = position

    def update_today_bought(self, symbol: str, quantity: int):
        """更新今日买入数量"""
        self._today_bought[symbol] = self._today_bought.get(symbol, 0) + quantity

    def reset_today_bought(self):
        """重置今日买入记录（新的一天）"""
        self._today_bought.clear()

    def execute_order(self, event: OrderEvent) -> Optional[FillEvent]:
        """
        执行订单

        Args:
            event: 订单事件

        Returns:
            成交事件，如果订单被拒绝则返回None
        """
        symbol = event.symbol

        # 获取当前市场数据
        if symbol not in self._current_bars:
            logger.warning(f"没有{symbol}的市场数据")
            return None

        bar_data = self._current_bars[symbol]

        # 创建订单对象
        order = Order(
            symbol=event.symbol,
            side=event.side,
            quantity=event.quantity,
            price=event.price,
            order_type=event.order_type,
            datetime=event.datetime,
        )

        # 获取昨收价（用于涨跌停检查）
        # 这里简化处理，使用前一根K线的收盘价
        last_close = bar_data.get("prev_close", bar_data["close"])

        # 获取当前持仓
        position = self._positions.get(symbol)
        today_bought = self._today_bought.get(symbol, 0)

        # 验证订单
        passed, reason, msg = self.rules_engine.validate_order(
            order=order,
            position=position,
            last_close=last_close,
            today_bought=today_bought,
            check_time=False,  # 回测时不需要检查时间
        )

        if not passed:
            logger.debug(f"订单被拒绝: {msg}")
            return None

        # 计算成交价格
        fill_price = self._calculate_fill_price(order, bar_data)

        # 计算成交数量
        fill_quantity = event.quantity

        # 更新持仓记录
        if event.side == "buy":
            self.update_today_bought(symbol, fill_quantity)

        # 创建成交事件
        commission = self._calculate_commission(fill_quantity, fill_price)

        fill_event = FillEvent(
            datetime=event.datetime,
            order_id=event.order_id,
            symbol=symbol,
            side=event.side,
            quantity=fill_quantity,
            price=fill_price,
            commission=commission,
        )

        logger.debug(f"订单成交: {event.side} {symbol} " f"{fill_quantity}股 @{fill_price:.2f}")

        return fill_event

    def _calculate_fill_price(
        self,
        order: Order,
        bar_data: Dict,
    ) -> float:
        """
        计算成交价格（考虑滑点）

        Args:
            order: 订单
            bar_data: K线数据

        Returns:
            成交价格
        """
        # 确定基础价格
        if self.fill_at_close:
            base_price = bar_data["close"]
        elif order.order_type == "market" or order.price is None:
            # 市价单：使用当前价
            base_price = bar_data.get("current_price", bar_data["close"])
        else:
            # 限价单：使用订单价格
            base_price = order.price

        # 应用滑点
        if self.slippage_model.enabled:
            fill_price = self._apply_slippage(
                base_price=base_price,
                side=order.side,
                bar_data=bar_data,
            )
        else:
            fill_price = base_price

        # 四舍五入到0.01
        return round(fill_price, 2)

    def _apply_slippage(
        self,
        base_price: float,
        side: str,
        bar_data: Dict,
    ) -> float:
        """
        应用滑点

        Args:
            base_price: 基础价格
            side: 买卖方向
            bar_data: K线数据

        Returns:
            考虑滑点后的价格
        """
        if self.slippage_model.mode == "none":
            return base_price

        # 计算滑点
        if self.slippage_model.mode == "percentage":
            slippage = base_price * self.slippage_model.rate
            # 添加随机性
            slippage *= random.uniform(0.5, 1.5)

        elif self.slippage_model.mode == "linear":
            # 根据成交额计算滑点
            volume = bar_data.get("volume", 0)
            impact = min(volume * self.slippage_model.rate, self.slippage_model.max_slippage)
            slippage = base_price * impact

        else:
            slippage = 0.0

        # 买入时价格上涨，卖出时价格下跌
        if side == "buy":
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage

        # 确保价格不超出K线范围
        high = bar_data.get("high", base_price)
        low = bar_data.get("low", base_price)

        return max(low, min(fill_price, high))

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """计算佣金"""
        commission = quantity * price * self.transaction_cost.commission_rate
        return max(commission, self.transaction_cost.min_commission)

    def calculate_stamp_duty(self, quantity: int, price: float, side: str) -> float:
        """计算印花税"""
        if side == "sell":
            return quantity * price * self.transaction_cost.stamp_duty_rate
        return 0.0


class SimulationExecutionHandler(ExecutionHandler):
    """模拟执行处理器（简单版本）"""

    def __init__(
        self,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0001,
    ):
        """
        Args:
            commission_rate: 佣金率
            slippage_rate: 滑点率
        """
        slippage_model = SlippageModel(
            enabled=True,
            mode="percentage",
            rate=slippage_rate,
        )

        transaction_cost = TransactionCost(
            commission_rate=commission_rate,
        )

        super().__init__(
            slippage_model=slippage_model,
            transaction_cost=transaction_cost,
            fill_at_close=True,  # 简化处理，用收盘价成交
        )


__all__ = [
    "SlippageModel",
    "TransactionCost",
    "ExecutionHandler",
    "SimulationExecutionHandler",
]
