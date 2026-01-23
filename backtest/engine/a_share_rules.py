"""
A股交易规则引擎
实现A股市场的特殊交易规则：
- T+1交易规则
- 涨跌停限制
- 交易时间
- 最小申报单位
"""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
from typing import List, Optional, Tuple

import pandas as pd

from config.settings import market as market_config
from config.symbols import MarketType, get_market_type
from utils.logging import get_logger
from utils.time_utils import get_trading_sessions, is_trading_day, is_trading_time

logger = get_logger(__name__)


class OrderRejectReason(Enum):
    """订单拒绝原因"""

    NOT_TRADING_DAY = "非交易日"
    NOT_TRADING_TIME = "非交易时间"
    T_PLUS_ONE = "T+1限制：当天买入不可当天卖出"
    LIMIT_UP = "涨停价格限制"
    LIMIT_DOWN = "跌停价格限制"
    MIN_ORDER_SIZE = "小于最小申报单位(100股)"
    PRICE_PRECISION = "价格精度不符合要求"
    INSUFFICIENT_POSITION = "持仓不足"
    INVALID_PRICE = "价格无效"
    MARKET_CLOSED = "市场已收盘"


@dataclass
class Order:
    """订单数据类"""

    symbol: str
    side: str  # 'buy' 或 'sell'
    quantity: int  # 数量（股）
    price: Optional[float] = None  # 价格（None表示市价单）
    order_type: str = "limit"  # 'limit' 或 'market'
    datetime: Optional[datetime] = None

    def __post_init__(self):
        if self.datetime is None:
            self.datetime = datetime.now()


@dataclass
class Position:
    """持仓数据类"""

    symbol: str
    quantity: int  # 持仓数量
    available_qty: int  # 可用数量（扣除T+1冻结）
    avg_price: float  # 平均成本
    today_bought: int = 0  # 今日买入数量（T+1冻结）


class AShareRulesEngine:
    """A股交易规则引擎"""

    def __init__(self):
        self.min_order_size = market_config.MIN_ORDER_SIZE  # 100股
        self.t_plus_one = market_config.T_PLUS_ONE

        # 涨跌停限制
        self.main_board_limit = market_config.MAIN_BOARD_LIMIT  # 10%
        self.sme_board_limit = market_config.SME_BOARD_LIMIT  # 20%

    # ====================
    # T+1规则检查
    # ====================

    def check_t_plus_one(
        self,
        order: Order,
        position: Optional[Position],
        today_bought: int,
    ) -> Tuple[bool, Optional[OrderRejectReason]]:
        """
        检查T+1规则

        Args:
            order: 订单
            position: 当前持仓
            today_bought: 今日买入数量

        Returns:
            (是否通过, 拒绝原因)
        """
        if not self.t_plus_one:
            return True, None

        # 卖单需要检查T+1
        if order.side == "sell":
            if position is None or position.quantity < order.quantity:
                return False, OrderRejectReason.INSUFFICIENT_POSITION

            # 可用数量 = 总持仓 - 今日买入
            available = position.quantity - today_bought

            if available < order.quantity:
                logger.debug(
                    f"T+1限制: {order.symbol} 可用{available}股，" f"今日买入{today_bought}股被冻结"
                )
                return False, OrderRejectReason.T_PLUS_ONE

        return True, None

    # ====================
    # 涨跌停检查
    # ====================

    def check_limit_price(
        self,
        order: Order,
        last_close: float,
    ) -> Tuple[bool, Optional[OrderRejectReason], Optional[float]]:
        """
        检查涨跌停价格限制

        Args:
            order: 订单
            last_close: 昨收价

        Returns:
            (是否通过, 拒绝原因, 涨停价/跌停价)
        """
        # 市价单跳过价格检查
        if order.order_type == "market":
            return True, None, None

        # 获取涨跌停限制
        limit_percent = self._get_limit_percent(order.symbol)

        # 计算涨跌停价格
        limit_up = last_close * (1 + limit_percent)
        limit_down = last_close * (1 - limit_percent)

        # A股价格精度：0.01元
        limit_up = round(limit_up, 2)
        limit_down = round(limit_down, 2)

        # 检查买入价格不超过涨停价
        if order.side == "buy":
            if order.price > limit_up:
                logger.debug(f"买入价格{order.price}超过涨停价{limit_up}")
                return False, OrderRejectReason.LIMIT_UP, limit_up

        # 检查卖出价格不低于跌停价
        elif order.side == "sell":
            if order.price < limit_down:
                logger.debug(f"卖出价格{order.price}低于跌停价{limit_down}")
                return False, OrderRejectReason.LIMIT_DOWN, limit_down

        return True, None, None

    def _get_limit_percent(self, symbol: str) -> float:
        """获取涨跌停限制比例"""
        market = get_market_type(symbol)

        if market in [MarketType.ChiNext, MarketType.STAR]:
            return self.sme_board_limit  # 20%
        else:
            return self.main_board_limit  # 10%

    def get_limit_prices(
        self,
        symbol: str,
        last_close: float,
    ) -> Tuple[float, float]:
        """
        获取涨停价和跌停价

        Args:
            symbol: 股票代码
            last_close: 昨收价

        Returns:
            (涨停价, 跌停价)
        """
        limit_percent = self._get_limit_percent(symbol)

        limit_up = round(last_close * (1 + limit_percent), 2)
        limit_down = round(last_close * (1 - limit_percent), 2)

        return limit_up, limit_down

    # ====================
    # 最小申报单位检查
    # ====================

    def check_min_order_size(self, order: Order) -> Tuple[bool, Optional[OrderRejectReason]]:
        """
        检查最小申报单位

        Args:
            order: 订单

        Returns:
            (是否通过, 拒绝原因)
        """
        if order.quantity < self.min_order_size:
            logger.debug(f"订单数量{order.quantity}小于最小单位{self.min_order_size}")
            return False, OrderRejectReason.MIN_ORDER_SIZE

        # 必须是100的整数倍
        if order.quantity % self.min_order_size != 0:
            logger.debug(f"订单数量{order.quantity}必须是100的整数倍")
            return False, OrderRejectReason.MIN_ORDER_SIZE

        return True, None

    # ====================
    # 价格精度检查
    # ====================

    def check_price_precision(self, order: Order) -> Tuple[bool, Optional[OrderRejectReason]]:
        """
        检查价格精度（A股为0.01元）

        Args:
            order: 订单

        Returns:
            (是否通过, 拒绝原因)
        """
        if order.price is not None:
            # 检查是否为0.01的倍数
            if round(order.price, 2) != order.price:
                logger.debug(f"价格{order.price}精度不符合要求（应为0.01的倍数）")
                return False, OrderRejectReason.PRICE_PRECISION

        return True, None

    # ====================
    # 交易时间检查
    # ====================

    def check_trading_time(
        self,
        order: Order,
    ) -> Tuple[bool, Optional[OrderRejectReason]]:
        """
        检查交易时间

        Args:
            order: 订单

        Returns:
            (是否通过, 拒绝原因)
        """
        dt = order.datetime

        # 检查是否为交易日
        if not is_trading_day(dt):
            logger.debug(f"{dt.date()} 不是交易日")
            return False, OrderRejectReason.NOT_TRADING_DAY

        # 检查是否在交易时间
        if not is_trading_time(dt):
            logger.debug(f"{dt.time()} 不在交易时间")
            return False, OrderRejectReason.NOT_TRADING_TIME

        return True, None

    # ====================
    # 综合订单验证
    # ====================

    def validate_order(
        self,
        order: Order,
        position: Optional[Position] = None,
        last_close: Optional[float] = None,
        today_bought: int = 0,
        check_time: bool = True,
    ) -> Tuple[bool, Optional[OrderRejectReason], str]:
        """
        综合验证订单

        Args:
            order: 订单
            position: 持仓
            last_close: 昨收价
            today_bought: 今日买入数量
            check_time: 是否检查交易时间

        Returns:
            (是否通过, 拒绝原因, 详细信息)
        """
        # 1. 检查交易时间
        if check_time:
            passed, reason = self.check_trading_time(order)
            if not passed:
                return False, reason, f"非交易时间: {order.datetime}"

        # 2. 检查最小申报单位
        passed, reason = self.check_min_order_size(order)
        if not passed:
            return False, reason, f"订单数量{order.quantity}不符合要求"

        # 3. 检查价格精度
        passed, reason = self.check_price_precision(order)
        if not passed:
            return False, reason, f"价格{order.price}精度不符合要求"

        # 4. 检查涨跌停
        if last_close is not None and order.price is not None:
            passed, reason, limit_price = self.check_limit_price(order, last_close)
            if not passed:
                if reason == OrderRejectReason.LIMIT_UP:
                    return False, reason, f"买入价{order.price}超过涨停价{limit_price}"
                else:
                    return False, reason, f"卖出价{order.price}低于跌停价{limit_price}"

        # 5. 检查T+1
        if position is not None:
            passed, reason = self.check_t_plus_one(order, position, today_bought)
            if not passed:
                return False, reason, f"T+1限制: 今日买入{today_bought}股被冻结"

        return True, None, "订单验证通过"

    # ====================
    # 订单价格调整
    # ====================

    def adjust_order_price(
        self,
        order: Order,
        last_close: float,
        current_price: Optional[float] = None,
    ) -> Order:
        """
        调整订单价格以满足规则

        Args:
            order: 原始订单
            last_close: 昨收价
            current_price: 当前价（市价单时使用）

        Returns:
            调整后的订单
        """
        adjusted_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.price,
            order_type=order.order_type,
            datetime=order.datetime,
        )

        # 市价单：使用当前价或涨跌停价
        if adjusted_order.price is None:
            if current_price is not None:
                adjusted_order.price = current_price
            else:
                # 使用昨收价
                adjusted_order.price = last_close

        # 确保价格在涨跌停范围内
        limit_up, limit_down = self.get_limit_prices(order.symbol, last_close)

        if adjusted_order.side == "buy":
            # 买入不超过涨停价
            adjusted_order.price = min(adjusted_order.price, limit_up)
        else:
            # 卖出不低於跌停价
            adjusted_order.price = max(adjusted_order.price, limit_down)

        # 四舍五入到0.01
        adjusted_order.price = round(adjusted_order.price, 2)

        return adjusted_order

    # ====================
    # 实用方法
    # ====================

    def can_sell_today(self, symbol: str, position: Position) -> int:
        """
        计算今日可卖出数量

        Args:
            symbol: 股票代码
            position: 持仓

        Returns:
            可卖出数量
        """
        if position is None:
            return 0

        if self.t_plus_one:
            # T+1: 可卖 = 总持仓 - 今日买入
            return max(0, position.quantity - position.today_bought)
        else:
            # T+0: 可卖 = 总持仓
            return position.quantity


# 便捷函数
def create_order(
    symbol: str,
    side: str,
    quantity: int,
    price: Optional[float] = None,
    order_type: str = "limit",
) -> Order:
    """创建订单"""
    return Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_type=order_type,
    )


__all__ = [
    "ASharesRulesEngine",
    "Order",
    "Position",
    "OrderRejectReason",
    "create_order",
]
