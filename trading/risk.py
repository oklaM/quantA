"""
风险控制模块
提供订单风险评估和限制
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


class ActionType(Enum):
    """订单类型"""

    BUY = "buy"
    SELL = "sell"
    CANCEL = "cancel"


@dataclass
class OrderRequest:
    """订单请求"""

    symbol: str
    action: ActionType
    quantity: int
    price: Optional[float] = None
    order_type: str = "limit"  # limit, market
    datetime: Optional[datetime] = None

    def __post_init__(self):
        if self.datetime is None:
            self.datetime = datetime.now()

        if isinstance(self.action, str):
            self.action = ActionType(self.action)


class RiskController:
    """
    风险控制器

    检查订单是否符合风险控制规则
    """

    def __init__(
        self,
        max_daily_loss_ratio: float = 0.05,
        max_single_order_amount: float = 1_000_000,
        max_position_ratio: float = 0.30,
        stop_loss_ratio: float = 0.05,
        take_profit_ratio: float = 0.15,
    ):
        """
        Args:
            max_daily_loss_ratio: 单日最大亏损比例
            max_single_order_amount: 单笔订单最大金额
            max_position_ratio: 单个股票最大持仓比例
            stop_loss_ratio: 止损比例
            take_profit_ratio: 止盈比例
        """
        self.max_daily_loss_ratio = max_daily_loss_ratio
        self.max_single_order_amount = max_single_order_amount
        self.max_position_ratio = max_position_ratio
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio

        # 今日统计
        self._daily_orders: List[OrderRequest] = []
        self._daily_pnl = 0.0

        logger.info("RiskController初始化")

    def validate_order(
        self,
        order: OrderRequest,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        验证订单是否符合风控规则

        Args:
            order: 订单请求
            context: 上下文信息（账户、持仓等）

        Returns:
            (是否通过, 拒绝原因)
        """
        context = context or {}

        # 检查订单金额
        if order.price:
            order_amount = order.quantity * order.price
            if order_amount > self.max_single_order_amount:
                reason = f"单笔订单金额超限: {order_amount:.2f} > {self.max_single_order_amount}"
                return False, reason

        # 检查持仓比例
        if context and "portfolio" in context:
            portfolio = context["portfolio"]
            current_value = portfolio.get("total_value", 0)
            position_value = portfolio.get("positions", {}).get(order.symbol, 0)

            if current_value > 0:
                position_ratio = (
                    position_value + order.quantity * (order.price or 0)
                ) / current_value
                if position_ratio > self.max_position_ratio:
                    reason = f"持仓比例超限: {position_ratio:.2%} > {self.max_position_ratio:.2%}"
                    return False, reason

        # 检查当日亏损
        if self._daily_pnl < 0:
            loss_ratio = abs(self._daily_pnl) / context.get("initial_cash", 1_000_000)
            if loss_ratio > self.max_daily_loss_ratio:
                reason = (
                    f"当日亏损超限: {loss_ratio:.2%} > {self.max_daily_loss_ratio:.2%}"
                )
                return False, reason

        # 记录订单
        self._daily_orders.append(order)

        logger.debug(
            f"订单通过风控: {order.symbol} {order.action.value} {order.quantity}"
        )
        return True, None

    def update_pnl(self, pnl: float):
        """
        更新当日盈亏

        Args:
            pnl: 盈亏金额
        """
        self._daily_pnl += pnl
        logger.debug(f"更新当日盈亏: {self._daily_pnl:.2f}")

    def reset_daily(self):
        """重置当日统计"""
        self._daily_orders.clear()
        self._daily_pnl = 0.0
        logger.info("重置当日统计")

    def get_daily_summary(self) -> Dict[str, Any]:
        """获取当日统计摘要"""
        return {
            "order_count": len(self._daily_orders),
            "daily_pnl": self._daily_pnl,
            "daily_pnl_ratio": self._daily_pnl / 1_000_000,  # 假设初始100万
        }


__all__ = ["ActionType", "OrderRequest", "RiskController"]
