"""
实盘交易风控规则系统
提供完整的前置风控、实时监控和异常处理
"""

from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, time
import threading
from collections import defaultdict

from utils.logging import get_logger

logger = get_logger(__name__)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """操作类型"""
    BUY = "buy"
    SELL = "sell"
    CANCEL = "cancel"


class RiskRuleType(Enum):
    """风控规则类型"""
    # 资金相关
    CASH_LIMIT = "cash_limit"
    SINGLE_ORDER_LIMIT = "single_order_limit"
    DAILY_VOLUME_LIMIT = "daily_volume_limit"

    # 持仓相关
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    LEVERAGE_LIMIT = "leverage_limit"

    # 品种相关
    STOCK_BLACKLIST = "stock_blacklist"
    ST_STOCK_LIMIT = "st_stock_limit"

    # 时间相关
    TRADING_TIME_LIMIT = "trading_time_limit"
    OPEN_AUCTION_LIMIT = "open_auction_limit"
    CLOSE_AUCTION_LIMIT = "close_auction_limit"

    # 价格相关
    PRICE_LIMIT = "price_limit"
    IMPACT_COST_LIMIT = "impact_cost_limit"

    # 交易频率
    ORDER_FREQUENCY_LIMIT = "order_frequency_limit"
    CANCEL_FREQUENCY_LIMIT = "cancel_frequency_limit"

    # 策略相关
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    passed: bool
    rule_type: RiskRuleType
    rule_name: str
    message: str
    risk_level: RiskLevel = RiskLevel.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    action: ActionType
    quantity: int
    price: Optional[float] = None
    order_type: str = "limit"  # market, limit
    strategy_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class BaseRiskRule:
    """风控规则基类"""

    def __init__(
        self,
        name: str,
        rule_type: RiskRuleType,
        enabled: bool = True,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
    ):
        self.name = name
        self.rule_type = rule_type
        self.enabled = enabled
        self.risk_level = risk_level

    def check(
        self,
        order: OrderRequest,
        context: Dict[str, Any],
    ) -> RiskCheckResult:
        """
        执行风控检查

        Args:
            order: 订单请求
            context: 上下文数据（账户、持仓、市场数据等）

        Returns:
            检查结果
        """
        raise NotImplementedError("子类必须实现check方法")

    def enable(self):
        """启用规则"""
        self.enabled = True
        logger.info(f"风控规则已启用: {self.name}")

    def disable(self):
        """禁用规则"""
        self.enabled = False
        logger.info(f"风控规则已禁用: {self.name}")


class CashLimitRule(BaseRiskRule):
    """资金限制规则"""

    def __init__(
        self,
        min_available_cash: float = 0,
        max_cash_usage_ratio: float = 1.0,
    ):
        super().__init__(
            name="资金限制",
            rule_type=RiskRuleType.CASH_LIMIT,
            risk_level=RiskLevel.CRITICAL,
        )
        self.min_available_cash = min_available_cash
        self.max_cash_usage_ratio = max_cash_usage_ratio

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查资金是否充足"""
        account = context.get('account', {})
        available_cash = account.get('available_cash', 0)

        if order.action == ActionType.SELL:
            return RiskCheckResult(
                passed=True,
                rule_type=self.rule_type,
                rule_name=self.name,
                message="卖出操作无需检查资金",
            )

        # 计算所需资金
        required_cash = order.quantity * (order.price or 0)

        # 检查最小可用资金
        if available_cash - required_cash < self.min_available_cash:
            return RiskCheckResult(
                passed=False,
                rule_type=self.rule_type,
                rule_name=self.name,
                message=f"资金不足: 订单需要{required_cash:,.2f}，可用{available_cash:,.2f}",
                risk_level=RiskLevel.CRITICAL,
                metadata={
                    'required': required_cash,
                    'available': available_cash,
                    'shortage': self.min_available_cash,
                },
            )

        # 检查资金使用比例
        total_asset = account.get('total_asset', 0)
        if total_asset > 0:
            cash_ratio = required_cash / total_asset
            if cash_ratio > self.max_cash_usage_ratio:
                return RiskCheckResult(
                    passed=False,
                    rule_type=self.rule_type,
                    rule_name=self.name,
                    message=f"单笔资金使用超限: {cash_ratio:.2%} > {self.max_cash_usage_ratio:.2%}",
                    risk_level=RiskLevel.HIGH,
                )

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="资金检查通过",
        )


class SingleOrderLimitRule(BaseRiskRule):
    """单笔订单限制"""

    def __init__(
        self,
        max_quantity: int = 1000000,
        max_amount: float = 10000000,
    ):
        super().__init__(
            name="单笔订单限制",
            rule_type=RiskRuleType.SINGLE_ORDER_LIMIT,
            risk_level=RiskLevel.HIGH,
        )
        self.max_quantity = max_quantity
        self.max_amount = max_amount

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查单笔订单是否超限"""
        # 数量检查
        if order.quantity > self.max_quantity:
            return RiskCheckResult(
                passed=False,
                rule_type=self.rule_type,
                rule_name=self.name,
                message=f"单笔数量超限: {order.quantity} > {self.max_quantity}",
                risk_level=RiskLevel.HIGH,
                metadata={
                    'order_quantity': order.quantity,
                    'max_quantity': self.max_quantity,
                },
            )

        # 金额检查
        if order.price:
            amount = order.quantity * order.price
            if amount > self.max_amount:
                return RiskCheckResult(
                    passed=False,
                    rule_type=self.rule_type,
                    rule_name=self.name,
                    message=f"单笔金额超限: {amount:,.2f} > {self.max_amount:,.2f}",
                    risk_level=RiskLevel.HIGH,
                    metadata={
                        'order_amount': amount,
                        'max_amount': self.max_amount,
                    },
                )

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="单笔订单检查通过",
        )


class DailyVolumeLimitRule(BaseRiskRule):
    """日交易量限制"""

    def __init__(
        self,
        max_daily_volume: float = 50000000,
    ):
        super().__init__(
            name="日交易量限制",
            rule_type=RiskRuleType.DAILY_VOLUME_LIMIT,
            risk_level=RiskLevel.HIGH,
        )
        self.max_daily_volume = max_daily_volume

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查日交易量"""
        stats = context.get('daily_stats', {})
        traded_volume = stats.get('traded_volume', 0)

        if order.price:
            order_volume = order.quantity * order.price
        else:
            # 市价单估算
            order_volume = order.quantity * 100  # 粗略估计

        if traded_volume + order_volume > self.max_daily_volume:
            return RiskCheckResult(
                passed=False,
                rule_type=self.rule_type,
                rule_name=self.name,
                message=f"日交易量超限: {traded_volume + order_volume:,.2f} > {self.max_daily_volume:,.2f}",
                risk_level=RiskLevel.HIGH,
                metadata={
                    'current': traded_volume,
                    'order': order_volume,
                    'total': traded_volume + order_volume,
                    'limit': self.max_daily_volume,
                },
            )

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="日交易量检查通过",
        )


class PositionLimitRule(BaseRiskRule):
    """持仓限制"""

    def __init__(
        self,
        max_positions: int = 50,
        max_single_position_ratio: float = 0.3,
    ):
        super().__init__(
            name="持仓限制",
            rule_type=RiskRuleType.POSITION_LIMIT,
            risk_level=RiskLevel.MEDIUM,
        )
        self.max_positions = max_positions
        self.max_single_position_ratio = max_single_position_ratio

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查持仓限制"""
        positions = context.get('positions', [])
        account = context.get('account', {})

        # 持仓数量检查（买入）
        if order.action == ActionType.BUY:
            current_positions = len([p for p in positions if p['quantity'] > 0])
            symbol_positions = [p for p in positions if p['symbol'] == order.symbol]

            if not symbol_positions and current_positions >= self.max_positions:
                return RiskCheckResult(
                    passed=False,
                    rule_type=self.rule_type,
                    rule_name=self.name,
                    message=f"持仓数量超限: 当前{current_positions}，最多{self.max_positions}",
                    risk_level=RiskLevel.MEDIUM,
                )

        # 单一持仓比例检查
        total_asset = account.get('total_asset', 0)
        if total_asset > 0 and order.price:
            # 计算订单后的持仓价值
            current_value = sum(
                p['market_value']
                for p in positions
                if p['symbol'] == order.symbol
            )

            order_value = order.quantity * order.price
            new_value = current_value + order_value if order.action == ActionType.BUY else current_value - order_value

            if new_value / total_asset > self.max_single_position_ratio:
                return RiskCheckResult(
                    passed=False,
                    rule_type=self.rule_type,
                    rule_name=self.name,
                    message=f"单一持仓比例超限: {new_value / total_asset:.2%} > {self.max_single_position_ratio:.2%}",
                    risk_level=RiskLevel.MEDIUM,
                    metadata={
                        'symbol': order.symbol,
                        'current_ratio': current_value / total_asset if total_asset > 0 else 0,
                        'new_ratio': new_value / total_asset,
                        'limit': self.max_single_position_ratio,
                    },
                )

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="持仓检查通过",
        )


class StockBlacklistRule(BaseRiskRule):
    """股票黑名单规则"""

    def __init__(self, blacklist: Optional[List[str]] = None):
        super().__init__(
            name="股票黑名单",
            rule_type=RiskRuleType.STOCK_BLACKLIST,
            risk_level=RiskLevel.CRITICAL,
        )
        self.blacklist = set(blacklist or [])

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查是否在黑名单中"""
        if order.symbol in self.blacklist:
            return RiskCheckResult(
                passed=False,
                rule_type=self.rule_type,
                rule_name=self.name,
                message=f"股票在黑名单中: {order.symbol}",
                risk_level=RiskLevel.CRITICAL,
                metadata={'symbol': order.symbol},
            )

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="黑名单检查通过",
        )

    def add_to_blacklist(self, symbol: str):
        """添加到黑名单"""
        self.blacklist.add(symbol)
        logger.warning(f"股票已添加到黑名单: {symbol}")

    def remove_from_blacklist(self, symbol: str):
        """从黑名单移除"""
        self.blacklist.discard(symbol)
        logger.info(f"股票已从黑名单移除: {symbol}")


class TradingTimeLimitRule(BaseRiskRule):
    """交易时间限制"""

    def __init__(
        self,
        allowed_start: time = time(9, 30),
        allowed_end: time = time(15, 0),
        break_start: time = time(11, 30),
        break_end: time = time(13, 0),
    ):
        super().__init__(
            name="交易时间限制",
            rule_type=RiskRuleType.TRADING_TIME_LIMIT,
            risk_level=RiskLevel.HIGH,
        )
        self.allowed_start = allowed_start
        self.allowed_end = allowed_end
        self.break_start = break_start
        self.break_end = break_end

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查交易时间"""
        now = datetime.now().time()

        # 检查是否在交易时间段内
        if not (self.allowed_start <= now <= self.allowed_end):
            return RiskCheckResult(
                passed=False,
                rule_type=self.rule_type,
                rule_name=self.name,
                message=f"非交易时间: {now}",
                risk_level=RiskLevel.HIGH,
            )

        # 检查是否在午间休市
        if self.break_start <= now <= self.break_end:
            return RiskCheckResult(
                passed=False,
                rule_type=self.rule_type,
                rule_name=self.name,
                message=f"午间休市时间: {now}",
                risk_level=RiskLevel.HIGH,
            )

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="交易时间检查通过",
        )


class DailyLossLimitRule(BaseRiskRule):
    """日亏损限制"""

    def __init__(
        self,
        max_daily_loss_ratio: float = 0.05,  # 5%
        stop_trading: bool = True,
    ):
        super().__init__(
            name="日亏损限制",
            rule_type=RiskRuleType.DAILY_LOSS_LIMIT,
            risk_level=RiskLevel.CRITICAL,
        )
        self.max_daily_loss_ratio = max_daily_loss_ratio
        self.stop_trading = stop_trading

    def check(self, order: OrderRequest, context: Dict[str, Any]) -> RiskCheckResult:
        """检查日亏损"""
        stats = context.get('daily_stats', {})
        daily_pnl = stats.get('daily_pnl', 0)
        initial_asset = stats.get('initial_asset', 0)

        if initial_asset > 0:
            daily_return = daily_pnl / initial_asset

            if daily_return < -self.max_daily_loss_ratio:
                message = f"日亏损超限: {daily_return:.2%} < -{self.max_daily_loss_ratio:.2%}"

                if self.stop_trading:
                    return RiskCheckResult(
                        passed=False,
                        rule_type=self.rule_type,
                        rule_name=self.name,
                        message=message,
                        risk_level=RiskLevel.CRITICAL,
                        metadata={
                            'daily_return': daily_return,
                            'limit': -self.max_daily_loss_ratio,
                        },
                    )
                else:
                    # 仅告警，不阻止交易
                    logger.warning(message)

        return RiskCheckResult(
            passed=True,
            rule_type=self.rule_type,
            rule_name=self.name,
            message="日亏损检查通过",
        )


class RiskManager:
    """
    风控管理器

    管理所有风控规则，执行风控检查
    """

    def __init__(self):
        self.rules: List[BaseRiskRule] = []
        self.lock = threading.Lock()

        # 统计信息
        self.check_count = 0
        self.reject_count = 0
        self.rejects_by_rule: Dict[str, int] = defaultdict(int)

        logger.info("风控管理器初始化")

    def add_rule(self, rule: BaseRiskRule):
        """添加风控规则"""
        with self.lock:
            self.rules.append(rule)
            logger.info(f"添加风控规则: {rule.name}")

    def remove_rule(self, rule_name: str):
        """移除风控规则"""
        with self.lock:
            self.rules = [r for r in self.rules if r.name != rule_name]
            logger.info(f"移除风控规则: {rule_name}")

    def check_order(
        self,
        order: OrderRequest,
        context: Dict[str, Any],
    ) -> List[RiskCheckResult]:
        """
        检查订单（执行所有规则）

        Args:
            order: 订单请求
            context: 上下文数据

        Returns:
            检查结果列表
        """
        with self.lock:
            self.check_count += 1

        results = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                result = rule.check(order, context)
                results.append(result)

                # 统计拒绝次数
                if not result.passed:
                    self.reject_count += 1
                    self.rejects_by_rule[rule.name] += 1
                    logger.warning(
                        f"风控拒绝: {rule.name} - {result.message}"
                    )

            except Exception as e:
                logger.error(f"风控检查异常 {rule.name}: {e}")

        return results

    def is_order_allowed(
        self,
        order: OrderRequest,
        context: Dict[str, Any],
    ) -> tuple[bool, List[str]]:
        """
        判断订单是否允许

        Args:
            order: 订单请求
            context: 上下文数据

        Returns:
            (是否允许, 拒绝原因列表)
        """
        results = self.check_order(order, context)

        rejects = [r.message for r in results if not r.passed]
        allowed = len(rejects) == 0

        return allowed, rejects

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            return {
                'total_checks': self.check_count,
                'total_rejects': self.reject_count,
                'reject_ratio': self.reject_count / self.check_count if self.check_count > 0 else 0,
                'rejects_by_rule': dict(self.rejects_by_rule),
                'active_rules': len([r for r in self.rules if r.enabled]),
                'total_rules': len(self.rules),
            }


class RiskController:
    """
    风控控制器

    提供高层风控接口
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 风控配置
        """
        self.config = config or {}
        self.manager = RiskManager()

        # 初始化默认规则
        self._init_default_rules()

    def _init_default_rules(self):
        """初始化默认风控规则"""
        # 资金限制
        self.manager.add_rule(CashLimitRule(
            min_available_cash=self.config.get('min_available_cash', 0),
            max_cash_usage_ratio=self.config.get('max_cash_usage_ratio', 1.0),
        ))

        # 单笔订单限制
        self.manager.add_rule(SingleOrderLimitRule(
            max_quantity=self.config.get('max_single_order_quantity', 1000000),
            max_amount=self.config.get('max_single_order_amount', 10000000),
        ))

        # 日交易量限制
        self.manager.add_rule(DailyVolumeLimitRule(
            max_daily_volume=self.config.get('max_daily_volume', 50000000),
        ))

        # 持仓限制
        self.manager.add_rule(PositionLimitRule(
            max_positions=self.config.get('max_positions', 50),
            max_single_position_ratio=self.config.get('max_single_position_ratio', 0.3),
        ))

        # 交易时间限制
        self.manager.add_rule(TradingTimeLimitRule())

        # 日亏损限制
        self.manager.add_rule(DailyLossLimitRule(
            max_daily_loss_ratio=self.config.get('max_daily_loss_ratio', 0.05),
        ))

        # 股票黑名单
        blacklist = self.config.get('stock_blacklist', [])
        if blacklist:
            self.manager.add_rule(StockBlacklistRule(blacklist))

    def validate_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, List[str]]:
        """
        验证订单

        Args:
            symbol: 股票代码
            action: 'buy' or 'sell'
            quantity: 数量
            price: 价格
            context: 上下文数据

        Returns:
            (是否允许, 拒绝原因列表)
        """
        # 构建订单请求
        order = OrderRequest(
            symbol=symbol,
            action=ActionType.BUY if action == 'buy' else ActionType.SELL,
            quantity=quantity,
            price=price,
        )

        # 执行风控检查
        context = context or {}
        return self.manager.is_order_allowed(order, context)

    def add_custom_rule(self, rule: BaseRiskRule):
        """添加自定义风控规则"""
        self.manager.add_rule(rule)

    def get_statistics(self) -> Dict[str, Any]:
        """获取风控统计"""
        return self.manager.get_statistics()

    def enable_rule(self, rule_name: str):
        """启用规则"""
        for rule in self.manager.rules:
            if rule.name == rule_name:
                rule.enable()
                break

    def disable_rule(self, rule_name: str):
        """禁用规则"""
        for rule in self.manager.rules:
            if rule.name == rule_name:
                rule.disable()
                break


__all__ = [
    'RiskLevel',
    'ActionType',
    'RiskRuleType',
    'RiskCheckResult',
    'OrderRequest',
    'BaseRiskRule',
    'CashLimitRule',
    'SingleOrderLimitRule',
    'DailyVolumeLimitRule',
    'PositionLimitRule',
    'StockBlacklistRule',
    'TradingTimeLimitRule',
    'DailyLossLimitRule',
    'RiskManager',
    'RiskController',
]
