"""
XTP券商接口模块

华泰证券XTP接口的Python封装
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型"""
    MARKET = 1  # 市价单
    LIMIT = 2   # 限价单


class OrderSide(Enum):
    """订单方向"""
    BUY = 48  # 买入
    SELL = 49 # 卖出


class OrderStatus(Enum):
    """订单状态"""
    INIT = 0          # 初始
    ALL_TRADED = 1    # 全部成交
    PART_TRADED = 2   # 部分成交
    CANCELLED = 3     # 已撤销
    REJECTED = 4      # 被拒绝
    NOT_TRADED = 5    // 未成交


@dataclass
class XTPOrder:
    """XTP订单"""
    order_id: int
    order_client_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    filled_quantity: int = 0
    avg_price: float = 0.0
    status: OrderStatus = OrderStatus.INIT
    order_time: Optional[datetime] = None
    error_msg: Optional[str] = None


@dataclass
class XTPPosition:
    """XTP持仓"""
    symbol: str
    quantity: int
    available_quantity: int
    avg_price: float
    current_price: float
    market_value: float
    pnl: float


@dataclass
class XTPAccount:
    """XTP账户"""
    account_id: str
    total_asset: float
    buying_power: float
    cash: float
    frozen_cash: float
    market_value: float
    pnl: float


class XTPBroker:
    """
    XTP券商接口

    注意：这是模拟实现，实际使用需要安装XTP的C++ SDK
    """

    def __init__(
        self,
        account_id: str,
        password: str,
        client_id: int = 1,
        td_ip: str = "127.0.0.1",
        td_port: int = 6001,
        md_ip: str = "127.0.0.1",
        md_port: int = 6002,
        software_key: Optional[str] = None,
    ):
        """
        初始化XTP接口

        Args:
            account_id: 账户ID
            password: 密码
            client_id: 客户端ID
            td_ip: 交易服务器IP
            td_port: 交易服务器端口
            md_ip: 行情服务器IP
            md_port: 行情服务器端口
            software_key: 软件密钥
        """
        self.account_id = account_id
        self.password = password
        self.client_id = client_id
        self.td_ip = td_ip
        self.td_port = td_port
        self.md_ip = md_ip
        self.md_port = md_port
        self.software_key = software_key

        self.connected = False
        self.logged_in = False

        # 订单存储
        self.orders: Dict[int, XTPOrder] = {}
        self.order_id_counter = 0

        logger.info(f"XTP接口已初始化: {account_id}")

    def connect(self) -> bool:
        """
        连接服务器

        Returns:
            是否连接成功
        """
        logger.info(f"连接XTP服务器: {self.td_ip}:{self.td_port}")

        # 模拟连接
        # 实际实现需要调用XTP C++ SDK的API
        self.connected = True
        logger.info("XTP连接成功（模拟模式）")

        return True

    def disconnect(self) -> bool:
        """
        断开连接

        Returns:
            是否断开成功
        """
        logger.info("断开XTP连接")

        # 实际实现需要调用XTP C++ SDK的API
        self.connected = False
        self.logged_in = False

        return True

    def login(self) -> bool:
        """
        登录

        Returns:
            是否登录成功
        """
        if not self.connected:
            logger.error("未连接，请先调用connect()")
            return False

        logger.info(f"登录XTP账户: {self.account_id}")

        # 模拟登录
        # 实际实现需要调用XTP C++ SDK的API
        self.logged_in = True
        logger.info("XTP登录成功（模拟模式）")

        return True

    def logout(self) -> bool:
        """
        登出

        Returns:
            是否登出成功
        """
        logger.info("登出XTP账户")

        # 实际实现需要调用XTP C++ SDK的API
        self.logged_in = False

        return True

    def is_ready(self) -> bool:
        """检查是否就绪"""
        return self.connected and self.logged_in

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        quantity: int,
        client_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        下单

        Args:
            symbol: 股票代码
            side: 方向
            order_type: 类型
            price: 价格
            quantity: 数量
            client_id: 客户端ID

        Returns:
            订单ID
        """
        if not self.is_ready():
            logger.error("XTP未就绪，无法下单")
            return None

        # 检查参数
        if quantity <= 0:
            logger.error(f"无效的数量: {quantity}")
            return None

        if order_type == OrderType.LIMIT and price <= 0:
            logger.error(f"限价单价格必须大于0: {price}")
            return None

        # 检查最小申报单位（A股100股）
        if quantity % 100 != 0:
            logger.warning(f"数量不是100的整数倍: {quantity}")

        # 生成订单ID
        self.order_id_counter += 1
        order_id = self.order_id_counter

        # 创建订单
        order = XTPOrder(
            order_id=order_id,
            order_client_id=client_id or 0,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            order_time=datetime.now(),
        )

        self.orders[order_id] = order

        logger.info(f"下单成功: ID={order_id}, {symbol}, {side.name}, {quantity}股, ¥{price}")

        # 实际实现需要调用XTP C++ SDK的下单API
        # result = xtp_api.insert_order(...)

        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """
        撤单

        Args:
            order_id: 订单ID

        Returns:
            是否成功
        """
        if not self.is_ready():
            logger.error("XTP未就绪，无法撤单")
            return False

        if order_id not in self.orders:
            logger.error(f"订单不存在: {order_id}")
            return False

        order = self.orders[order_id]

        # 检查订单状态
        if order.status in [OrderStatus.ALL_TRADED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            logger.warning(f"订单{order_id}状态为{order.status.name}，无法撤销")
            return False

        # 实际实现需要调用XTP C++ SDK的撤单API
        # result = xtp_api.cancel_order(order_id)

        order.status = OrderStatus.CANCELLED
        logger.info(f"撤单成功: {order_id}")

        return True

    def get_order(self, order_id: int) -> Optional[XTPOrder]:
        """
        查询订单

        Args:
            order_id: 订单ID

        Returns:
            订单信息
        """
        return self.orders.get(order_id)

    def get_account(self) -> Optional[XTPAccount]:
        """
        查询账户

        Returns:
            账户信息
        """
        if not self.is_ready():
            return None

        # 模拟返回
        # 实际实现需要调用XTP C++ SDK的查询API
        return XTPAccount(
            account_id=self.account_id,
            total_asset=1000000.0,
            buying_power=800000.0,
            cash=900000.0,
            frozen_cash=100000.0,
            market_value=100000.0,
            pnl=0.0,
        )

    def get_positions(self) -> List[XTPPosition]:
        """
        查询持仓

        Returns:
            持仓列表
        """
        if not self.is_ready():
            return []

        # 模拟返回
        # 实际实现需要调用XTP C++ SDK的查询API
        return []

    def subscribe_market_data(self, symbols: List[str]) -> bool:
        """
        订阅行情

        Args:
            symbols: 股票代码列表

        Returns:
            是否成功
        """
        logger.info(f"订阅行情: {symbols}")

        # 实际实现需要调用XTP C++ SDK的订阅API
        # result = xtp_api.subscribe_market_data(symbols)

        return True


class SimulatedXTPBroker(XTPBroker):
    """
    模拟XTP接口

    用于测试和回测
    """

    def __init__(self, initial_cash: float = 1000000.0):
        """
        初始化模拟接口

        Args:
            initial_cash: 初始资金
        """
        super().__init__(
            account_id="simulated",
            password="simulated",
        )

        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}

        logger.info(f"模拟XTP接口已初始化: ¥{initial_cash:,.2f}")

    def connect(self) -> bool:
        """连接"""
        self.connected = True
        logger.info("模拟XTP连接成功")
        return True

    def login(self) -> bool:
        """登录"""
        self.logged_in = True
        logger.info("模拟XTP登录成功")
        return True

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: float,
        quantity: int,
        client_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        模拟下单

        Returns:
            订单ID
        """
        # 检查资金（买入）
        if side == OrderSide.BUY:
            required = price * quantity
            if required > self.cash:
                logger.error(f"资金不足: 需要 ¥{required:,.2f}, 可用 ¥{self.cash:,.2f}")
                return None

        # 检查持仓（卖出）
        if side == OrderSide.SELL:
            if symbol not in self.positions or self.positions[symbol] < quantity:
                available = self.positions.get(symbol, 0)
                logger.error(f"持仓不足: 需要 {quantity}, 可用 {available}")
                return None

        # 调用父类下单
        order_id = super().place_order(symbol, side, order_type, price, quantity, client_id)

        if order_id is not None:
            # 模拟即时成交
            order = self.orders[order_id]
            order.status = OrderStatus.ALL_TRADED
            order.filled_quantity = quantity
            order.avg_price = price

            # 更新资金和持仓
            if side == OrderSide.BUY:
                self.cash -= price * quantity
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            else:
                self.positions[symbol] = self.positions[symbol] - quantity
                self.cash += price * quantity

        return order_id

    def get_account(self) -> Optional[XTPAccount]:
        """查询账户"""
        market_value = sum(
            pos * 10.0  # 简化：假设市值为10元
            for pos in self.positions.values()
        )

        return XTPAccount(
            account_id=self.account_id,
            total_asset=self.cash + market_value,
            buying_power=self.cash,
            cash=self.cash,
            frozen_cash=0.0,
            market_value=market_value,
            pnl=(self.cash + market_value) - self.initial_cash,
        )

    def get_positions(self) -> List[XTPPosition]:
        """查询持仓"""
        positions = []

        for symbol, quantity in self.positions.items():
            if quantity > 0:
                positions.append(XTPPosition(
                    symbol=symbol,
                    quantity=quantity,
                    available_quantity=quantity,
                    avg_price=10.0,
                    current_price=10.0,
                    market_value=quantity * 10.0,
                    pnl=0.0,
                ))

        return positions


def create_xtp_broker(
    account_id: str,
    password: str,
    simulated: bool = True,
    **kwargs
) -> XTPBroker:
    """
    创建XTP接口实例

    Args:
        account_id: 账户ID
        password: 密码
        simulated: 是否使用模拟模式
        **kwargs: 其他参数

    Returns:
        XTP接口实例
    """
    if simulated:
        return SimulatedXTPBroker()

    return XTPBroker(
        account_id=account_id,
        password=password,
        **kwargs
    )


__all__ = [
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'XTPOrder',
    'XTPPosition',
    'XTPAccount',
    'XTPBroker',
    'SimulatedXTPBroker',
    'create_xtp_broker',
]
