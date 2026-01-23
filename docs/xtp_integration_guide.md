# 华泰XTP API对接方案

## 1. XTP概述

### 1.1 XTP简介
华泰证券XTP（_extended Trading Platform_）是专业的量化交易接口，提供：
- **极速行情**: 毫秒级行情推送
- **快速交易**: 微秒级订单响应
- **稳定可靠**: 7x24小时稳定运行
- **完整功能**: 支持股票、期权、期货等多品种

### 1.2 适用场景
- 程序化交易
- 算法交易执行
- 高频策略交易
- 套利交易
- 大额资金分仓

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│                    quantA Trading System                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  策略引擎   │  │  风控系统   │  │  监控告警   │           │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
└────────┼───────────────┼───────────────┼──────────────────┘
         │               │               │
┌────────┴───────────────┴───────────────┴──────────────────┐
│                    XTP Adapter Layer                        │
│  ┌──────────────────┐          ┌──────────────────┐       │
│  │  Quote Adapter   │          │  Trade Adapter   │       │
│  │  (行情适配器)     │          │  (交易适配器)     │       │
│  └────────┬─────────┘          └────────┬─────────┘       │
└───────────┼──────────────────────────────┼─────────────────┘
            │                              │
┌───────────┴──────────────────────────────┴─────────────────┐
│                     XTP C API (xtpquote.dll / xtptrader.dll) │
└───────────────────────┬──────────────────────────────────────┘
                        │
┌───────────────────────┴──────────────────────────────────────┐
│              华泰XTP交易服务器 (行情/交易柜台)                  │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件

**XTP Client (Python ctypes封装)**
- 封装XTP C API为Python接口
- 处理异步回调
- 管理连接状态

**Quote Adapter (行情适配器)**
- 订阅行情数据
- 处理行情回调
- 更新本地数据缓存

**Trade Adapter (交易适配器)**
- 订单管理
- 持仓查询
- 资金查询
- 成交通知

**Risk Manager (风控集成)**
- 订单前置检查
- 持仓限制
- 资金限制

## 3. 环境准备

### 3.1 申请XTP账号

```python
# 1. 联系华泰证券申请XTP账号
# 需要提供：
# - 营业执照
# - 法人身份证
# - 组织机构代码证
# - 税务登记证

# 2. 开通权限
# - 行业权限: Level-1/Level-2行情
# - 交易权限: 股票、期权、期货等
# - 账户类型: 普通账户、信用账户

# 3. 获取认证信息
XTP_CLIENT_ID = "your_client_id"      # 客户端ID
XTP_ACCOUNT_ID = "your_account_id"    # 资金账号
XTP_PASSWORD = "your_password"        # 密码
XTP_AUTH_CODE = "your_auth_code"      # 授权码
XTP_HARDWARE_ID = "your_hardware_id"  # 硬件绑定
```

### 3.2 SDK安装

**Windows环境**
```bash
# 1. 下载XTP API SDK
# https://xtp.huataifinance.com/

# 2. 解压到指定目录
# XTP/
# ├── include/          # 头文件
# ├── x64/             # 64位DLL
# │   ├── xtpquote.dll
# │   └── xtptrader.dll
# └── x86/             # 32位DLL

# 3. 安装Python依赖
pip install ctypes numpy pandas

# 4. 配置环境变量
set XTP_SDK_PATH=C:\\xtp\\api
set PATH=%PATH%;%XTP_SDK_PATH%\\x64
```

**Linux环境**
```bash
# 1. 下载Linux版SDK
# 包含: libxtpquote.so, libxtptrader.so

# 2. 安装依赖库
sudo apt-get install build-essential python3-dev

# 3. 创建符号链接
sudo ln -s /path/to/libxtpquote.so /usr/local/lib/
sudo ln -s /path/to/libxtptrader.so /usr/local/lib/
sudo ldconfig
```

## 4. API封装

### 4.1 基础封装

**xtp_client.py**
```python
"""
XTP API客户端封装
"""
import ctypes
from ctypes import c_char, c_int, c_double, c_void_p
from enum import IntEnum
from typing import Callable, Dict, Any
import threading

class XTPErrorCode(IntEnum):
    """XTP错误码"""
    SUCCESS = 0
    NETWORK_ERROR = 1
    LOGIN_FAILED = 2
    INSUFFICIENT_MONEY = 3
    ORDER_FAILED = 4

class XTPApiType(IntEnum):
    """API类型"""
    QUOTE = 1  # 行情API
    TRADER = 2  # 交易API

class XTPClient:
    """XTP客户端基类"""

    def __init__(
        self,
        client_id: int,
        account_id: str,
        password: str,
        sdk_path: str,
    ):
        self.client_id = client_id
        self.account_id = account_id
        self.password = password
        self.sdk_path = sdk_path

        self._connected = False
        self._logged_in = False
        self._session_id = 0

        # 加载DLL
        self._load_library()

    def _load_library(self):
        """加载XTP DLL"""
        try:
            if self.api_type == XTPApiType.QUOTE:
                dll_name = "xtpquote.dll"
            else:
                dll_name = "xtptrader.dll"

            self.lib = ctypes.CDLL(
                f"{self.sdk_path}/{dll_name}"
            )
            logger.info(f"成功加载 {dll_name}")
        except Exception as e:
            logger.error(f"加载XTP DLL失败: {e}")
            raise

    def connect(self, ip: str, port: int) -> bool:
        """连接服务器"""
        # 实现连接逻辑
        pass

    def login(self) -> bool:
        """登录"""
        # 实现登录逻辑
        pass

    def disconnect(self):
        """断开连接"""
        # 实现断开逻辑
        pass
```

### 4.2 行情API

**xtp_quote.py**
```python
"""
XTP行情API
"""
import ctypes
from typing import Callable, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketData:
    """行情数据"""
    symbol: str
    exchange: int
    last_price: float
    bid_price: List[float]  # 买一到买五
    ask_price: List[float]  # 卖一到卖五
    bid_volume: List[int]
    ask_volume: List[int]
    timestamp: datetime
    volume: int
    turnover: float
    open_price: float
    high_price: float
    low_price: float
    pre_close_price: float

class XTPQuoteClient(XTPClient):
    """XTP行情客户端"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, api_type=XTPApiType.QUOTE)

        # 回调函数
        self._on_quote: Callable[[MarketData], None] = None
        self._subscribed_symbols = set()

    def subscribe_quote(
        self,
        symbols: List[str],
        callback: Callable[[MarketData], None],
    ):
        """
        订阅行情

        Args:
            symbols: 股票代码列表 ['600000.SH', '000001.SZ']
            callback: 行情回调函数
        """
        self._on_quote = callback

        for symbol in symbols:
            if symbol not in self._subscribed_symbols:
                self._subscribe(symbol)
                self._subscribed_symbols.add(symbol)

        logger.info(f"订阅行情: {symbols}")

    def _subscribe(self, symbol: str):
        """订阅单只股票"""
        # 调用XTP API订阅
        # XTP_QueryAllTickers可以查询所有股票
        pass

    def unsubscribe_quote(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            if symbol in self._subscribed_symbols:
                self._unsubscribe(symbol)
                self._subscribed_symbols.remove(symbol)

    def get_market_data(self, symbol: str) -> MarketData:
        """获取最新行情"""
        # 从缓存获取或查询最新行情
        pass

    def get_lts_day_begin(self) -> datetime:
        """获取当天交易开始时间"""
        # XTP_GetLtsDayBegin
        pass
```

### 4.3 交易API

**xtp_trade.py**
```python
"""
XTP交易API
"""
from typing import List, Optional, Dict
from dataclasses import dataclass
from enum import IntEnum

class OrderType(IntEnum):
    """订单类型"""
    MARKET = 1  # 市价单
    LIMIT = 2   # 限价单
    BEST5_OR_CANCEL = 3  # 本五档剩余撤销

class Side(IntEnum):
    """买卖方向"""
    BUY = 1
    SELL = 2

class OrderStatus(IntEnum):
    """订单状态"""
    NOT_SUBMITTED = 0
    SUBMITTING = 1
    SUBMITTED = 2
    CANCELING = 3
    CANCELED = 4
    PARTIAL_FILLED = 5
    FILLED = 6
    AUCTION_ONLY = 7
    REJECTED = 8

@dataclass
class Order:
    """订单"""
    order_id: int
    symbol: str
    side: Side
    order_type: OrderType
    quantity: int
    price: Optional[float]
    filled_quantity: int = 0
    filled_amount: float = 0.0
    status: OrderStatus = OrderStatus.NOT_SUBMITTED
    create_time: Optional[datetime] = None
    update_time: Optional[datetime] = None

@dataclass
class Trade:
    """成交"""
    trade_id: int
    order_id: int
    symbol: str
    side: Side
    quantity: int
    price: float
    trade_time: datetime

class XTPTradeClient(XTPClient):
    """XTP交易客户端"""

    def __init__(
        self,
        client_id: int,
        account_id: str,
        password: str,
        auth_code: str,
        hardware_id: str,
        sdk_path: str,
    ):
        super().__init__(
            client_id,
            account_id,
            password,
            sdk_path,
            api_type=XTPApiType.TRADER,
        )
        self.auth_code = auth_code
        self.hardware_id = hardware_id

        self._orders: Dict[int, Order] = {}
        self._trades: Dict[int, Trade] = {}

        # 回调
        self._on_order: Callable[[Order], None] = None
        self._on_trade: Callable[[Trade], None] = None

    def login(self) -> bool:
        """
        登录交易服务器

        Returns:
            是否成功
        """
        # 1. 连接服务器
        # XTP_Connect

        # 2. 登录
        # XTP_Login

        # 3. 查询资产
        self.query_asset()

        return True

    def insert_order(
        self,
        symbol: str,
        side: Side,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
    ) -> int:
        """
        下单

        Args:
            symbol: 股票代码
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格（限价单必需）

        Returns:
            订单ID
        """
        # 风控检查
        if not self._check_risk(symbol, side, quantity, price):
            logger.error("风控检查失败，拒绝订单")
            return -1

        # 调用XTP API下单
        # XTP_InsertOrder

        # 返回订单ID
        order_id = self._generate_order_id()
        return order_id

    def cancel_order(self, order_id: int) -> bool:
        """撤单"""
        # XTP_CancelOrder
        pass

    def query_asset(self) -> Dict[str, float]:
        """查询资产"""
        # XTP_QueryAsset
        pass

    def query_position(self, symbol: Optional[str] = None) -> List[Dict]:
        """查询持仓"""
        # XTP_QueryPosition
        pass

    def query_order(self, order_id: int) -> Optional[Order]:
        """查询单"""
        return self._orders.get(order_id)

    def query_orders(self) -> List[Order]:
        """查询所有订单"""
        return list(self._orders.values())

    def query_trades(self, start_time: datetime, end_time: datetime) -> List[Trade]:
        """查询成交"""
        # XTP_QueryTrades
        pass

    def _check_risk(
        self,
        symbol: str,
        side: Side,
        quantity: int,
        price: Optional[float],
    ) -> bool:
        """风控检查"""
        # 调用风控模块检查
        pass

    def set_order_callback(self, callback: Callable[[Order], None]):
        """设置订单回调"""
        self._on_order = callback

    def set_trade_callback(self, callback: Callable[[Trade], None]):
        """设置成交回调"""
        self._on_trade = callback
```

## 5. 集成实现

### 5.1 交易执行器

**xtp_executor.py**
```python
"""
XTP交易执行器
"""
from trading.executor import BaseExecutor
from xtp_trade import XTPTradeClient, Order, Side, OrderType
from monitoring import AlertManager, Alert, AlertLevel, AlertType

class XTPExecutor(BaseExecutor):
    """XTP交易执行器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # 创建XTP客户端
        self.client = XTPTradeClient(
            client_id=config['client_id'],
            account_id=config['account_id'],
            password=config['password'],
            auth_code=config['auth_code'],
            hardware_id=config['hardware_id'],
            sdk_path=config['sdk_path'],
        )

        # 配置
        self.trade_server = config['trade_server']
        self.quote_server = config['quote_server']

        # 回调
        self.client.set_order_callback(self._on_order_update)
        self.client.set_trade_callback(self._on_trade_update)

        # 告警
        self.alert_manager = AlertManager()

    def connect(self) -> bool:
        """连接"""
        try:
            # 连接交易服务器
            if not self.client.connect(
                ip=self.trade_server['ip'],
                port=self.trade_server['port'],
            ):
                raise Exception("连接交易服务器失败")

            # 登录
            if not self.client.login():
                raise Exception("登录失败")

            logger.info("XTP连接成功")
            return True

        except Exception as e:
            logger.error(f"XTP连接失败: {e}")
            self.alert_manager.send_alert(Alert(
                alert_id="xtp_connect_failed",
                alert_type=AlertType.CONNECTION_ERROR,
                level=AlertLevel.CRITICAL,
                title="XTP连接失败",
                message=str(e),
            ))
            return False

    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: Optional[float] = None,
        order_type: str = "limit",
    ) -> str:
        """
        提交订单

        Args:
            symbol: 股票代码
            side: 'buy' or 'sell'
            quantity: 数量
            price: 价格（限价单）
            order_type: 'market' or 'limit'

        Returns:
            订单ID
        """
        # 转换参数
        xtp_side = Side.BUY if side == 'buy' else Side.SELL
        xtp_type = OrderType.MARKET if order_type == 'market' else OrderType.LIMIT

        # 提交订单
        order_id = self.client.insert_order(
            symbol=symbol,
            side=xtp_side,
            order_type=xtp_type,
            quantity=quantity,
            price=price,
        )

        if order_id > 0:
            logger.info(f"订单提交成功: {order_id}")
            return str(order_id)
        else:
            logger.error("订单提交失败")
            return ""

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        return self.client.cancel_order(int(order_id))

    def get_position(self, symbol: Optional[str] = None) -> List[Dict]:
        """获取持仓"""
        return self.client.query_position(symbol)

    def get_account(self) -> Dict[str, float]:
        """获取账户信息"""
        return self.client.query_asset()

    def _on_order_update(self, order: Order):
        """订单状态更新回调"""
        logger.info(f"订单更新: {order.order_id} - {order.status}")

        # 通知策略
        if self.strategy:
            self.strategy.on_order_update(order)

    def _on_trade_update(self, trade: Trade):
        """成交通知回调"""
        logger.info(f"成交通知: {trade.trade_id} - {trade.symbol} {trade.side} {trade.quantity}@{trade.price}")

        # 通知策略
        if self.strategy:
            self.strategy.on_trade(trade)
```

### 5.2 行情订阅器

**xtp_quote_subscriber.py**
```python
"""
XTP行情订阅器
"""
from data.market import BaseSubscriber
from xtp_quote import XTPQuoteClient, MarketData
from utils.logging import get_logger

logger = get_logger(__name__)

class XTPQuoteSubscriber(BaseSubscriber):
    """XTP行情订阅器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.client = XTPQuoteClient(
            client_id=config['client_id'],
            account_id=config['account_id'],
            password=config['password'],
            sdk_path=config['sdk_path'],
        )

        self.quote_server = config['quote_server']
        self._callbacks = []

    def connect(self) -> bool:
        """连接"""
        try:
            if not self.client.connect(
                ip=self.quote_server['ip'],
                port=self.quote_server['port'],
            ):
                raise Exception("连接行情服务器失败")

            if not self.client.login():
                raise Exception("行情登录失败")

            logger.info("XTP行情连接成功")
            return True

        except Exception as e:
            logger.error(f"XTP行情连接失败: {e}")
            return False

    def subscribe(self, symbols: List[str], callback: Callable):
        """订阅行情"""
        self._callbacks.append(callback)

        self.client.subscribe_quote(
            symbols=symbols,
            callback=self._on_quote,
        )

        logger.info(f"订阅行情: {symbols}")

    def _on_quote(self, data: MarketData):
        """行情回调"""
        # 分发到所有回调
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"行情回调错误: {e}")

    def get_snapshot(self, symbol: str) -> Optional[MarketData]:
        """获取快照"""
        return self.client.get_market_data(symbol)
```

## 6. 风控集成

### 6.1 前置风控

```python
class XTPRiskManager:
    """XTP风控管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.max_single_order = config.get('max_single_order', 1_000_000)
        self.max_daily_volume = config.get('max_daily_volume', 10_000_000)
        self.max_position_ratio = config.get('max_position_ratio', 0.3)

        self.daily_traded = 0.0

    def check_order(self, order: Order, account: Dict, positions: List[Dict]) -> bool:
        """
        订单风控检查

        Args:
            order: 订单
            account: 账户信息
            positions: 持仓列表

        Returns:
            是否通过
        """
        # 1. 资金检查
        if order.side == Side.BUY:
            required = order.quantity * order.price if order.price else 0
            if required > account['available_cash']:
                logger.warning(f"资金不足: 需要{required}, 可用{account['available_cash']}")
                return False

        # 2. 单笔金额检查
        amount = order.quantity * (order.price or 0)
        if amount > self.max_single_order:
            logger.warning(f"单笔金额超限: {amount} > {self.max_single_order}")
            return False

        # 3. 日交易量检查
        if self.daily_traded + amount > self.max_daily_volume:
            logger.warning(f"日交易量超限: {self.daily_traded + amount} > {self.max_daily_volume}")
            return False

        # 4. 持仓集中度检查
        if order.side == Side.BUY:
            current_value = sum(
                p['market_value']
                for p in positions
                if p['symbol'] == order.symbol
            )
            total_value = account['total_asset']
            if (current_value + amount) / total_value > self.max_position_ratio:
                logger.warning(f"持仓集中度过高")
                return False

        return True

    def on_trade(self, trade: Trade):
        """成交后更新"""
        self.daily_traded += trade.quantity * trade.price
```

## 7. 使用示例

### 7.1 完整交易流程

**examples/xtp_trading_example.py**
```python
"""
XTP实盘交易示例
"""
from xtp_trade import XTPTradeClient, Side, OrderType
from xtp_quote import XTPQuoteClient
from monitoring import AlertManager
import time

# 配置
config = {
    # XTP账号信息
    'client_id': 1,
    'account_id': 'your_account',
    'password': 'your_password',
    'auth_code': 'your_auth_code',
    'hardware_id': 'your_hardware_id',
    'sdk_path': 'C:/xtp/api',

    # 服务器地址
    'trade_server': {
        'ip': '120.27.164.138',  # 模拟柜台
        'port': 6001,
    },
    'quote_server': {
        'ip': '120.27.164.138',
        'port': 6002,
    },
}

# 1. 创建交易客户端
trade_client = XTPTradeClient(**config)

# 2. 连接登录
if not trade_client.login():
    print("登录失败")
    exit(1)

# 3. 查询资产
asset = trade_client.query_asset()
print(f"总资产: {asset['total_asset']}")
print(f"可用资金: {asset['available_cash']}")

# 4. 查询持仓
positions = trade_client.query_position()
for pos in positions:
    print(f"持仓: {pos['symbol']} {pos['quantity']}股")

# 5. 创建行情客户端
quote_client = XTPQuoteClient(**config)
quote_client.connect()

# 6. 订阅行情
def on_quote(data):
    print(f"行情: {data.symbol} 价格:{data.last_price}")

quote_client.subscribe_quote(['600000.SH'], on_quote)

# 7. 下单测试
order_id = trade_client.insert_order(
    symbol='600000.SH',
    side=Side.BUY,
    order_type=OrderType.LIMIT,
    quantity=100,
    price=10.50,
)

print(f"订单ID: {order_id}")

# 8. 查询订单状态
time.sleep(1)
order = trade_client.query_order(order_id)
print(f"订单状态: {order.status}")

# 9. 断开连接
trade_client.disconnect()
```

### 7.2 策略集成

```python
from strategies.base import BaseStrategy
from xtp_executor import XTPExecutor

class MyStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)

        # 创建XTP执行器
        self.executor = XTPExecutor(config)
        self.executor.connect()

    def on_bar(self, bar):
        # 策略逻辑
        if self.should_buy(bar):
            self.executor.submit_order(
                symbol=bar.symbol,
                side='buy',
                quantity=1000,
                price=bar.close,
            )

# 运行策略
strategy = MyStrategy(config)
strategy.run()
```

## 8. 注意事项

### 8.1 交易时间
```python
# A股交易时间
TRADING_TIMES = {
    'morning': {'start': '09:30', 'end': '11:30'},
    'afternoon': {'start': '13:00', 'end': '15:00'},
}

# 集合竞价时间
CALL_AUCTION = {
    'morning': '09:15-09:25',
    'afternoon': '14:57-15:00',
}
```

### 8.2 交易规则
```python
# 买卖单位
ROUND_LOT = 100  # 最小交易单位（手）

# 价格变动单位
PRICE_TICK = {
    'stock': 0.01,      # 普通股票
    'fund': 0.001,      # 基金
    'bond': 0.01,       # 债券
}

# 涨跌停限制
LIMIT_RATIO = 0.10  # 10%涨跌停（主板/中小板）
LIMIT_RATIO_ST = 0.05  # 5%涨跌停（ST股票）
LIMIT_RATIO_GEM = 0.20  # 20%涨跌停（创业板/科创板）
```

### 8.3 错误处理

```python
XTP_ERROR_CODES = {
    0: "成功",
    1: "网络错误",
    2: "登录失败",
    3: "发送失败",
    4: "接收失败",
    5: "登录未完成",
    6: "未登录",
    7: "插入订单失败",
    8: "撤单失败",
    # ... 更多错误码
}
```

## 9. 性能优化

### 9.1 行情缓存
```python
from collections import deque
import threading

class QuoteCache:
    """行情缓存"""

    def __init__(self, max_size=1000):
        self.cache = {}
        self.lock = threading.Lock()
        self.max_size = max_size

    def update(self, symbol: str, data: MarketData):
        with self.lock:
            if symbol not in self.cache:
                self.cache[symbol] = deque(maxlen=self.max_size)
            self.cache[symbol].append(data)

    def get_latest(self, symbol: str) -> Optional[MarketData]:
        with self.lock:
            if symbol in self.cache and self.cache[symbol]:
                return self.cache[symbol][-1]
        return None
```

### 9.2 异步处理
```python
import asyncio
import threading

class AsyncXTPClient:
    """异步XTP客户端"""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def submit_order_async(self, order: Order) -> int:
        """异步下单"""
        # 在事件循环中执行
        return await self.loop.run_in_executor(None, self.submit_order, order)
```

## 10. 监控告警

```python
from monitoring import AlertManager, Alert, AlertLevel, AlertType

class XTPMonitor:
    """XTP监控"""

    def __init__(self, client):
        self.client = client
        self.alert_manager = AlertManager()

        # 监控指标
        self.last_heartbeat = time.time()

    def check_connection(self):
        """检查连接状态"""
        if time.time() - self.last_heartbeat > 60:
            self.alert_manager.send_alert(Alert(
                alert_id="xtp_heartbeat_lost",
                alert_type=AlertType.CONNECTION_ERROR,
                level=AlertLevel.CRITICAL,
                title="XTP心跳丢失",
                message="超过60秒未收到XTP心跳",
            ))

    def check_order_rejected(self, order):
        """检查订单被拒"""
        if order.status == OrderStatus.REJECTED:
            self.alert_manager.send_alert(Alert(
                alert_id=f"order_rejected_{order.order_id}",
                alert_type=AlertType.ORDER_REJECTED,
                level=AlertLevel.WARNING,
                title="订单被拒",
                message=f"订单{order.order_id}被柜台拒绝",
                metadata={'order': order.to_dict()},
            ))
```

## 11. 参考资料

- [华泰XTP官网](https://xtp.huataifinance.com/)
- [XTP API文档](https://xtp.huataifinance.com/doc/)
- [XTP下载中心](https://xtp.huataifinance.com/download/)
- [量化交易交流群](技术支持)
