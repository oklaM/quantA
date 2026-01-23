"""
数据源基类
定义数据源的统一接口
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)


class BaseDataProvider(ABC):
    """
    数据源基类
    所有数据源提供者应继承此类并实现相应方法
    """

    def __init__(self, name: str):
        self.name = name
        self._is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        建立连接

        Returns:
            是否连接成功
        """
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @property
    def is_connected(self) -> bool:
        """是否已连接"""
        return self._is_connected

    # ====================
    # 基础行情数据
    # ====================

    @abstractmethod
    def get_daily_bar(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",  # qfq-前复权, hfq-后复权, ''-不复权
    ) -> pd.DataFrame:
        """
        获取日线数据

        Args:
            symbol: 股票代码 (如 "600519.SH" 或 "600519")
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型

        Returns:
            DataFrame with columns: [open, high, low, close, volume, amount]
        """
        pass

    @abstractmethod
    def get_minute_bar(
        self,
        symbol: str,
        trade_date: str,
        period: int = 1,
    ) -> pd.DataFrame:
        """
        获取分钟线数据

        Args:
            symbol: 股票代码
            trade_date: 交易日期 (YYYYMMDD)
            period: 分钟周期 (1, 5, 15, 30, 60)

        Returns:
            DataFrame with columns: [time, open, high, low, close, volume, amount]
        """
        pass

    @abstractmethod
    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """
        获取实时行情

        Args:
            symbols: 股票代码列表

        Returns:
            DataFrame with columns: [symbol, last_price, bid_price, ask_price, ...]
        """
        pass

    # ====================
    # 基础信息数据
    # ====================

    @abstractmethod
    def get_stock_list(self, market: Optional[str] = None) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            market: 市场类型 (SH/SZ)

        Returns:
            DataFrame with columns: [symbol, name, market, industry, list_date]
        """
        pass

    @abstractmethod
    def get_index_list(self) -> pd.DataFrame:
        """
        获取指数列表

        Returns:
            DataFrame with columns: [symbol, name, market]
        """
        pass

    @abstractmethod
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取股票基本信息

        Args:
            symbol: 股票代码

        Returns:
            股票信息字典
        """
        pass

    # ====================
    # 财务数据
    # ====================

    @abstractmethod
    def get_financial(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取财务数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            财务数据DataFrame
        """
        pass

    # ====================
    # 实用方法
    # ====================

    def normalize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码格式

        Args:
            symbol: 原始代码

        Returns:
            标准化后的代码 (如 "600519.SH")
        """
        symbol = symbol.strip().upper()

        # 已经有后缀
        if "." in symbol:
            return symbol

        # 添加后缀
        if symbol.startswith("6"):
            return f"{symbol}.SH"
        elif symbol.startswith(("0", "3")):
            return f"{symbol}.SZ"
        else:
            return symbol

    def validate_date_range(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> tuple:
        """
        验证日期范围

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            (start_date, end_date)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        if start_date is None:
            # 默认获取最近一年数据
            from datetime import timedelta
            start = datetime.now() - timedelta(days=365)
            start_date = start.strftime("%Y%m%d")

        return start_date, end_date


class DataProviderError(Exception):
    """数据源异常基类"""
    pass


class ConnectionError(DataProviderError):
    """连接异常"""
    pass


class DataNotFoundError(DataProviderError):
    """数据不存在异常"""
    pass


class RateLimitError(DataProviderError):
    """请求频率限制异常"""
    pass


__all__ = [
    'BaseDataProvider',
    'DataProviderError',
    'ConnectionError',
    'DataNotFoundError',
    'RateLimitError',
]
