"""
数据管理器
负责数据获取、缓存和管理
"""

from typing import Dict, List, Optional

import pandas as pd

from data.market.sources.akshare_provider import AKShareProvider
from data.market.sources.base_provider import BaseDataProvider
from data.market.sources.tushare_provider import TushareProvider
from utils.logging import get_logger

logger = get_logger(__name__)


class DataManager:
    """
    数据管理器

    统一管理不同数据源的获取和缓存
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            provider: 数据提供者类型 ("akshare", "tushare")，默认使用AKShare
            cache_dir: 缓存目录
        """
        # 根据provider名称创建对应的provider对象
        if provider == "tushare":
            self.provider = TushareProvider()
        else:  # 默认使用AKShare
            self.provider = AKShareProvider()

        self.cache_dir = cache_dir
        self._cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"DataManager初始化，使用数据源: {self.provider.__class__.__name__}")

    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        获取股票数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存

        Returns:
            K线数据
        """
        cache_key = f"{symbol}_{start_date}_{end_date}"

        if use_cache and cache_key in self._cache:
            logger.debug(f"从缓存获取数据: {symbol}")
            return self._cache[cache_key].copy()

        logger.info(f"获取数据: {symbol} ({start_date} ~ {end_date})")

        try:
            # 根据provider类型调用相应的方法
            if hasattr(self.provider, 'get_daily_bar'):
                data = self.provider.get_daily_bar(symbol, start_date, end_date)
            elif hasattr(self.provider, 'get_stock_data'):
                data = self.provider.get_stock_data(symbol, start_date, end_date)
            else:
                raise AttributeError(f"Provider {self.provider.__class__.__name__} 没有可用的数据获取方法")

            if use_cache and not data.empty:
                self._cache[cache_key] = data.copy()

            return data

        except Exception as e:
            logger.error(f"获取数据失败: {symbol}, 错误: {e}")
            return pd.DataFrame()

    def get_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量获取股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            股票数据字典 {symbol: DataFrame}
        """
        result = {}

        for symbol in symbols:
            data = self.get_stock_data(symbol, start_date, end_date)
            if not data.empty:
                result[symbol] = data

        return result

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("缓存已清空")

    def get_cached_symbols(self) -> List[str]:
        """获取已缓存的股票列表"""
        return list(self._cache.keys())


__all__ = ["DataManager"]
