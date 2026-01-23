"""
数据采集调度器
负责从数据源采集数据并存储到时序数据库
"""

from typing import Optional, List
import pandas as pd
from datetime import datetime, timedelta

from data.market.storage.timeseries_db import get_timeseries_db, TABLE_SCHEMAS
from data.market.sources.tushare_provider import TushareProvider
from data.market.sources.akshare_provider import AKShareProvider
from config.settings import data as data_config
from utils.logging import get_logger
from utils.time_utils import is_trading_day, get_previous_trading_day

logger = get_logger(__name__)


class DataCollector:
    """数据采集调度器"""

    def __init__(
        self,
        provider: Optional[str] = None,
        storage: Optional[str] = None,
    ):
        """
        Args:
            provider: 数据源 ("tushare", "akshare")
            storage: 存储类型 ("duckdb", "clickhouse")
        """
        # 初始化数据源
        self.provider_name = provider or ("tushare" if data_config.TUSHARE_TOKEN else "akshare")
        self._init_provider()

        # 初始化存储
        self.db = get_timeseries_db()
        self.db.connect()

        # 初始化表
        self._init_tables()

    def _init_provider(self):
        """初始化数据源"""
        if self.provider_name == "tushare":
            self.provider = TushareProvider()
        elif self.provider_name == "akshare":
            self.provider = AKShareProvider()
        else:
            raise ValueError(f"不支持的数据源: {self.provider_name}")

        self.provider.connect()
        logger.info(f"使用数据源: {self.provider_name}")

    def _init_tables(self):
        """初始化数据库表"""
        for table_name, schema in TABLE_SCHEMAS.items():
            if not self.db.table_exists(table_name):
                self.db.create_table(table_name, schema)
                logger.info(f"创建表: {table_name}")

    # ====================
    # 数据采集方法
    # ====================

    def collect_daily_bar(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> int:
        """
        采集日线数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型

        Returns:
            采集的记录数
        """
        logger.info(f"开始采集日线数据: {len(symbols)}只股票")

        total_count = 0

        for i, symbol in enumerate(symbols):
            try:
                # 获取数据
                df = self.provider.get_daily_bar(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )

                if df.empty:
                    logger.warning(f"未获取到数据: {symbol}")
                    continue

                # 存储到数据库
                self.db.write("daily_bar", df)
                total_count += len(df)

                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(symbols)}, 已采集{total_count}条记录")

            except Exception as e:
                logger.error(f"采集数据失败 {symbol}: {e}")
                continue

        logger.info(f"日线数据采集完成: {total_count}条记录")
        return total_count

    def collect_minute_bar(
        self,
        symbols: List[str],
        trade_date: str,
        period: int = 1,
    ) -> int:
        """
        采集分钟线数据

        Args:
            symbols: 股票代码列表
            trade_date: 交易日期
            period: 分钟周期

        Returns:
            采集的记录数
        """
        logger.info(f"开始采集分钟线数据: {len(symbols)}只股票, 日期={trade_date}")

        total_count = 0

        for i, symbol in enumerate(symbols):
            try:
                # 获取数据
                df = self.provider.get_minute_bar(
                    symbol=symbol,
                    trade_date=trade_date,
                    period=period,
                )

                if df.empty:
                    logger.warning(f"未获取到数据: {symbol} {trade_date}")
                    continue

                # 存储到数据库
                self.db.write("minute_bar", df)
                total_count += len(df)

                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(symbols)}, 已采集{total_count}条记录")

            except Exception as e:
                logger.error(f"采集数据失败 {symbol}: {e}")
                continue

        logger.info(f"分钟线数据采集完成: {total_count}条记录")
        return total_count

    def collect_stock_list(self) -> int:
        """
        采集股票列表

        Returns:
            股票数量
        """
        logger.info("开始采集股票列表")

        try:
            # 获取股票列表
            df = self.provider.get_stock_list()

            if df.empty:
                logger.warning("未获取到股票列表")
                return 0

            # 存储到数据库
            self.db.write("stock_info", df)

            logger.info(f"股票列表采集完成: {len(df)}只股票")
            return len(df)

        except Exception as e:
            logger.error(f"采集股票列表失败: {e}")
            return 0

    # ====================
    # 便捷方法
    # ====================

    def update_daily_data(self, symbols: List[str], days: int = 5):
        """
        更新最近的日线数据

        Args:
            symbols: 股票代码列表
            days: 更新最近多少天
        """
        # 计算日期范围
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")

        self.collect_daily_bar(symbols, start_date, end_date)

    def update_today_data(self, symbols: List[str]):
        """
        更新今日数据

        Args:
            symbols: 股票代码列表
        """
        if not is_trading_day():
            logger.info("今天不是交易日，跳过数据更新")
            return

        today = datetime.now().strftime("%Y%m%d")
        self.collect_daily_bar(symbols, today, today)

    def get_daily_bar(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        从数据库读取日线数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            数据DataFrame
        """
        return self.db.read("daily_bar", symbol, start_date, end_date)

    def get_all_symbols(self) -> List[str]:
        """
        获取所有股票代码

        Returns:
            股票代码列表
        """
        df = self.db.read("stock_info")
        return df["symbol"].tolist() if not df.empty else []

    # ====================
    # 清理方法
    # ====================

    def close(self):
        """关闭连接"""
        if self.provider:
            self.provider.disconnect()
        if self.db:
            self.db.disconnect()
        logger.info("数据采集器已关闭")


# 便捷函数
def create_collector(provider: str = "akshare") -> DataCollector:
    """
    创建数据采集器

    Args:
        provider: 数据源 ("tushare", "akshare")

    Returns:
        DataCollector实例
    """
    return DataCollector(provider=provider)


__all__ = [
    'DataCollector',
    'create_collector',
]
