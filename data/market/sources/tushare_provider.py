"""
Tushare数据源实现
https://tushare.pro
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import data as data_config
from data.market.sources.base_provider import (
    BaseDataProvider,
    ConnectionError,
    DataNotFoundError,
    DataProviderError,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# 导入tushare
try:
    import tushare as ts
except ImportError:
    ts = None
    logger.warning("tushare未安装，请运行: pip install tushare")


class TushareProvider(BaseDataProvider):
    """Tushare数据源提供者"""

    def __init__(self, token: Optional[str] = None):
        super().__init__("Tushare")

        self.token = token or data_config.TUSHARE_TOKEN
        if not self.token:
            raise ValueError("Tushare token未设置，请在配置中设置TUSHARE_TOKEN")

        self._api: Optional[Any] = None

    def connect(self) -> bool:
        """建立连接"""
        if ts is None:
            raise ImportError("tushare未安装")

        try:
            # 设置token
            ts.set_token(self.token)
            self._api = ts.pro_api()
            self._is_connected = True
            logger.info("Tushare连接成功")
            return True
        except Exception as e:
            logger.error(f"Tushare连接失败: {e}")
            raise ConnectionError(f"连接失败: {e}")

    def disconnect(self):
        """断开连接"""
        self._api = None
        self._is_connected = False
        logger.info("Tushare已断开")

    @property
    def api(self) -> Any:
        """获取API实例"""
        if not self.is_connected:
            self.connect()
        return self._api

    # ====================
    # 基础行情数据
    # ====================

    def get_daily_bar(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ) -> pd.DataFrame:
        """获取日线数据"""
        # 标准化代码和日期
        symbol = self.normalize_symbol(symbol)
        ts_symbol = symbol.replace(".", "")

        start_date, end_date = self.validate_date_range(start_date, end_date)
        start_date = start_date.replace("-", "")
        end_date = end_date.replace("-", "")

        # 复权参数映射
        adj_map = {"qfq": "qfq", "hfq": "hfq", "": ""}
        adj = adj_map.get(adjust, "qfq")

        try:
            if adj:
                # 复权数据
                df = self.api.daily(
                    ts_code=ts_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adj=adj,
                )
            else:
                # 不复权数据
                df = self.api.daily(
                    ts_code=ts_symbol,
                    start_date=start_date,
                    end_date=end_date,
                )

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到数据: {symbol}")

            # 标准化列名
            df = df.rename(columns={
                "ts_code": "symbol",
                "trade_date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
            })

            # 按日期排序
            df = df.sort_values("date").reset_index(drop=True)

            return df[["symbol", "date", "open", "high", "low", "close", "volume", "amount"]]

        except Exception as e:
            logger.error(f"获取日线数据失败: {e}")
            raise DataProviderError(f"获取日线数据失败: {e}")

    def get_minute_bar(
        self,
        symbol: str,
        trade_date: str,
        period: int = 1,
    ) -> pd.DataFrame:
        """获取分钟线数据"""
        # 标准化
        symbol = self.normalize_symbol(symbol)
        ts_symbol = symbol.replace(".", "")
        trade_date = trade_date.replace("-", "")

        # Tushare分钟线接口：1min, 5min, 15min, 30min, 60min
        freq_map = {1: "1min", 5: "5min", 15: "15min", 30: "30min", 60: "60min"}
        freq = freq_map.get(period, "1min")

        try:
            df = self.api.stk_mins(
                ts_code=ts_symbol,
                trade_date=trade_date,
                freq=freq,
            )

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到分钟线数据: {symbol} {trade_date}")

            # 标准化列名
            df = df.rename(columns={
                "ts_code": "symbol",
                "trade_time": "time",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume",
                "amount": "amount",
            })

            return df[["symbol", "time", "open", "high", "low", "close", "volume", "amount"]]

        except Exception as e:
            logger.error(f"获取分钟线数据失败: {e}")
            raise DataProviderError(f"获取分钟线数据失败: {e}")

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情"""
        try:
            # 标准化代码
            ts_symbols = [self.normalize_symbol(s).replace(".", "") for s in symbols]

            # 分批获取（Tushare单次最多500个）
            all_data = []
            batch_size = 500
            for i in range(0, len(ts_symbols), batch_size):
                batch = ts_symbols[i:i + batch_size]
                df = self.api.stk_mins(ts_code=",".join(batch))
                all_data.append(df)

            df = pd.concat(all_data, ignore_index=True)

            # 标准化列名
            df = df.rename(columns={
                "ts_code": "symbol",
                "last_price": "price",
                "bid_px1": "bid_price",
                "ask_px1": "ask_price",
            })

            return df

        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            raise DataProviderError(f"获取实时行情失败: {e}")

    # ====================
    # 基础信息数据
    # ====================

    def get_stock_list(self, market: Optional[str] = None) -> pd.DataFrame:
        """获取股票列表"""
        try:
            df = self.api.stock_basic(
                exchange='',
                list_status='L',  # L-上市, D-退市, P-暂停上市
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )

            # 过滤市场
            if market:
                df = df[df['market'] == market]

            # 标准化列名
            df = df.rename(columns={
                "ts_code": "symbol",
                "name": "name",
                "industry": "industry",
                "market": "market",
                "list_date": "list_date",
            })

            return df

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise DataProviderError(f"获取股票列表失败: {e}")

    def get_index_list(self) -> pd.DataFrame:
        """获取指数列表"""
        try:
            df = self.api.index_basic(market='SSE')  # 上交所指数
            df_sz = self.api.index_basic(market='SZSE')  # 深交所指数
            df = pd.concat([df, df_sz], ignore_index=True)

            # 标准化列名
            df = df.rename(columns={
                "ts_code": "symbol",
                "name": "name",
                "market": "market",
            })

            return df[["symbol", "name", "market"]]

        except Exception as e:
            logger.error(f"获取指数列表失败: {e}")
            raise DataProviderError(f"获取指数列表失败: {e}")

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票基本信息"""
        try:
            symbol = self.normalize_symbol(symbol)
            ts_symbol = symbol.replace(".", "")

            df = self.api.stock_basic(ts_code=ts_symbol)

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到股票信息: {symbol}")

            return df.iloc[0].to_dict()

        except Exception as e:
            logger.error(f"获取股票信息失败: {e}")
            raise DataProviderError(f"获取股票信息失败: {e}")

    # ====================
    # 财务数据
    # ====================

    def get_financial(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取财务数据"""
        try:
            symbol = self.normalize_symbol(symbol)
            ts_symbol = symbol.replace(".", "")

            start_date, end_date = self.validate_date_range(start_date, end_date)
            start_date = start_date.replace("-", "")
            end_date = end_date.replace("-", "")

            # 获取利润表数据
            df = self.api.income(
                ts_code=ts_symbol,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到财务数据: {symbol}")

            return df

        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            raise DataProviderError(f"获取财务数据失败: {e}")


__all__ = ['TushareProvider']
