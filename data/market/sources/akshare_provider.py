"""
AKShare数据源实现
https://akshare.akfamily.xyz
AKShare是一个免费、开源的财经数据接口库
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from data.market.sources.base_provider import (
    BaseDataProvider,
    DataProviderError,
    ConnectionError,
    DataNotFoundError,
)
from utils.logging import get_logger

logger = get_logger(__name__)

# 导入akshare
try:
    import akshare as ak
except ImportError:
    ak = None
    logger.warning("akshare未安装，请运行: pip install akshare")


class AKShareProvider(BaseDataProvider):
    """AKShare数据源提供者（免费，无需token）"""

    def __init__(self):
        super().__init__("AKShare")
        self._requires_token = False

    def connect(self) -> bool:
        """建立连接（AKShare无需连接）"""
        if ak is None:
            raise ImportError("akshare未安装")

        self._is_connected = True
        logger.info("AKShare就绪")
        return True

    def disconnect(self):
        """断开连接"""
        self._is_connected = False
        logger.info("AKShare已断开")

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
        # 标准化代码
        symbol = self.normalize_symbol(symbol)
        original_symbol = symbol.replace(".", "")

        # AKShare的复权参数: ""-不复权, "qfq"-前复权, "hfq"-后复权
        adjust_param = adjust if adjust in ["qfq", "hfq"] else ""

        # 日期格式转换
        start_date, end_date = self.validate_date_range(start_date, end_date)
        start_date_str = start_date.replace("-", "")
        end_date_str = end_date.replace("-", "")

        try:
            # AKShare stock_zh_a_hist接口
            df = ak.stock_zh_a_hist(
                symbol=original_symbol,
                period="daily",
                start_date=start_date_str,
                end_date=end_date_str,
                adjust=adjust_param,
            )

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到数据: {symbol}")

            # AKShare返回的列名是中文，需要转换
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            })

            # 添加symbol列
            df["symbol"] = symbol

            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"])

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
        original_symbol = symbol.replace(".", "")
        trade_date_str = trade_date.replace("-", "")

        # AKShare的分钟周期: 1, 5, 15, 30, 60
        period_map = {1: "1", 5: "5", 15: "15", 30: "30", 60: "60"}
        adjust_period = period_map.get(period, "1")

        try:
            # AKShare stock_zh_a_hist_min_em接口
            df = ak.stock_zh_a_hist_min_em(
                symbol=original_symbol,
                period=adjust_period,
                adjust="",  # 分钟线通常不复权
                start_date=trade_date_str,
                end_date=trade_date_str,
            )

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到分钟线数据: {symbol} {trade_date}")

            # 转换列名
            df = df.rename(columns={
                "时间": "time",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
            })

            df["symbol"] = symbol

            return df[["symbol", "time", "open", "high", "low", "close", "volume", "amount"]]

        except Exception as e:
            logger.error(f"获取分钟线数据失败: {e}")
            raise DataProviderError(f"获取分钟线数据失败: {e}")

    def get_realtime_quote(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情"""
        try:
            # AKShare stock_zh_a_spot_em接口（全部A股实时行情）
            df = ak.stock_zh_a_spot_em()

            if df is None or df.empty:
                raise DataNotFoundError("未获取到实时行情")

            # 转换列名
            df = df.rename(columns={
                "代码": "symbol",
                "最新价": "price",
                "买一": "bid_price",
                "卖一": "ask_price",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "change_pct",
                "涨跌额": "change_amount",
                "今开": "open",
                "昨收": "pre_close",
                "最高": "high",
                "最低": "low",
            })

            # 过滤指定股票
            if symbols:
                normalized_symbols = [s.replace(".", "") for s in symbols]
                df = df[df["symbol"].isin(normalized_symbols)]

            return df[["symbol", "price", "bid_price", "ask_price", "volume",
                      "amount", "change_pct", "open", "high", "low"]]

        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            raise DataProviderError(f"获取实时行情失败: {e}")

    # ====================
    # 基础信息数据
    # ====================

    def get_stock_list(self, market: Optional[str] = None) -> pd.DataFrame:
        """获取股票列表"""
        try:
            # AKShare stock_info_a_code_name接口
            df = ak.stock_info_a_code_name()

            if df is None or df.empty:
                raise DataNotFoundError("未获取到股票列表")

            # 转换列名
            df = df.rename(columns={
                "code": "symbol",
                "name": "name",
            })

            # 添加市场后缀
            def add_market_suffix(code):
                if code.startswith("6"):
                    return f"{code}.SH"
                elif code.startswith(("0", "3")):
                    return f"{code}.SZ"
                return code

            df["symbol"] = df["symbol"].apply(add_market_suffix)

            # 添加市场列
            def get_market(code):
                if code.startswith("6"):
                    return "SH"
                elif code.startswith(("0", "3")):
                    return "SZ"
                return "UNKNOWN"

            df["market"] = df["symbol"].str.split(".").str[1]

            # 过滤市场
            if market:
                df = df[df["market"] == market]

            return df

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            raise DataProviderError(f"获取股票列表失败: {e}")

    def get_index_list(self) -> pd.DataFrame:
        """获取指数列表"""
        try:
            # AKShare stock_zh_index_spot接口
            df = ak.stock_zh_index_spot_em()

            if df is None or df.empty:
                raise DataNotFoundError("未获取到指数列表")

            # 转换列名
            df = df.rename(columns={
                "代码": "symbol",
                "名称": "name",
            })

            # 判断市场
            def get_market(symbol):
                if symbol.startswith("000") or symbol.startswith("688"):
                    return "SH"
                elif symbol.startswith("399"):
                    return "SZ"
                return "UNKNOWN"

            df["market"] = df["symbol"].apply(get_market)

            return df[["symbol", "name", "market"]]

        except Exception as e:
            logger.error(f"获取指数列表失败: {e}")
            raise DataProviderError(f"获取指数列表失败: {e}")

    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票基本信息"""
        try:
            original_symbol = symbol.replace(".", "")

            # AKShare stock_individual_info_em接口
            df = ak.stock_individual_info_em(symbol=original_symbol)

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到股票信息: {symbol}")

            # 转换为字典
            info_dict = dict(zip(df["item"], df["value"]))

            # 添加symbol
            info_dict["symbol"] = symbol

            return info_dict

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
            original_symbol = symbol.replace(".", "")

            # AKShare stock_profit_sheet_by_report_em接口（利润表）
            df = ak.stock_profit_sheet_by_report_em(symbol=original_symbol)

            if df is None or df.empty:
                raise DataNotFoundError(f"未找到财务数据: {symbol}")

            # 日期过滤
            if start_date:
                start_date = pd.to_datetime(start_date)
                df["报告期"] = pd.to_datetime(df["报告期"])
                df = df[df["报告期"] >= start_date]

            if end_date:
                end_date = pd.to_datetime(end_date)
                df = df[df["报告期"] <= end_date]

            return df

        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            raise DataProviderError(f"获取财务数据失败: {e}")


__all__ = ['AKShareProvider']
