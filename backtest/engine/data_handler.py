"""
数据处理器
为回测引擎提供数据
"""

from datetime import datetime
from typing import Dict, Iterator, List, Optional

import pandas as pd

from backtest.engine.event_engine import BarEvent, DataHandler
from utils.logging import get_logger

logger = get_logger(__name__)


class SimpleDataHandler(DataHandler):
    """简单数据处理器"""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Args:
            data: {symbol: DataFrame} 格式的数据
                  DataFrame必须包含: date, open, high, low, close, volume
        """
        self.data = data
        self.symbols = list(data.keys())

        # 数据迭代器
        self._iterators: Dict[str, Iterator] = {}

        # 当前K线
        self._current_bars: Dict[str, BarEvent] = {}

        # 数据索引
        self._data_indices: Dict[str, int] = {}

        self.reset()

    def reset(self):
        """重置数据处理器"""
        self._iterators = {}
        self._current_bars = {}
        self._data_indices = {}

        for symbol in self.symbols:
            df = self.data[symbol]

            # 确保日期列是datetime类型
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            self._iterators[symbol] = iter(df.to_dict("records"))
            self._data_indices[symbol] = 0

    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取下一根K线"""
        try:
            bar_data = next(self._iterators[symbol])
            self._data_indices[symbol] += 1

            # 解析日期
            if "date" in bar_data:
                bar_datetime = pd.to_datetime(bar_data["date"])
            elif "datetime" in bar_data:
                bar_datetime = pd.to_datetime(bar_data["datetime"])
            else:
                bar_datetime = datetime.now()

            event = BarEvent(
                datetime=bar_datetime,
                symbol=symbol,
                open=float(bar_data["open"]),
                high=float(bar_data["high"]),
                low=float(bar_data["low"]),
                close=float(bar_data["close"]),
                volume=int(bar_data["volume"]),
            )

            # 保存前一收盘价（用于涨跌停计算）
            event.data["prev_close"] = self._get_prev_close(symbol)

            self._current_bars[symbol] = event
            return event

        except (StopIteration, KeyError):
            return None

    def get_current_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取当前K线"""
        return self._current_bars.get(symbol)

    def _get_prev_close(self, symbol: str) -> Optional[float]:
        """获取前一根K线的收盘价"""
        if symbol in self._current_bars:
            return self._current_bars[symbol].close
        return None


class CSVDataHandler(DataHandler):
    """CSV文件数据处理器"""

    def __init__(self, csv_paths: Dict[str, str]):
        """
        Args:
            csv_paths: {symbol: csv_file_path}
        """
        self.data = {}

        for symbol, path in csv_paths.items():
            self.data[symbol] = pd.read_csv(path)

        self._handler = SimpleDataHandler(self.data)

    def reset(self):
        """重置"""
        self._handler.reset()

    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取下一根K线"""
        return self._handler.get_next_bar(symbol)

    def get_current_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取当前K线"""
        return self._handler.get_current_bar(symbol)


class DatabaseDataHandler(DataHandler):
    """数据库数据处理器"""

    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        """
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data: Dict[str, pd.DataFrame] = {}

        self._load_data()

    def _load_data(self):
        """从数据库加载数据"""
        try:
            from data.market.storage.timeseries_db import get_timeseries_db

            db = get_timeseries_db()
            db.connect()

            for symbol in self.symbols:
                df = db.read(
                    table_name="daily_bar",
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                )

                if not df.empty:
                    self.data[symbol] = df
                    logger.info(f"加载{symbol}数据: {len(df)}条记录")
                else:
                    logger.warning(f"未找到{symbol}的数据")

        except Exception as e:
            logger.error(f"加载数据失败: {e}")

    def reset(self):
        """重置"""
        self._handler = SimpleDataHandler(self.data)
        self._handler.reset()

    def get_next_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取下一根K线"""
        if not hasattr(self, "_handler"):
            self.reset()
        return self._handler.get_next_bar(symbol)

    def get_current_bar(self, symbol: str) -> Optional[BarEvent]:
        """获取当前K线"""
        if not hasattr(self, "_handler"):
            self.reset()
        return self._handler.get_current_bar(symbol)


__all__ = [
    "SimpleDataHandler",
    "CSVDataHandler",
    "DatabaseDataHandler",
]
