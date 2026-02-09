"""
时间工具模块
提供A股交易日历、时间解析等功能
"""

from datetime import datetime, time, timedelta
from typing import List, Optional, Set, Union

import pandas as pd

from config.settings import market
from utils.logging import get_logger

logger = get_logger(__name__)


# A股交易日历缓存
_trading_days_cache: Optional[Set[pd.Timestamp]] = None


def is_trading_day(date: Union[pd.Timestamp, datetime, str]) -> bool:
    """
    判断是否为交易日

    Args:
        date: 日期

    Returns:
        是否为交易日
    """
    dt = pd.to_datetime(date).date()

    # 简单判断：排除周末
    # TODO: 集成真实的交易日历数据（可从Tushare获取）
    weekday = dt.weekday()
    if weekday >= 5:  # 周六、周日
        return False

    # TODO: 排除节假日
    # 需要维护一个节假日列表或从API获取

    return True


def is_trading_time(dt: Optional[datetime] = None) -> bool:
    """
    判断当前时间是否在交易时段内

    Args:
        dt: 时间，默认为当前时间

    Returns:
        是否在交易时段
    """
    if dt is None:
        dt = datetime.now()

    # 检查是否为交易日
    if not is_trading_day(dt):
        return False

    # 检查是否在交易时间
    current_time = dt.time()

    morning_start = pd.to_datetime(market.MORNING_START).time()
    morning_end = pd.to_datetime(market.MORNING_END).time()
    afternoon_start = pd.to_datetime(market.AFTERNOON_START).time()
    afternoon_end = pd.to_datetime(market.AFTERNOON_END).time()

    return (
        morning_start <= current_time <= morning_end
        or afternoon_start <= current_time <= afternoon_end
    )


def get_trading_days(
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
) -> List[pd.Timestamp]:
    """
    获取日期范围内的所有交易日

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        交易日列表
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    trading_days = []
    current = start

    while current <= end:
        if is_trading_day(current):
            trading_days.append(current)
        current += timedelta(days=1)

    return trading_days


def get_previous_trading_day(
    date: Union[str, datetime, pd.Timestamp],
    n: int = 1,
) -> pd.Timestamp:
    """
    获取前n个交易日

    Args:
        date: 基准日期
        n: 向前多少个交易日

    Returns:
        交易日
    """
    dt = pd.to_datetime(date)
    count = 0

    while count < n:
        dt -= timedelta(days=1)
        if is_trading_day(dt):
            count += 1

    return dt


def get_next_trading_day(
    date: Union[str, datetime, pd.Timestamp],
    n: int = 1,
) -> pd.Timestamp:
    """
    获取后n个交易日

    Args:
        date: 基准日期
        n: 向后多少个交易日

    Returns:
        交易日
    """
    dt = pd.to_datetime(date)
    count = 0

    while count < n:
        dt += timedelta(days=1)
        if is_trading_day(dt):
            count += 1

    return dt


def get_trading_sessions(date: Union[str, datetime, pd.Timestamp]) -> List[tuple]:
    """
    获取指定日期的交易时段

    Args:
        date: 日期

    Returns:
        [(开始时间, 结束时间), ...]
    """
    dt = pd.to_datetime(date).date()

    if not is_trading_day(dt):
        return []

    morning_start = datetime.combine(dt, pd.to_datetime(market.MORNING_START).time())
    morning_end = datetime.combine(dt, pd.to_datetime(market.MORNING_END).time())
    afternoon_start = datetime.combine(
        dt, pd.to_datetime(market.AFTERNOON_START).time()
    )
    afternoon_end = datetime.combine(dt, pd.to_datetime(market.AFTERNOON_END).time())

    return [
        (morning_start, morning_end),
        (afternoon_start, afternoon_end),
    ]


def get_current_trading_session(dt: Optional[datetime] = None) -> Optional[tuple]:
    """
    获取当前所在的交易时段

    Args:
        dt: 时间，默认为当前时间

    Returns:
        (开始时间, 结束时间) 或 None
    """
    if dt is None:
        dt = datetime.now()

    sessions = get_trading_sessions(dt)

    for start, end in sessions:
        if start <= dt <= end:
            return (start, end)

    return None


def time_to_seconds(t: Union[time, str]) -> int:
    """
    将时间转换为当天的秒数

    Args:
        t: 时间对象或字符串

    Returns:
        秒数
    """
    if isinstance(t, str):
        t = pd.to_datetime(t).time()

    return t.hour * 3600 + t.minute * 60 + t.second


def seconds_to_time(seconds: int) -> time:
    """
    将秒数转换为时间对象

    Args:
        seconds: 秒数

    Returns:
        时间对象
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return time(hour=hours, minute=minutes, second=secs)


def parse_period(period: str) -> pd.Timedelta:
    """
    解析时间周期字符串

    Args:
        period: 时间周期，如 '1min', '5min', '1h', '1d'

    Returns:
        Timedelta对象
    """
    # 映射常见周期
    period_map = {
        "1min": pd.Timedelta(minutes=1),
        "5min": pd.Timedelta(minutes=5),
        "15min": pd.Timedelta(minutes=15),
        "30min": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "1w": pd.Timedelta(weeks=1),
        "1M": pd.Timedelta(days=30),  # 近似
    }

    if period in period_map:
        return period_map[period]

    # 尝试pandas解析
    try:
        return pd.Timedelta(period)
    except Exception as e:
        logger.warning(f"无法解析周期 '{period}': {e}")
        return pd.Timedelta(minutes=1)


# A股节假日列表（需要定期更新）
# TODO: 从Tushare等API自动获取
HOLIDAYS_2024 = [
    # 元旦
    "2024-01-01",
    # 春节
    "2024-02-10",
    "2024-02-11",
    "2024-02-12",
    "2024-02-13",
    "2024-02-14",
    "2024-02-15",
    "2024-02-16",
    "2024-02-17",
    # 清明节
    "2024-04-04",
    "2024-04-05",
    "2024-04-06",
    # 劳动节
    "2024-05-01",
    "2024-05-02",
    "2024-05-03",
    "2024-05-04",
    "2024-05-05",
    # 端午节
    "2024-06-10",
    # 中秋节
    "2024-09-15",
    "2024-09-16",
    "2024-09-17",
    # 国庆节
    "2024-10-01",
    "2024-10-02",
    "2024-10-03",
    "2024-10-04",
    "2024-10-05",
    "2024-10-06",
    "2024-10-07",
]


def is_holiday(date: Union[pd.Timestamp, datetime, str]) -> bool:
    """
    判断是否为节假日

    Args:
        date: 日期

    Returns:
        是否为节假日
    """
    dt = pd.to_datetime(date).date()
    date_str = dt.strftime("%Y-%m-%d")

    # 简单实现：检查是否在节假日列表中
    # 生产环境应该使用更完整的数据源
    return date_str in HOLIDAYS_2024


__all__ = [
    "is_trading_day",
    "is_trading_time",
    "get_trading_days",
    "get_previous_trading_day",
    "get_next_trading_day",
    "get_trading_sessions",
    "get_current_trading_session",
    "time_to_seconds",
    "seconds_to_time",
    "parse_period",
    "is_holiday",
]
