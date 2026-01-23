"""
通用辅助函数
"""

import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


def timing_decorator(func):
    """函数执行时间装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"{func.__name__} 执行时间: {elapsed_time:.2f}秒")
        return result

    return wrapper


def format_number(
    value: Union[int, float],
    decimal_places: int = 2,
    percentage: bool = False,
) -> str:
    """
    格式化数字

    Args:
        value: 数值
        decimal_places: 小数位数
        percentage: 是否显示为百分比

    Returns:
        格式化后的字符串
    """
    if pd.isna(value):
        return "N/A"

    if percentage:
        return f"{value * 100:.{decimal_places}f}%"
    else:
        return f"{value:,.{decimal_places}f}"


def format_money(value: Union[int, float], unit: str = "万") -> str:
    """
    格式化金额

    Args:
        value: 金额
        unit: 单位 ("万", "亿")

    Returns:
        格式化后的字符串
    """
    if pd.isna(value):
        return "N/A"

    if unit == "万":
        return f"{value / 10000:,.2f}万"
    elif unit == "亿":
        return f"{value / 100000000:,.2f}亿"
    else:
        return f"{value:,.2f}"


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    计算收益率

    Args:
        prices: 价格序列

    Returns:
        收益率序列
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    计算对数收益率

    Args:
        prices: 价格序列

    Returns:
        对数收益率序列
    """
    return np.log(prices / prices.shift(1))


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    计算累计收益率

    Args:
        returns: 收益率序列

    Returns:
        累计收益率序列
    """
    return (1 + returns).cumprod() - 1


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """
    计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年交易周期数

    Returns:
        夏普比率
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """
    计算索提诺比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（年化）
        periods_per_year: 每年交易周期数

    Returns:
        索提诺比率
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    计算最大回撤

    Args:
        prices: 价格序列

    Returns:
        最大回撤
    """
    cumulative = (1 + prices.pct_change()).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    计算波动率（年化）

    Args:
        returns: 收益率序列
        periods_per_year: 每年交易周期数

    Returns:
        年化波动率
    """
    return returns.std() * np.sqrt(periods_per_year)


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    将列表分块

    Args:
        lst: 原始列表
        chunk_size: 块大小

    Returns:
        分块后的列表
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个字典（后面的覆盖前面的）

    Args:
        *dicts: 多个字典

    Returns:
        合并后的字典
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_divide(numerator: Union[int, float], denominator: Union[int, float]) -> float:
    """
    安全除法（避免除零错误）

    Args:
        numerator: 分子
        denominator: 分母

    Returns:
        除法结果，分母为0时返回0
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内

    Args:
        value: 原始值
        min_val: 最小值
        max_val: 最大值

    Returns:
        限制后的值
    """
    return max(min_val, min(max_val, value))


def normalize(
    data: pd.Series,
    method: str = "minmax",
) -> pd.Series:
    """
    数据归一化

    Args:
        data: 数据序列
        method: 归一化方法 (minmax, zscore, rank)

    Returns:
        归一化后的序列
    """
    if method == "minmax":
        min_val = data.min()
        max_val = data.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(data), index=data.index)
        return (data - min_val) / (max_val - min_val)

    elif method == "zscore":
        mean = data.mean()
        std = data.std()
        if std == 0:
            return pd.Series([0] * len(data), index=data.index)
        return (data - mean) / std

    elif method == "rank":
        return data.rank(pct=True)

    else:
        raise ValueError(f"未知的归一化方法: {method}")


def retry_on_exception(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
):
    """
    重试装饰器

    Args:
        max_retries: 最大重试次数
        exceptions: 需要重试的异常类型
        delay: 重试延迟（秒）
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"{func.__name__} 失败 (尝试 {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} 失败，已达最大重试次数")

            raise last_exception

        return wrapper

    return decorator


__all__ = [
    "timing_decorator",
    "format_number",
    "format_money",
    "calculate_returns",
    "calculate_log_returns",
    "calculate_cumulative_returns",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_volatility",
    "chunk_list",
    "merge_dicts",
    "safe_divide",
    "clamp",
    "normalize",
    "retry_on_exception",
]
