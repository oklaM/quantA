"""
技术指标计算模块
实现常用的金融技术指标
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """技术指标计算类"""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)

        Args:
            series: 价格序列
            period: 周期

        Returns:
            SMA序列
        """
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)

        Args:
            series: 价格序列
            period: 周期

        Returns:
            EMA序列
        """
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD指标 (Moving Average Convergence Divergence)

        Args:
            series: 价格序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        Returns:
            (MACD线, 信号线, 柱状图)
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        相对强弱指标 (Relative Strength Index)

        Args:
            series: 价格序列
            period: 周期

        Returns:
            RSI序列 (0-100)
        """
        delta = series.diff()

        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def kdj(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        n: int = 9,
        m1: int = 3,
        m2: int = 3,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        KDJ指标 (Stochastic Oscillator)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            n: RSV周期
            m1: K值平滑周期
            m2: D值平滑周期

        Returns:
            (K线, D线, J线)
        """
        low_n = low.rolling(window=n, min_periods=1).min()
        high_n = high.rolling(window=n, min_periods=1).max()

        rsv = (close - low_n) / (high_n - low_n) * 100

        k = rsv.ewm(com=m1 - 1, adjust=False).mean()
        d = k.ewm(com=m2 - 1, adjust=False).mean()
        j = 3 * k - 2 * d

        return k, d, j

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        布林带 (Bollinger Bands)

        Args:
            series: 价格序列
            period: 周期
            std_dev: 标准差倍数

        Returns:
            (上轨, 中轨, 下轨)
        """
        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        真实波动幅度 (Average True Range)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期

        Returns:
            ATR序列
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period, min_periods=1).mean()

        return atr

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        能量潮指标 (On Balance Volume)

        Args:
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            OBV序列
        """
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    @staticmethod
    def wvad(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        威廉变异离散量 (Williams Variable Accumulation/Distribution)

        Args:
            open: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            WVAD序列
        """
        wvad = ((close - open) / (high - low)) * volume
        wvad_cumsum = wvad.cumsum()

        return wvad_cumsum

    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
    ) -> pd.Series:
        """
        顺势指标 (Commodity Channel Index)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期

        Returns:
            CCI序列
        """
        tp = (high + low + close) / 3
        ma_tp = tp.rolling(window=period, min_periods=1).mean()
        md = tp.rolling(window=period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
        )

        cci = (tp - ma_tp) / (0.015 * md)

        return cci

    @staticmethod
    def ao(
        high: pd.Series,
        low: pd.Series,
        fast: int = 5,
        slow: int = 34,
    ) -> pd.Series:
        """
        动量振荡指标 (Awesome Oscillator)

        Args:
            high: 最高价序列
            low: 最低价序列
            fast: 快线周期
            slow: 慢线周期

        Returns:
            AO序列
        """
        median_price = (high + low) / 2
        ao = (
            median_price.rolling(window=fast, min_periods=1).mean()
            - median_price.rolling(window=slow, min_periods=1).mean()
        )

        return ao

    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """
        动量指标 (Momentum)

        Args:
            series: 价格序列
            period: 周期

        Returns:
            动量序列
        """
        return series - series.shift(period)

    @staticmethod
    def roc(series: pd.Series, period: int = 12) -> pd.Series:
        """
        变动率指标 (Rate of Change)

        Args:
            series: 价格序列
            period: 周期

        Returns:
            ROC序列 (%)
        """
        roc = ((series - series.shift(period)) / series.shift(period)) * 100
        return roc

    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        威廉指标 (Williams %R)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期

        Returns:
            威廉%R序列 (-100到0)
        """
        high_n = high.rolling(window=period, min_periods=1).max()
        low_n = low.rolling(window=period, min_periods=1).min()

        willr = (high_n - close) / (high_n - low_n) * (-100)

        return willr


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    为DataFrame添加常用技术指标

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        添加了技术指标的DataFrame
    """
    df = df.copy()

    # 确保必要的列存在
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # MA系列
    for period in [5, 10, 20, 60, 120]:
        df[f"ma{period}"] = TechnicalIndicators.sma(df["close"], period)

    # EMA系列
    for period in [12, 26]:
        df[f"ema{period}"] = TechnicalIndicators.ema(df["close"], period)

    # MACD
    macd, signal, hist = TechnicalIndicators.macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    # RSI
    df["rsi"] = TechnicalIndicators.rsi(df["close"])

    # KDJ
    k, d, j = TechnicalIndicators.kdj(df["high"], df["low"], df["close"])
    df["k"] = k
    df["d"] = d
    df["j"] = j

    # 布林带
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"])
    df["boll_upper"] = upper
    df["boll_middle"] = middle
    df["boll_lower"] = lower

    # ATR
    df["atr"] = TechnicalIndicators.atr(df["high"], df["low"], df["close"])

    # OBV
    df["obv"] = TechnicalIndicators.obv(df["close"], df["volume"])

    # CCI
    df["cci"] = TechnicalIndicators.cci(df["high"], df["low"], df["close"])

    # AO
    df["ao"] = TechnicalIndicators.ao(df["high"], df["low"])

    # ROC
    df["roc"] = TechnicalIndicators.roc(df["close"])

    # Williams %R
    df["willr"] = TechnicalIndicators.williams_r(df["high"], df["low"], df["close"])

    return df


__all__ = [
    "TechnicalIndicators",
    "add_indicators",
    "SMA",
    "EMA",
    "MACD",
    "RSI",
    "BOLLINGER_BANDS",
    "ATR",
    "KDJ",
    "STOCH",
    "OBV",
]


# Convenience function wrappers for backward compatibility
def SMA(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average wrapper"""
    return TechnicalIndicators.sma(series, period)


def EMA(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average wrapper"""
    return TechnicalIndicators.ema(series, period)


def MACD(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD wrapper"""
    return TechnicalIndicators.macd(series, fast, slow, signal)


def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI wrapper"""
    return TechnicalIndicators.rsi(series, period)


def BOLLINGER_BANDS(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands wrapper"""
    return TechnicalIndicators.bollinger_bands(series, period, std_dev)


def ATR(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR wrapper"""
    return TechnicalIndicators.atr(high, low, close, period)


def KDJ(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """KDJ wrapper"""
    return TechnicalIndicators.kdj(high, low, close, n, m1, m2)


def STOCH(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
    m: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic wrapper (alias for KDJ with simplified parameters)"""
    k, d, _ = TechnicalIndicators.kdj(high, low, close, n, m, 1)
    return k, d


def OBV(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV wrapper"""
    return TechnicalIndicators.obv(close, volume)
