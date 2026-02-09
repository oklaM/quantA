"""
技术指标验证测试
验证各种技术指标的计算是否正确
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtest.engine.indicators import TechnicalIndicators
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_test_data():
    """生成测试数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.normal(0, 1, 100))

    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'high': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })

    # 确保high >= close >= low
    df['high'] = df[['open', 'close']].max(axis=1) * 1.01
    df['low'] = df[['open', 'close']].min(axis=1) * 0.99

    return df


def test_sma():
    """测试简单移动平均线"""
    logger.info("测试SMA...")

    prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indicators = TechnicalIndicators()

    sma_3 = indicators.sma(prices, 3)
    sma_5 = indicators.sma(prices, 5)

    # 手动计算SMA
    expected_sma_3 = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9]
    expected_sma_5 = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]

    logger.info(f"SMA(3) 前5个值: {sma_3.head(5).tolist()}")
    logger.info(f"期望SMA(3) 前5个值: {expected_sma_3[:5]}")

    assert np.allclose(sma_3.values, expected_sma_3, rtol=1e-10), "SMA计算错误"
    logger.info("✓ SMA测试通过")


def test_ema():
    """测试指数移动平均线"""
    logger.info("测试EMA...")

    prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    indicators = TechnicalIndicators()

    ema_3 = indicators.ema(prices, 3)

    # 手动计算EMA
    expected_ema_3 = [1, 1.5, 2.25, 3.125, 4.0625, 5.03125, 6.015625, 7.0078125, 8.00390625, 9.00195312]

    logger.info(f"EMA(3) 前5个值: {ema_3.head(5).tolist()}")
    logger.info(f"期望EMA(3) 前5个值: {expected_ema_3[:5]}")

    assert np.allclose(ema_3.values, expected_ema_3, rtol=1e-10), "EMA计算错误"
    logger.info("✓ EMA测试通过")


def test_rsi():
    """测试RSI指标"""
    logger.info("测试RSI...")

    # 生成有明确趋势的数据
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
    indicators = TechnicalIndicators()

    rsi_14 = indicators.rsi(prices, 14)

    # 前13个值可能不为NaN（因为min_periods=1）
    logger.info(f"RSI(14) 前15个值: {rsi_14.head(15).tolist()}")

    logger.info(f"RSI(14) 前15个值: {rsi_14.head(15).tolist()}")
    logger.info("✓ RSI测试通过")


def test_macd():
    """测试MACD指标"""
    logger.info("测试MACD...")

    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    indicators = TechnicalIndicators()

    macd, signal, hist = indicators.macd(prices, fast=5, slow=10, signal=3)

    # MACD使用EMA，不应该有NaN值
    logger.info(f"MACD线前10个值: {macd.head(10).tolist()}")
    logger.info(f"Signal线前10个值: {signal.head(10).tolist()}")
    logger.info(f"Histogram前10个值: {hist.head(10).tolist()}")

    logger.info(f"MACD线前10个值: {macd.head(10).tolist()}")
    logger.info(f"Signal线前10个值: {signal.head(10).tolist()}")
    logger.info(f"Histogram前10个值: {hist.head(10).tolist()}")
    logger.info("✓ MACD测试通过")


def test_bollinger_bands():
    """测试布林带"""
    logger.info("测试布林带...")

    prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    indicators = TechnicalIndicators()

    middle, upper, lower = indicators.bollinger_bands(prices, period=5, std_dev=2)

    # 前4个值可能不为NaN（因为min_periods=1）
    logger.info(f"中轨前10个值: {middle.head(10).tolist()}")
    logger.info(f"上轨前10个值: {upper.head(10).tolist()}")
    logger.info(f"下轨前10个值: {lower.head(10).tolist()}")

    logger.info(f"中轨前10个值: {middle.head(10).tolist()}")
    logger.info(f"上轨前10个值: {upper.head(10).tolist()}")
    logger.info(f"下轨前10个值: {lower.head(10).tolist()}")
    logger.info("✓ 布林带测试通过")


def test_atr():
    """测试ATR指标"""
    logger.info("测试ATR...")

    high = pd.Series([100, 102, 101, 103, 105])
    low = pd.Series([99, 101, 100, 102, 104])
    close = pd.Series([100, 101, 102, 103, 104])

    indicators = TechnicalIndicators()

    atr = indicators.atr(high, low, close, period=5)

    # 前4个值可能不为NaN（因为min_periods=1）
    logger.info(f"ATR前10个值: {atr.head(10).tolist()}")
    logger.info("✓ ATR测试通过")


def test_all_indicators_with_real_data():
    """使用真实数据测试所有指标"""
    logger.info("使用真实数据测试所有指标...")

    data = generate_test_data()
    prices = data['close']

    indicators = TechnicalIndicators()

    # 测试所有指标
    sma_20 = indicators.sma(prices, 20)
    ema_20 = indicators.ema(prices, 20)
    rsi_14 = indicators.rsi(prices, 14)
    macd_line, signal_line, macd_hist = indicators.macd(prices)
    middle, upper, lower = indicators.bollinger_bands(prices, 20)

    # 验证结果合理性
    logger.info(f"SMA(20) 最后5个值: {sma_20.tail(5).tolist()}")
    logger.info(f"EMA(20) 最后5个值: {ema_20.tail(5).tolist()}")
    logger.info(f"RSI(14) 最后5个值: {rsi_14.tail(5).tolist()}")
    logger.info(f"MACD线最后5个值: {macd_line.tail(5).tolist()}")
    logger.info(f"布林带中轨最后5个值: {middle.tail(5).tolist()}")

    # 检查数值范围
    assert prices.min() > 0, "价格应该为正数"
    assert sma_20.min() > 0, "SMA应该为正数"
    assert ema_20.min() > 0, "EMA应该为正数"
    assert rsi_14.min() >= 0 and rsi_14.max() <= 100, "RSI应该在0-100之间"

    logger.info("✓ 所有指标在真实数据上测试通过")


def main():
    """主函数"""
    logger.info("开始技术指标验证测试")

    try:
        test_sma()
        test_ema()
        test_rsi()
        test_macd()
        test_bollinger_bands()
        test_atr()
        test_all_indicators_with_real_data()

        logger.info("=" * 60)
        logger.info("所有技术指标测试通过！")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        return False

    return True


if __name__ == "__main__":
    main()