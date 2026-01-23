"""
quantA策略使用指南示例
展示如何使用quantA进行策略回测
"""

import sys
sys.path.insert(0, '/home/rowan/Projects/quantA')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import BuyAndHoldStrategy, MovingAverageCrossStrategy
from backtest.engine.strategies import (
    BollingerBandsStrategy,
    MACDStrategy,
    RSIStrategy,
    BreakoutStrategy,
)
from backtest.engine.portfolio import Portfolio
from backtest.engine.execution import SimulationExecutionHandler
from backtest.engine.data_handler import SimpleDataHandler
from utils import logger


def generate_sample_data(symbol: str = "600519.SH", days: int = 100):
    """
    生成示例数据

    Args:
        symbol: 股票代码
        days: 天数

    Returns:
        DataFrame: 包含OHLCV数据的DataFrame
    """
    logger.info(f"生成{days}天的示例数据...")

    # 生成日期范围
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

    # 生成随机价格数据（模拟真实股价波动）
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # 日收益率
    prices = 100 * np.exp(np.cumsum(returns))  # 价格序列

    # 生成OHLCV
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    })

    # 确保high >= close >= low
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, days))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, days))

    logger.info(f"数据生成完成: {len(data)}行")
    return data


def example_1_buy_and_hold():
    """示例1: 买入持有策略"""
    logger.info("=" * 60)
    logger.info("示例1: 买入持有策略")
    logger.info("=" * 60)

    # 生成数据
    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=100)

    # 创建策略
    strategy = BuyAndHoldStrategy(symbol=symbol, quantity=1000)

    # 创建回测引擎
    engine = BacktestEngine(
        data={symbol: data},
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    results = engine.run()

    # 打印结果
    logger.info(f"\n回测结果:")
    logger.info(f"  总资金: {results['account']['total_value']:,.2f}")
    logger.info(f"  可用资金: {results['account']['cash']:,.2f}")
    logger.info(f"  持仓市值: {results['account'].get('position_value', 0):,.2f}")
    logger.info(f"  总收益率: {results['performance'].get('total_return_pct', 0):.2f}%")
    logger.info(f"  夏普比率: {results['performance'].get('sharpe_ratio', 0):.2f}")
    logger.info(f"  最大回撤: {results['performance'].get('max_drawdown', 0):.2%}")

    return results


def example_2_ma_cross():
    """示例2: 双均线交叉策略"""
    logger.info("\n" + "=" * 60)
    logger.info("示例2: 双均线交叉策略")
    logger.info("=" * 60)

    # 生成数据
    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=200)

    # 创建策略
    strategy = MovingAverageCrossStrategy(
        symbol=symbol,
        fast_period=5,
        slow_period=20,
        quantity=1000
    )

    # 创建回测引擎
    engine = BacktestEngine(
        data={symbol: data},
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    results = engine.run()

    # 打印结果
    logger.info(f"\n回测结果:")
    logger.info(f"  总资金: {results['account']['total_value']:,.2f}")
    logger.info(f"  总收益率: {results['performance'].get('total_return_pct', 0):.2f}%")
    logger.info(f"  夏普比率: {results['performance'].get('sharpe_ratio', 0):.2f}")
    logger.info(f"  最大回撤: {results['performance'].get('max_drawdown', 0):.2%}")

    return results


def example_3_bollinger_bands():
    """示例3: 布林带策略"""
    logger.info("\n" + "=" * 60)
    logger.info("示例3: 布林带策略")
    logger.info("=" * 60)

    # 生成数据
    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=100)

    # 创建策略
    strategy = BollingerBandsStrategy(
        symbol=symbol,
        period=20,
        std_dev=2.0,
        quantity=1000
    )

    # 创建回测引擎
    engine = BacktestEngine(
        data={symbol: data},
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    results = engine.run()

    # 打印结果
    logger.info(f"\n回测结果:")
    logger.info(f"  总资金: {results['account']['total_value']:,.2f}")
    logger.info(f"  总收益率: {results['performance'].get('total_return_pct', 0):.2f}%")

    return results


def example_4_macd():
    """示例4: MACD策略"""
    logger.info("\n" + "=" * 60)
    logger.info("示例4: MACD策略")
    logger.info("=" * 60)

    # 生成数据
    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=100)

    # 创建策略
    strategy = MACDStrategy(
        symbol=symbol,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        quantity=1000
    )

    # 创建回测引擎
    engine = BacktestEngine(
        data={symbol: data},
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    results = engine.run()

    # 打印结果
    logger.info(f"\n回测结果:")
    logger.info(f"  总资金: {results['account']['total_value']:,.2f}")
    logger.info(f"  总收益率: {results['performance'].get('total_return_pct', 0):.2f}%")

    return results


def example_5_rsi():
    """示例5: RSI策略"""
    logger.info("\n" + "=" * 60)
    logger.info("示例5: RSI策略")
    logger.info("=" * 60)

    # 生成数据
    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=100)

    # 创建策略
    strategy = RSIStrategy(
        symbol=symbol,
        period=14,
        oversold=30.0,
        overbought=70.0,
        quantity=1000
    )

    # 创建回测引擎
    engine = BacktestEngine(
        data={symbol: data},
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    results = engine.run()

    # 打印结果
    logger.info(f"\n回测结果:")
    logger.info(f"  总资金: {results['account']['total_value']:,.2f}")
    logger.info(f"  总收益率: {results['performance'].get('total_return_pct', 0):.2f}%")

    return results


def main():
    """主函数"""
    logger.info("quantA 策略使用指南")
    logger.info("=" * 60)

    # 运行所有示例
    try:
        example_1_buy_and_hold()
        example_2_ma_cross()
        example_3_bollinger_bands()
        example_4_macd()
        example_5_rsi()

        logger.info("\n" + "=" * 60)
        logger.info("所有示例运行完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"运行示例时出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()
