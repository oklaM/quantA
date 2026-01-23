"""
高级回测策略使用示例

展示如何使用quantA的多种回测策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine.backtest import BacktestEngine, run_backtest
from backtest.engine.strategies import (
    BollingerBandsStrategy,
    MACDStrategy,
    RSIStrategy,
    BreakoutStrategy,
    DualThrustStrategy,
    GridTradingStrategy,
    MomentumStrategy,
)
from backtest.engine.strategy import BuyAndHoldStrategy, MovingAverageCrossStrategy


def generate_sample_data(symbol: str, days: int = 250, start_price: float = 100.0):
    """
    生成示例数据

    Args:
        symbol: 股票代码
        days: 天数
        start_price: 起始价格

    Returns:
        DataFrame: 包含OHLCV数据
    """
    np.random.seed(42)

    # 生成价格序列（几何布朗运动）
    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * (1 + returns).cumprod()

    # 生成日期
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days,
        freq='D'
    )

    # 生成OHLC
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))

        data.append({
            'datetime': date,
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': np.random.randint(1000000, 10000000)
        })

    df = pd.DataFrame(data)
    return df


def print_backtest_results(results: dict, strategy_name: str):
    """打印回测结果"""
    print("\n" + "="*70)
    print(f"{strategy_name} - 回测结果")
    print("="*70)

    account = results['account']
    performance = results['performance']
    stats = results['stats']

    print(f"\n账户信息:")
    print(f"  初始资金: ¥{account['initial_cash']:,.2f}")
    print(f"  最终资产: ¥{account['total_value']:,.2f}")
    print(f"  总收益: ¥{account['total_value'] - account['initial_cash']:,.2f}")
    print(f"  总收益率: {account['total_return_pct']:.2f}%")
    print(f"  交易次数: {account['total_trades']}")

    print(f"\n绩效指标:")
    print(f"  夏普比率: {performance['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {performance['max_drawdown']:.2f}%")
    print(f"  波动率: {performance['volatility']:.2f}%")

    print(f"\n统计信息:")
    print(f"  总K线数: {stats['total_bars']}")
    print(f"  总订单数: {stats['total_orders']}")
    print(f"  成交数: {stats['total_fills']}")


def example_bollinger_bands():
    """示例1：布林带策略"""
    print("\n" + "="*70)
    print("示例1：布林带策略")
    print("="*70)

    # 生成数据
    symbol = "000001.SZ"
    df = generate_sample_data(symbol, days=250, start_price=100.0)

    # 创建策略
    strategy = BollingerBandsStrategy(
        symbol=symbol,
        period=20,
        std_dev=2.0,
        quantity=1000,
    )

    # 运行回测
    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
    )

    print_backtest_results(results, "布林带策略")

    return results


def example_macd():
    """示例2：MACD策略"""
    print("\n" + "="*70)
    print("示例2：MACD策略")
    print("="*70)

    symbol = "600519.SH"
    df = generate_sample_data(symbol, days=250, start_price=1800.0)

    strategy = MACDStrategy(
        symbol=symbol,
        fast_period=12,
        slow_period=26,
        signal_period=9,
        quantity=100,
    )

    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    print_backtest_results(results, "MACD策略")

    return results


def example_rsi():
    """示例3：RSI策略"""
    print("\n" + "="*70)
    print("示例3：RSI策略")
    print("="*70)

    symbol = "300001.SZ"
    df = generate_sample_data(symbol, days=250, start_price=50.0)

    strategy = RSIStrategy(
        symbol=symbol,
        period=14,
        oversold=30.0,
        overbought=70.0,
        quantity=2000,
    )

    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    print_backtest_results(results, "RSI策略")

    return results


def example_breakout():
    """示例4：突破策略"""
    print("\n" + "="*70)
    print("示例4：突破策略")
    print("="*70)

    symbol = "000002.SZ"
    df = generate_sample_data(symbol, days=250, start_price=20.0)

    strategy = BreakoutStrategy(
        symbol=symbol,
        period=20,
        quantity=5000,
    )

    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    print_backtest_results(results, "突破策略")

    return results


def example_dual_thrust():
    """示例5：Dual Thrust策略"""
    print("\n" + "="*70)
    print("示例5：Dual Thrust策略")
    print("="*70)

    symbol = "600000.SH"
    df = generate_sample_data(symbol, days=250, start_price=10.0)

    strategy = DualThrustStrategy(
        symbol=symbol,
        period=5,
        k1=0.5,
        k2=0.5,
        quantity=10000,
    )

    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    print_backtest_results(results, "Dual Thrust策略")

    return results


def example_grid_trading():
    """示例6：网格交易策略"""
    print("\n" + "="*70)
    print("示例6：网格交易策略")
    print("="*70)

    symbol = "510300.SH"  # 沪深300ETF
    df = generate_sample_data(symbol, days=250, start_price=4.0)

    # 使用中间价格作为基准
    base_price = df['close'].median()

    strategy = GridTradingStrategy(
        symbol=symbol,
        base_price=base_price,
        grid_count=10,
        grid_spacing=0.01,  # 1%
        grid_quantity=10000,
    )

    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    print_backtest_results(results, "网格交易策略")

    return results


def example_momentum():
    """示例7：动量策略"""
    print("\n" + "="*70)
    print("示例7：动量策略")
    print("="*70)

    symbol = "600036.SH"
    df = generate_sample_data(symbol, days=250, start_price=35.0)

    strategy = MomentumStrategy(
        symbol=symbol,
        lookback=20,
        momentum_threshold=0.02,
        quantity=2000,
    )

    data = {symbol: df}
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    print_backtest_results(results, "动量策略")

    return results


def example_strategy_comparison():
    """示例8：策略对比"""
    print("\n" + "="*70)
    print("示例8：多策略对比")
    print("="*70)

    symbol = "000001.SZ"
    df = generate_sample_data(symbol, days=250, start_price=100.0)
    data = {symbol: df}

    # 定义多个策略
    strategies = {
        "买入持有": BuyAndHoldStrategy(symbol=symbol, quantity=1000),
        "双均线": MovingAverageCrossStrategy(symbol=symbol, quantity=1000),
        "布林带": BollingerBandsStrategy(symbol=symbol, quantity=1000),
        "MACD": MACDStrategy(symbol=symbol, quantity=1000),
        "RSI": RSIStrategy(symbol=symbol, quantity=1000),
        "突破": BreakoutStrategy(symbol=symbol, quantity=1000),
        "动量": MomentumStrategy(symbol=symbol, quantity=1000),
    }

    # 运行所有策略
    results_summary = []

    for name, strategy in strategies.items():
        print(f"\n运行策略: {name}...")
        results = run_backtest(
            data=data,
            strategy=strategy,
            initial_cash=1000000.0,
            commission_rate=0.0003,
            slippage_rate=0.0001,
        )

        results_summary.append({
            '策略': name,
            '总收益率': results['account']['total_return_pct'],
            '夏普比率': results['performance']['sharpe_ratio'],
            '最大回撤': results['performance']['max_drawdown'],
            '交易次数': results['account']['total_trades'],
        })

    # 创建对比表
    summary_df = pd.DataFrame(results_summary)

    print("\n" + "="*70)
    print("策略对比结果")
    print("="*70)
    print(summary_df.to_string(index=False))

    # 找出最佳策略
    best_return = summary_df.loc[summary_df['总收益率'].idxmax()]
    best_sharpe = summary_df.loc[summary_df['夏普比率'].idxmax()]

    print(f"\n最高收益率策略: {best_return['策略']} ({best_return['总收益率']:.2f}%)")
    print(f"最高夏普比率策略: {best_sharpe['策略']} ({best_sharpe['夏普比率']:.2f})")

    return summary_df


def example_parameter_optimization():
    """示例9：参数优化（简单网格搜索）"""
    print("\n" + "="*70)
    print("示例9：双均线策略参数优化")
    print("="*70)

    symbol = "000001.SZ"
    df = generate_sample_data(symbol, days=250, start_price=100.0)
    data = {symbol: df}

    # 参数网格
    fast_periods = [5, 10, 15]
    slow_periods = [20, 30, 40]

    results_list = []

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue

            print(f"\n测试参数: 快线={fast}, 慢线={slow}")

            strategy = MovingAverageCrossStrategy(
                symbol=symbol,
                fast_period=fast,
                slow_period=slow,
                quantity=1000,
            )

            results = run_backtest(
                data=data,
                strategy=strategy,
                initial_cash=1000000.0,
            )

            results_list.append({
                '快线': fast,
                '慢线': slow,
                '收益率': results['account']['total_return_pct'],
                '夏普比率': results['performance']['sharpe_ratio'],
                '最大回撤': results['performance']['max_drawdown'],
            })

    # 创建结果DataFrame
    results_df = pd.DataFrame(results_list)

    print("\n" + "="*70)
    print("参数优化结果")
    print("="*70)
    print(results_df.to_string(index=False))

    # 找出最佳参数
    best_params = results_df.loc[results_df['收益率'].idxmax()]
    print(f"\n最佳参数组合:")
    print(f"  快线周期: {best_params['快线']}")
    print(f"  慢线周期: {best_params['慢线']}")
    print(f"  收益率: {best_params['收益率']:.2f}%")
    print(f"  夏普比率: {best_params['夏普比率']:.2f}")

    return results_df


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 高级回测策略示例")
    print("="*70)

    try:
        # 运行各个示例
        # example_bollinger_bands()
        # example_macd()
        # example_rsi()
        # example_breakout()
        # example_dual_thrust()
        # example_grid_trading()
        # example_momentum()

        # 策略对比
        example_strategy_comparison()

        # 参数优化
        # example_parameter_optimization()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
