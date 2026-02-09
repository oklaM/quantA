#!/usr/bin/env python3
"""
组合回测测试脚本
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 设置 Python 路径
sys.path.insert(0, os.path.abspath('.'))

def generate_sample_data(symbols: list, days: int = 500, start_price: float = 100.0):
    """
    生成多个股票的示例数据

    Args:
        symbols: 股票代码列表
        days: 天数
        start_price: 起始价格

    Returns:
        数据字典 {symbol: DataFrame}
    """
    np.random.seed(42)

    data_dict = {}

    for i, symbol in enumerate(symbols):
        # 每个股票有不同的收益率特征
        mean_return = 0.0005 + i * 0.0001
        volatility = 0.02 - i * 0.001

        returns = np.random.normal(mean_return, volatility, days)
        prices = start_price * (1 + returns).cumprod()

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days,
            freq='D'
        )

        data = []
        for date, close in zip(dates, prices):
            high = close * (1 + abs(np.random.normal(0, 0.015)))
            low = close * (1 - abs(np.random.normal(0, 0.015)))
            open_price = close * (1 + np.random.normal(0, 0.008))

            data.append({
                'datetime': date,
                'symbol': symbol,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })

        data_dict[symbol] = pd.DataFrame(data)

    return data_dict

def test_single_strategy_multi_asset():
    """测试单策略多资产组合回测"""
    print("="*70)
    print("测试1：单策略多资产组合回测")
    print("="*70)

    try:
        from backtest.engine.strategies import BuyAndHoldStrategy, MACDStrategy
        from backtest.portfolio import Portfolio, PortfolioBacktestEngine, StrategyAllocation

        # 生成数据
        print("\n生成数据...")
        symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
        data_dict = generate_sample_data(symbols, days=500)

        # 创建策略配置
        print("创建策略配置...")
        strategy_allocations = []
        weight_per_symbol = 1.0 / len(symbols)

        for symbol in symbols:
            # 每个股票一个独立的策略实例
            strategy_allocations.append(
                StrategyAllocation(
                    strategy=BuyAndHoldStrategy(symbol=symbol),
                    symbols=[symbol],
                    weight=weight_per_symbol,
                )
            )

        # 创建组合回测引擎
        print("创建组合回测引擎...")
        engine = PortfolioBacktestEngine(
            data_dict=data_dict,
            strategies=strategy_allocations,
            initial_cash=10000000.0,  # 1000万
            commission_rate=0.0003,
        )

        # 运行回测
        print("运行组合回测...")
        results = engine.run()

        # 打印结果
        print("\n" + "="*70)
        print("回测结果")
        print("="*70)
        print(f"总收益: {results['total_return']:.2%}")
        print(f"年化收益: {results['annual_return']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")

        # 各策略价值
        print("\n各策略最终价值:")
        for strategy_id, value in results['strategy_values'].items():
            strategy_name = strategy_allocations[strategy_id].strategy.__class__.__name__
            symbol = strategy_allocations[strategy_id].symbols[0]
            weight = strategy_allocations[strategy_id].weight
            print(f"  策略{strategy_id} ({strategy_name} - {symbol}): "
                  f"¥{value:,.2f} (权重={weight:.1%})")

        # 测试权益曲线和风险指标
        print("\n风险指标分析:")
        equity_curve = pd.Series(results['equity_curve'])

        # 计算风险指标
        returns = equity_curve.pct_change().fillna(0)
        total_return = results['total_return']
        max_drawdown = results['max_drawdown']
        volatility = returns.std() * np.sqrt(252)

        print(f"  总收益率: {total_return:.2%}")
        print(f"  年化波动率: {volatility:.2%}")
        print(f"  最大回撤: {max_drawdown:.2%}")
        print(f"  风险调整收益: {total_return / max_drawdown:.2f} (Calmar比率)")

        return True

    except Exception as e:
        print(f"❌ 单策略多资产测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_strategy_portfolio():
    """测试多策略组合回测"""
    print("\n" + "="*70)
    print("测试2：多策略组合回测")
    print("="*70)

    try:
        from backtest.engine.strategies import BuyAndHoldStrategy, MACDStrategy
        from backtest.portfolio import Portfolio, PortfolioBacktestEngine, StrategyAllocation

        # 生成数据
        print("\n生成数据...")
        symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000300.SH']
        data_dict = generate_sample_data(symbols, days=500)

        # 创建策略配置
        print("创建多策略配置...")

        # 策略1：买入持有（1只股票）
        strategy1_allocations = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbols[0]),
                symbols=[symbols[0]],
                weight=0.5,  # 50%权重
            )
        ]

        # 策略2：MACD（1只股票）
        strategy2_allocations = [
            StrategyAllocation(
                strategy=MACDStrategy(
                    symbol=symbols[1],
                    fast_period=12,
                    slow_period=26,
                ),
                symbols=[symbols[1]],
                weight=0.5,  # 50%权重
            )
        ]

        all_strategies = strategy1_allocations + strategy2_allocations

        # 创建组合回测引擎
        print("创建组合回测引擎...")
        engine = PortfolioBacktestEngine(
            data_dict=data_dict,
            strategies=all_strategies,
            initial_cash=10000000.0,
            commission_rate=0.0003,
        )

        # 运行回测
        print("运行多策略组合回测...")
        results = engine.run()

        # 打印结果
        print("\n" + "="*70)
        print("多策略组合回测结果")
        print("="*70)
        print(f"总收益: {results['total_return']:.2%}")
        print(f"年化收益: {results['annual_return']:.2%}")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")

        # 各策略表现
        print("\n各策略表现:")
        for strategy_id, value in results['strategy_values'].items():
            strategy_alloc = all_strategies[strategy_id]
            strategy_name = strategy_alloc.strategy.__class__.__name__
            symbol = strategy_alloc.symbols[0]
            initial_value = 10000000.0 * strategy_alloc.weight
            strategy_return = (value - initial_value) / initial_value

            print(f"  策略{strategy_id} ({strategy_name} - {symbol}):")
            print(f"    初始价值: ¥{initial_value:,.2f}")
            print(f"    最终价值: ¥{value:,.2f}")
            print(f"    收益率: {strategy_return:.2%}")
            print(f"    权重: {strategy_alloc.weight:.1%}")

        # 测试组合再平衡效果
        print("\n组合再平衡分析:")
        portfolio_values = pd.Series(results['strategy_values'])
        weights = np.array([alloc.weight for alloc in all_strategies])

        # 计算当前权重
        current_values = np.array(list(results['strategy_values'].values()))
        current_weights = current_values / current_values.sum()

        print(f"  目标权重: {weights}")
        print(f"  实际权重: {current_weights}")
        print(f"  权重偏差: {np.abs(weights - current_weights).mean():.2%}")

        return True

    except Exception as e:
        print(f"❌ 多策略组合测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_analysis():
    """测试绩效分析功能"""
    print("\n" + "="*70)
    print("测试3：绩效分析功能")
    print("="*70)

    try:
        from backtest.engine.analysis import PerformanceAnalyzer

        # 创建分析器
        analyzer = PerformanceAnalyzer()

        # 生成模拟权益曲线
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        initial_value = 10000000
        returns = np.random.normal(0.001, 0.02, 252)  # 年化收益12%，波动率20%
        equity_curve = pd.Series(initial_value * (1 + returns).cumprod(), index=dates)

        # 进行分析
        metrics = analyzer.analyze(equity_curve)

        print("\n绩效分析结果:")
        print("="*50)
        print(f"总收益率: {metrics['total_return']:.2%}")
        print(f"年化收益率: {metrics['annual_return']:.2%}")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")
        print(f"最终权益: ¥{metrics['final_equity']:,.2f}")

        # 测试基准对比
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)  # 基准收益
        benchmark_equity = initial_value * (1 + benchmark_returns).cumprod()

        benchmark_metrics = analyzer.analyze(equity_curve, benchmark_returns=benchmark_returns)

        print(f"\n基准对比:")
        print(f"基准收益率: {benchmark_metrics['benchmark_return']:.2%}")
        print(f"超额收益: {benchmark_metrics['excess_return']:.2%}")

        # 测试交易分析
        print(f"\n交易分析:")
        # 生成模拟交易记录
        trades = pd.DataFrame({
            'entry_date': pd.date_range(start='2023-01-01', periods=50, freq='7D'),
            'exit_date': pd.date_range(start='2023-01-08', periods=50, freq='7D'),
            'pnl': np.random.normal(5000, 20000, 50),
            'return_rate': np.random.normal(0.02, 0.1, 50)
        })

        win_rate = analyzer.calculate_win_rate(trades)
        print(f"交易次数: {len(trades)}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均每笔收益: ¥{trades['pnl'].mean():,.2f}")

        return True

    except Exception as e:
        print(f"❌ 绩效分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_metrics():
    """测试风险指标计算"""
    print("\n" + "="*70)
    print("测试4：风险指标计算")
    print("="*70)

    try:
        from backtest.engine.analysis import PerformanceAnalyzer

        analyzer = PerformanceAnalyzer()

        # 生成模拟数据
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        returns = np.random.normal(0.001, 0.02, 252)  # 年化收益12%，波动率20%
        equity_curve = pd.Series(10000000 * (1 + returns).cumprod(), index=dates)

        # 计算风险指标
        metrics = analyzer.analyze(equity_curve)

        # 计算附加风险指标
        returns_series = analyzer.calculate_returns(equity_curve)

        # VaR计算
        var_95 = np.percentile(returns_series, 5)
        var_99 = np.percentile(returns_series, 1)

        # CVaR计算
        cvar_95 = returns_series[returns_series <= var_95].mean()

        # 波动率相关指标
        annual_volatility = returns_series.std() * np.sqrt(252)
        downside_volatility = returns_series[returns_series < 0].std() * np.sqrt(252)

        # 索提诺比率
        sortino_ratio = metrics['annual_return'] / downside_volatility if downside_volatility > 0 else 0

        print("\n风险指标分析:")
        print("="*50)
        print(f"年化波动率: {annual_volatility:.2%}")
        print(f"下行波动率: {downside_volatility:.2%}")
        print(f"VaR (95%): {var_95:.2%}")
        print(f"VaR (99%): {var_99:.2%}")
        print(f"CVaR (95%): {cvar_95:.2%}")
        print(f"索提诺比率: {sortino_ratio:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']:.2%}")

        # 风险调整收益分析
        print(f"\n风险调整收益:")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {sortino_ratio:.2f}")
        print(f"Calmar比率: {metrics['annual_return'] / abs(metrics['max_drawdown']):.2f}")

        # 回撤分析
        drawdowns = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        max_drawdown_date = drawdowns.idxmin()
        avg_drawdown = drawdowns.mean()

        print(f"\n回撤分析:")
        print(f"最大回撤日期: {max_drawdown_date.strftime('%Y-%m-%d')}")
        print(f"平均回撤: {avg_drawdown:.2%}")
        print(f"回撤标准差: {drawdowns.std():.2%}")

        return True

    except Exception as e:
        print(f"❌ 风险指标测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("quantA 组合回测和风险管理测试")
    print("="*70)

    try:
        # 运行所有测试
        test_results = []

        test_results.append(("单策略多资产", test_single_strategy_multi_asset()))
        test_results.append(("多策略组合", test_multi_strategy_portfolio()))
        test_results.append(("绩效分析", test_performance_analysis()))
        test_results.append(("风险指标", test_risk_metrics()))

        # 汇总结果
        print("\n" + "="*70)
        print("测试结果汇总")
        print("="*70)

        passed = 0
        for test_name, result in test_results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
            if result:
                passed += 1

        print(f"\n总结: {passed}/{len(test_results)} 测试通过")

        if passed == len(test_results):
            print("\n🎉 所有测试通过！组合回测和风险管理功能正常。")
            return True
        else:
            print(f"\n⚠️  {len(test_results) - passed} 个测试失败，需要进一步检查。")
            return False

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)