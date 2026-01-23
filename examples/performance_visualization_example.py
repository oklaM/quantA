"""
绩效可视化示例
展示如何使用可视化模块创建各种图表
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装")

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: plotly未安装")

from backtest.visualization import PerformanceVisualizer, InteractiveVisualizer
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.engine import BacktestEngine
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_sample_data(days: int = 500, start_price: float = 100.0):
    """生成示例数据"""
    np.random.seed(42)

    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * (1 + returns).cumprod()

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days,
        freq='D'
    )

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.015)))
        low = close * (1 - abs(np.random.normal(0, 0.015)))
        open_price = close * (1 + np.random.normal(0, 0.008))

        data.append({
            'datetime': date,
            'symbol': '000001.SZ',
            'open': open_price,
            'high': max(high, open_price, close),
            'low': min(low, open_price, close),
            'close': close,
            'volume': np.random.randint(1000000, 10000000)
        })

    df = pd.DataFrame(data)
    return df


def example_backtest_and_visualize():
    """示例1：回测并可视化结果"""
    print("\n" + "="*70)
    print("示例1：回测并可视化结果")
    print("="*70)

    if not MATPLOTLIB_AVAILABLE:
        print("\n无法运行：需要安装 matplotlib")
        print("安装命令: pip install matplotlib")
        return None

    # 生成数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 运行回测
    print("运行回测...")
    strategy = BuyAndHoldStrategy(symbol='000001.SZ')
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    results = engine.run()

    # 提取权益曲线
    equity_curve = pd.Series(results['equity_curve'])

    # 创建可视化器
    print("创建可视化...")
    visualizer = PerformanceVisualizer(
        results=results,
        equity_curve=equity_curve,
    )

    # 绘制各种图表
    print("\n绘制权益曲线...")
    visualizer.plot_equity_curve(save_path='output/equity_curve.png')

    print("绘制回撤图...")
    visualizer.plot_drawdown(save_path='output/drawdown.png')

    print("绘制收益分布...")
    visualizer.plot_returns_distribution(save_path='output/returns_distribution.png')

    print("绘制绩效仪表板...")
    visualizer.plot_performance_dashboard(save_path='output/performance_dashboard.png')

    print("\n创建摘要表格...")
    summary_table = visualizer.create_summary_table(save_path='output/performance_summary.csv')
    print("\n绩效摘要:")
    print(summary_table.to_string(index=False))

    return visualizer


def example_with_benchmark():
    """示例2：使用基准对比"""
    print("\n" + "="*70)
    print("示例2：使用基准对比")
    print("="*70)

    if not MATPLOTLIB_AVAILABLE:
        print("\n无法运行：需要安装 matplotlib")
        return None

    # 生成数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 运行策略回测
    print("运行策略回测...")
    strategy = BuyAndHoldStrategy(symbol='000001.SZ')
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )
    results = engine.run()
    equity_curve = pd.Series(results['equity_curve'])

    # 创建基准（买入持有基准，无手续费）
    print("创建基准...")
    initial_price = data['close'].iloc[0]
    benchmark_values = []
    for price in data['close']:
        benchmark_values.append(1000000.0 * (price / initial_price))
    benchmark_curve = pd.Series(benchmark_values, index=data.index)

    # 创建可视化器（包含基准）
    visualizer = PerformanceVisualizer(
        results=results,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
    )

    # 绘制带基准的图表
    print("绘制权益曲线（含基准）...")
    visualizer.plot_equity_curve(
        save_path='output/equity_curve_with_benchmark.png',
        show_benchmark=True,
    )

    print("绘制滚动指标...")
    visualizer.plot_rolling_metrics(
        window=60,
        save_path='output/rolling_metrics.png',
    )

    print("绘制绩效仪表板（含基准）...")
    visualizer.plot_performance_dashboard(
        save_path='output/performance_dashboard_with_benchmark.png',
    )

    return visualizer


def example_monthly_heatmap():
    """示例3：月度收益热力图"""
    print("\n" + "="*70)
    print("示例3：月度收益热力图")
    print("="*70)

    if not MATPLOTLIB_AVAILABLE:
        print("\n无法运行：需要安装 matplotlib")
        return None

    # 生成数据（需要足够长的时间跨度）
    print("\n生成回测数据（2年）...")
    data = generate_sample_data(days=750)

    # 运行回测
    print("运行回测...")
    strategy = BuyAndHoldStrategy(symbol='000001.SZ')
    engine = BacktestEngine(data=data, strategy=strategy, initial_cash=1000000.0)
    results = engine.run()

    equity_curve = pd.Series(results['equity_curve'])

    # 创建可视化器
    visualizer = PerformanceVisualizer(
        results=results,
        equity_curve=equity_curve,
    )

    # 绘制月度收益热力图
    print("绘制月度收益热力图...")
    visualizer.plot_monthly_returns(save_path='output/monthly_returns_heatmap.png')

    return visualizer


def example_interactive_visualization():
    """示例4：交互式可视化"""
    print("\n" + "="*70)
    print("示例4：交互式可视化")
    print("="*70)

    if not PLOTLY_AVAILABLE:
        print("\n无法运行：需要安装 plotly")
        print("安装命令: pip install plotly")
        return None

    # 生成数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 运行回测
    print("运行回测...")
    strategy = BuyAndHoldStrategy(symbol='000001.SZ')
    engine = BacktestEngine(data=data, strategy=strategy, initial_cash=1000000.0)
    results = engine.run()

    equity_curve = pd.Series(results['equity_curve'])

    # 创建交互式可视化器
    print("创建交互式可视化...")
    visualizer = InteractiveVisualizer(
        results=results,
        equity_curve=equity_curve,
    )

    # 创建交互式图表
    print("创建交互式权益曲线...")
    fig1 = visualizer.plot_interactive_equity_curve(
        save_path='output/interactive_equity_curve.html'
    )

    print("创建交互式仪表板...")
    fig2 = visualizer.plot_interactive_dashboard(
        save_path='output/interactive_dashboard.html'
    )

    print("\n交互式图表已保存为HTML文件，可在浏览器中打开查看")

    return visualizer


def example_custom_visualization():
    """示例5：自定义可视化"""
    print("\n" + "="*70)
    print("示例5：自定义可视化")
    print("="*70)

    if not MATPLOTLIB_AVAILABLE:
        print("\n无法运行：需要安装 matplotlib")
        return

    print("""
自定义可视化的方法：

1. 直接使用matplotlib创建自定义图表

    import matplotlib.pyplot as plt

    # 创建自定义图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 子图1：自定义指标
    axes[0, 0].plot(custom_metric_index, custom_metric_values)
    axes[0, 0].set_title('Custom Metric')

    # 子图2：交易信号
    axes[0, 1].scatter(trade_dates, trade_prices, c=trade_returns, cmap='RdYlGn')
    axes[0, 1].set_title('Trade Signals')

    # 子图3：持仓变化
    axes[1, 0].plot(dates, position_sizes)
    axes[1, 0].set_title('Position Sizes')

    # 子图4：资金使用率
    axes[1, 1].plot(dates, capital_usage)
    axes[1, 1].set_title('Capital Usage')

    plt.tight_layout()
    plt.savefig('custom_visualization.png', dpi=300)

2. 扩展PerformanceVisualizer类

    class CustomVisualizer(PerformanceVisualizer):
        def plot_custom_metric(self, metric_data, save_path=None):
            fig, ax = plt.subplots()
            ax.plot(metric_data.index, metric_data.values)
            ax.set_title('Custom Metric')

            if save_path:
                plt.savefig(save_path)

            return fig

3. 使用多个可视化器对比

    # 创建多个策略的可视化
    visualizers = []

    for strategy_name, strategy in strategies.items():
        results = backtest(strategy)
        viz = PerformanceVisualizer(results, results['equity_curve'])
        visualizers.append((strategy_name, viz))

    # 在同一张图上对比
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, viz in visualizers:
        ax.plot(viz.equity_curve.index, viz.equity_curve.values, label=name)

    ax.legend()
    ax.set_title('Strategy Comparison')
    plt.savefig('strategy_comparison.png')
""")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 绩效可视化示例")
    print("="*70)

    try:
        # 确保输出目录存在
        from pathlib import Path
        Path('output').mkdir(exist_ok=True)

        # 示例1：基本可视化
        # example_backtest_and_visualize()

        # 示例2：使用基准
        # example_with_benchmark()

        # 示例3：月度热力图
        # example_monthly_heatmap()

        # 示例4：交互式可视化
        # example_interactive_visualization()

        # 示例5：自定义可视化
        example_custom_visualization()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("图表已保存到 output/ 目录")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
