"""
参数优化示例
展示如何使用优化框架进行策略参数和超参数调优
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: optuna未安装，部分功能不可用。安装命令: pip install optuna")

from backtest.optimization import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    MultiObjectiveOptimizer,
    create_optimizer,
)
from backtest.engine.strategies import BollingerBandsStrategy
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_sample_data(days: int = 500, start_price: float = 100.0):
    """
    生成示例数据

    Args:
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


def example_grid_search():
    """示例1：网格搜索优化"""
    print("\n" + "="*70)
    print("示例1：网格搜索优化")
    print("="*70)

    # 准备数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 定义参数空间
    param_space = {
        'period': [10, 20, 30],
        'std_dev': [1.5, 2.0, 2.5],
    }

    print(f"参数空间: {param_space}")

    # 创建优化器
    optimizer = create_optimizer(
        optimizer_type='grid',
        data=data,
        strategy_class=BollingerBandsStrategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 执行优化
    print("\n开始网格搜索...")
    best_result = optimizer.optimize(
        param_space=param_space,
        optimization_target='sharpe_ratio',
    )

    print("\n" + "="*70)
    print("优化结果")
    print("="*70)
    print(f"最佳参数: {best_result.params}")
    print(f"夏普比率: {best_result.metrics['sharpe_ratio']:.4f}")
    print(f"年化收益: {best_result.metrics['annual_return']:.2%}")
    print(f"最大回撤: {best_result.metrics['max_drawdown']:.2%}")
    print(f"胜率: {best_result.metrics['win_rate']:.2%}")

    return best_result


def example_random_search():
    """示例2：随机搜索优化"""
    print("\n" + "="*70)
    print("示例2：随机搜索优化")
    print("="*70)

    # 准备数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 定义参数空间（支持范围）
    param_space = {
        'period': ['int', 5, 50],  # 整数范围 [5, 50]
        'std_dev': ['uniform', 1.0, 3.0],  # 均匀分布 [1.0, 3.0]
    }

    print(f"参数空间: {param_space}")

    # 创建优化器
    optimizer = create_optimizer(
        optimizer_type='random',
        data=data,
        strategy_class=BollingerBandsStrategy,
    )

    # 执行优化
    print("\n开始随机搜索 (20次试验)...")
    best_result = optimizer.optimize(
        param_space=param_space,
        optimization_target='sharpe_ratio',
        n_trials=20,
    )

    print("\n" + "="*70)
    print("优化结果")
    print("="*70)
    print(f"最佳参数: {best_result.params}")
    print(f"夏普比率: {best_result.metrics['sharpe_ratio']:.4f}")

    return best_result


def example_bayesian_optimization():
    """示例3：贝叶斯优化"""
    print("\n" + "="*70)
    print("示例3：贝叶斯优化")
    print("="*70)

    if not OPTUNA_AVAILABLE:
        print("\n无法运行：需要安装 optuna")
        print("安装命令: pip install optuna")
        return None

    # 准备数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 定义参数空间
    param_space = {
        'period': ('int', 5, 50),
        'std_dev': ('float', 1.0, 3.0),
    }

    print(f"参数空间: {param_space}")

    # 创建优化器
    optimizer = create_optimizer(
        optimizer_type='bayesian',
        data=data,
        strategy_class=BollingerBandsStrategy,
    )

    # 执行优化
    print("\n开始贝叶斯优化 (30次试验)...")
    best_result = optimizer.optimize(
        param_space=param_space,
        optimization_target='sharpe_ratio',
        n_trials=30,
    )

    print("\n" + "="*70)
    print("优化结果")
    print("="*70)
    print(f"最佳参数: {best_result.params}")
    print(f"夏普比率: {best_result.metrics['sharpe_ratio']:.4f}")
    print(f"年化收益: {best_result.metrics['annual_return']:.2%}")

    return best_result


def example_multi_objective_optimization():
    """示例4：多目标优化"""
    print("\n" + "="*70)
    print("示例4：多目标优化")
    print("="*70)

    if not OPTUNA_AVAILABLE:
        print("\n无法运行：需要安装 optuna")
        print("安装命令: pip install optuna")
        return None

    # 准备数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 定义参数空间
    param_space = {
        'period': ('int', 10, 40),
        'std_dev': ('float', 1.5, 2.5),
    }

    # 定义多个优化目标
    objectives = {
        'sharpe_ratio': 'maximize',  # 最大化夏普比率
        'max_drawdown': 'minimize',  # 最小化最大回撤
    }

    print(f"优化目标: {objectives}")

    # 创建优化器
    optimizer = create_optimizer(
        optimizer_type='multi_objective',
        data=data,
        strategy_class=BollingerBandsStrategy,
    )

    # 执行优化
    print("\n开始多目标优化 (30次试验)...")
    pareto_results = optimizer.optimize(
        param_space=param_space,
        objectives=objectives,
        n_trials=30,
    )

    print(f"\n找到 {len(pareto_results)} 个Pareto最优解：")

    for i, result in enumerate(pareto_results[:5]):  # 显示前5个
        print(f"\n解 {i+1}:")
        print(f"  参数: {result.params}")
        print(f"  夏普比率: {result.metrics['sharpe_ratio']:.4f}")
        print(f"  最大回撤: {result.metrics['max_drawdown']:.2%}")

    return pareto_results


def example_optimization_comparison():
    """示例5：对比不同优化方法"""
    print("\n" + "="*70)
    print("示例5：对比不同优化方法")
    print("="*70)

    # 准备数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 定义相同的参数空间
    param_spaces = {
        'grid': {
            'period': [10, 20, 30],
            'std_dev': [1.5, 2.0, 2.5],
        },
        'random': {
            'period': ['int', 5, 50],
            'std_dev': ['uniform', 1.0, 3.0],
        },
    }

    results_comparison = {}

    # 网格搜索
    print("\n1. 网格搜索")
    grid_optimizer = create_optimizer('grid', data, BollingerBandsStrategy)
    grid_result = grid_optimizer.optimize(
        param_space=param_spaces['grid'],
        optimization_target='sharpe_ratio',
    )
    results_comparison['grid'] = {
        'sharpe_ratio': grid_result.metrics['sharpe_ratio'],
        'params': grid_result.params,
    }

    # 随机搜索
    print("\n2. 随机搜索")
    random_optimizer = create_optimizer('random', data, BollingerBandsStrategy)
    random_result = random_optimizer.optimize(
        param_space=param_spaces['random'],
        optimization_target='sharpe_ratio',
        n_trials=20,
    )
    results_comparison['random'] = {
        'sharpe_ratio': random_result.metrics['sharpe_ratio'],
        'params': random_result.params,
    }

    # 对比结果
    print("\n" + "="*70)
    print("优化方法对比")
    print("="*70)

    comparison_df = pd.DataFrame(results_comparison).T
    print(comparison_df.to_string())

    return comparison_df


def example_optimization_with_different_metrics():
    """示例6：使用不同的优化指标"""
    print("\n" + "="*70)
    print("示例6：使用不同的优化指标")
    print("="*70)

    # 准备数据
    print("\n生成回测数据...")
    data = generate_sample_data(days=500)

    # 定义参数空间
    param_space = {
        'period': [10, 20, 30],
        'std_dev': [1.5, 2.0, 2.5],
    }

    # 测试不同的优化目标
    optimization_targets = [
        'sharpe_ratio',
        'annual_return',
        'max_drawdown',
        'win_rate',
    ]

    results_by_metric = {}

    for target in optimization_targets:
        print(f"\n优化目标: {target}")

        optimizer = create_optimizer('grid', data, BollingerBandsStrategy)

        # 对于最大回撤，我们希望最小化而非最大化
        if target == 'max_drawdown':
            # 需要特殊处理，这里暂时跳过
            print("  (跳过：需要最小化而非最大化)")
            continue

        result = optimizer.optimize(
            param_space=param_space,
            optimization_target=target,
        )

        results_by_metric[target] = {
            'params': result.params,
            'value': result.metrics[target],
        }

        print(f"  最佳参数: {result.params}")
        print(f"  {target}: {result.metrics[target]:.4f}")

    return results_by_metric


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 参数优化示例")
    print("="*70)

    try:
        # 示例1：网格搜索
        # example_grid_search()

        # 示例2：随机搜索
        # example_random_search()

        # 示例3：贝叶斯优化
        # example_bayesian_optimization()

        # 示例4：多目标优化
        # example_multi_objective_optimization()

        # 示例5：优化方法对比
        # example_optimization_comparison()

        # 示例6：不同优化指标
        example_optimization_with_different_metrics()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
