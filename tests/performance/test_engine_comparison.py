"""
性能对比测试
对比Python引擎和Rust引擎的性能
"""

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine as PythonBacktestEngine
from backtest.engine.rust_engine import RustBacktestEngine, check_rust_availability
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy


@pytest.mark.performance
@pytest.mark.slow
class TestEngineComparison:
    """Python vs Rust引擎性能对比"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """检查Rust引擎是否可用"""
        self.rust_available = check_rust_availability()

        if not self.rust_available:
            pytest.skip("Rust引擎未安装，跳过对比测试")

    def generate_test_data(
        self, num_stocks: int, start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """生成测试数据"""
        engine = PythonBacktestEngine(initial_cash=1000000)
        return engine.generate_mock_data(
            symbols=[f"{600000+i}.SH" for i in range(num_stocks)],
            start_date=start_date,
            end_date=end_date,
        )

    def test_single_stock_comparison(self):
        """测试单股票回测性能对比"""
        print("\n" + "=" * 70)
        print("单股票回测性能对比")
        print("=" * 70)

        # 生成测试数据
        data = self.generate_test_data(1, "2023-01-01", "2023-12-31")

        # Python引擎
        python_engine = PythonBacktestEngine(initial_cash=1000000)

        start_time = time.perf_counter()
        python_results = python_engine.run(BuyAndHoldStrategy(), data)
        python_time = time.perf_counter() - start_time

        # Rust引擎
        rust_engine = RustBacktestEngine(initial_cash=1000000)
        rust_engine.load_data(data)

        start_time = time.perf_counter()
        rust_results = rust_engine.run()
        rust_time = time.perf_counter() - start_time

        # 性能对比
        speedup = python_time / rust_time if rust_time > 0 else 0

        print(f"\nPython引擎:")
        print(f"  耗时: {python_time:.4f}秒")
        print(f"  总收益率: {python_results['total_return']:.2%}")

        print(f"\nRust引擎:")
        print(f"  耗时: {rust_time:.4f}秒")
        print(f"  总收益率: {rust_results['total_return']:.2%}")

        print(f"\n性能提升:")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  时间节省: {(1 - rust_time/python_time)*100:.1f}%")

        # Rust应该更快（或者至少相近）
        assert rust_time <= python_time * 2, "Rust引擎性能过差"

    def test_multi_stock_comparison(self):
        """测试多股票回测性能对比"""
        print("\n" + "=" * 70)
        print("多股票回测性能对比")
        print("=" * 70)

        num_stocks_list = [1, 5, 10, 20]
        results = []

        for num_stocks in num_stocks_list:
            print(f"\n测试 {num_stocks} 只股票:")

            # 生成测试数据
            data = self.generate_test_data(num_stocks, "2023-01-01", "2023-06-30")

            # Python引擎
            python_engine = PythonBacktestEngine(initial_cash=1000000)

            start_time = time.perf_counter()
            python_engine.run(BuyAndHoldStrategy(), data)
            python_time = time.perf_counter() - start_time

            # Rust引擎
            rust_engine = RustBacktestEngine(initial_cash=1000000)
            rust_engine.load_data(data)

            start_time = time.perf_counter()
            rust_engine.run()
            rust_time = time.perf_counter() - start_time

            speedup = python_time / rust_time if rust_time > 0 else 0

            print(f"  Python: {python_time:.4f}秒")
            print(f"  Rust:   {rust_time:.4f}秒")
            print(f"  加速比: {speedup:.2f}x")

            results.append(
                {
                    "stocks": num_stocks,
                    "python_time": python_time,
                    "rust_time": rust_time,
                    "speedup": speedup,
                }
            )

        # 打印汇总表
        print("\n" + "=" * 70)
        print("性能对比汇总")
        print("=" * 70)
        print(f"{'股票数':<10}{'Python(秒)':<15}{'Rust(秒)':<15}{'加速比':<10}")
        print("-" * 70)
        for r in results:
            print(
                f"{r['stocks']:<10}"
                f"{r['python_time']:<15.4f}"
                f"{r['rust_time']:<15.4f}"
                f"{r['speedup']:<10.2f}"
            )

        # 验证Rust引擎在大规模数据上的优势
        if len(results) >= 3:
            # 最后一个测试应该有明显优势
            assert results[-1]["rust_time"] < results[-1]["python_time"], "Rust引擎应该更快"

    def test_indicator_calculation_comparison(self):
        """测试技术指标计算性能对比"""
        print("\n" + "=" * 70)
        print("技术指标计算性能对比")
        print("=" * 70)

        # 生成测试数据
        size = 10000
        data = pd.Series(np.random.randn(size).cumsum() + 100)

        from backtest.engine.indicators import EMA, RSI, SMA

        # 测试SMA
        start_time = time.perf_counter()
        sma_result = SMA(data, 20)
        python_sma_time = time.perf_counter() - start_time

        print(f"\nSMA计算 ({size} 数据点):")
        print(f"  Python: {python_sma_time*1000:.2f}ms")

        # 如果Rust版本可用，进行对比
        # 这里只是演示，实际需要Rust实现指标计算
        print(f"  Rust:   待实现")

    def test_memory_usage_comparison(self):
        """测试内存使用对比"""
        print("\n" + "=" * 70)
        print("内存使用对比")
        print("=" * 70)

        import os

        import psutil

        process = psutil.Process(os.getpid())

        # 生成测试数据
        data = self.generate_test_data(20, "2020-01-01", "2023-12-31")

        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Python引擎
        python_engine = PythonBacktestEngine(initial_cash=1000000)
        python_engine.run(BuyAndHoldStrategy(), data)

        python_memory = process.memory_info().rss / (1024 * 1024)  # MB
        python_delta = python_memory - initial_memory

        # Rust引擎
        rust_engine = RustBacktestEngine(initial_cash=1000000)
        rust_engine.load_data(data)
        rust_engine.run()

        rust_memory = process.memory_info().rss / (1024 * 1024)  # MB
        rust_delta = rust_memory - python_memory

        print(f"\n初始内存: {initial_memory:.2f}MB")
        print(f"Python引擎内存增长: {python_delta:.2f}MB")
        print(f"Rust引擎内存增长: {rust_delta:.2f}MB")
        print(f"内存节省: {(1 - rust_delta/python_delta)*100:.1f}%")

        # Rust应该使用更少内存
        # 注意：这个测试可能不够精确，因为Python的GC等原因
        assert rust_delta <= python_delta * 1.5, "Rust引擎内存使用过高"

    def test_scalability_comparison(self):
        """测试扩展性对比"""
        print("\n" + "=" * 70)
        print("扩展性对比（不同数据规模）")
        print("=" * 70)

        test_configs = [
            (5, "2023-01-01", "2023-03-31"),  # 小规模
            (10, "2023-01-01", "2023-06-30"),  # 中规模
            (20, "2023-01-01", "2023-12-31"),  # 大规模
        ]

        python_times = []
        rust_times = []

        for num_stocks, start, end in test_configs:
            data = self.generate_test_data(num_stocks, start, end)

            # Python
            python_engine = PythonBacktestEngine(initial_cash=1000000)
            start_time = time.perf_counter()
            python_engine.run(BuyAndHoldStrategy(), data)
            python_time = time.perf_counter() - start_time
            python_times.append(python_time)

            # Rust
            rust_engine = RustBacktestEngine(initial_cash=1000000)
            rust_engine.load_data(data)
            start_time = time.perf_counter()
            rust_engine.run()
            rust_time = time.perf_counter() - start_time
            rust_times.append(rust_time)

        # 计算扩展性
        print("\n扩展性分析:")
        print(f"{'配置':<10}{'Python(秒)':<15}{'Rust(秒)':<15}{'Rust优势':<10}")
        print("-" * 70)

        for i, (num_stocks, _, _) in enumerate(test_configs):
            advantage = (python_times[i] - rust_times[i]) / python_times[i] * 100
            print(
                f"{f'{num_stocks}股票':<10}"
                f"{python_times[i]:<15.4f}"
                f"{rust_times[i]:<15.4f}"
                f"{advantage:<10.1f}%"
            )

    def test_strategy_complexity_comparison(self):
        """测试不同策略复杂度的性能对比"""
        print("\n" + "=" * 70)
        print("策略复杂度性能对比")
        print("=" * 70)

        data = self.generate_test_data(5, "2023-01-01", "2023-06-30")

        strategies = [
            ("BuyAndHold", BuyAndHoldStrategy()),
            ("MA_Cross_5_20", MovingAverageCrossStrategy(5, 20)),
            ("MA_Cross_10_30", MovingAverageCrossStrategy(10, 30)),
        ]

        for name, strategy in strategies:
            print(f"\n策略: {name}")

            # Python
            python_engine = PythonBacktestEngine(initial_cash=1000000)
            start_time = time.perf_counter()
            python_results = python_engine.run(strategy, data)
            python_time = time.perf_counter() - start_time

            print(f"  Python: {python_time:.4f}秒")
            print(f"  收益率: {python_results['total_return']:.2%}")

            # Rust (需要实现策略接口)
            rust_engine = RustBacktestEngine(initial_cash=1000000)
            rust_engine.load_data(data)

            start_time = time.perf_counter()
            rust_results = rust_engine.run(strategy)
            rust_time = time.perf_counter() - start_time

            speedup = python_time / rust_time if rust_time > 0 else 0

            print(f"  Rust:   {rust_time:.4f}秒")
            print(f"  收益率: {rust_results['total_return']:.2%}")
            print(f"  加速比: {speedup:.2f}x")


@pytest.mark.integration
def test_engine_results_consistency():
    """测试两个引擎结果的一致性"""
    if not check_rust_availability():
        pytest.skip("Rust引擎未安装")

    # 生成相同的测试数据
    python_engine = PythonBacktestEngine(initial_cash=1000000)
    data = python_engine.generate_mock_data(
        symbols=["600000.SH"],
        start_date="2023-01-01",
        end_date="2023-03-31",
    )

    # 运行Python引擎
    python_results = python_engine.run(BuyAndHoldStrategy(), data)

    # 运行Rust引擎
    rust_engine = RustBacktestEngine(initial_cash=1000000)
    rust_engine.load_data(data)
    rust_results = rust_engine.run()

    # 比较关键指标（允许1%误差）
    assert (
        abs(python_results["total_return"] - rust_results["total_return"]) < 0.01
    ), "收益率差异过大"

    assert (
        abs(python_results["sharpe_ratio"] - rust_results["sharpe_ratio"]) < 0.05
    ), "夏普比率差异过大"

    print("\n✓ 两个引擎的结果基本一致")


@pytest.fixture(scope="session", autouse=True)
def print_comparison_summary(request):
    """打印性能对比汇总"""
    yield

    if request.config.getoption("verbose") > 0:
        print("\n" + "=" * 70)
        print("性能对比测试完成")
        print("=" * 70)
        if check_rust_availability():
            print("✓ Rust引擎已安装")
        else:
            print("✗ Rust引擎未安装")
            print("  提示: 运行 'pip install maturin && cd rust_engine && maturin develop'")
