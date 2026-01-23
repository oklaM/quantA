"""
性能基准测试
测试系统在各种负载下的性能表现
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import psutil
import pytest

from backtest.engine import BacktestEngine
from backtest.engine.indicators import *
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy
from utils.performance import PerformanceProfiler


@pytest.fixture
def profiler():
    """性能分析器"""
    return PerformanceProfiler()


@pytest.fixture
def get_memory_usage():
    """获取当前内存使用"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB


@pytest.mark.performance
@pytest.mark.slow
class TestBacktestPerformance:
    """回测引擎性能测试"""

    def test_backtest_single_stock_performance(self, profiler):
        """测试单股票回测性能"""
        engine = BacktestEngine(initial_cash=1000000)

        # 生成不同规模的数据
        test_cases = [
            ("小规模", 1, "2023-01-01", "2023-03-31", 500),  # 1股票, 3个月
            ("中等规模", 5, "2023-01-01", "2023-06-30", 2000),  # 5股票, 6个月
            ("大规模", 10, "2023-01-01", "2023-12-31", 10000),  # 10股票, 1年
        ]

        results = {}

        for name, num_stocks, start, end, expected_bars in test_cases:
            with profiler.profile(f"回测-{name}"):
                data = engine.generate_mock_data(
                    symbols=[f"{600000+i}.SH" for i in range(num_stocks)],
                    start_date=start,
                    end_date=end,
                )

                strategy = BuyAndHoldStrategy()
                result = engine.run(strategy, data)

            execution_time = profiler.get_last_duration()

            # 计算性能指标
            total_bars = sum(len(df) for df in data.values())
            bars_per_second = total_bars / execution_time if execution_time > 0 else 0

            results[name] = {
                "stocks": num_stocks,
                "bars": total_bars,
                "time": execution_time,
                "bars_per_second": bars_per_second,
            }

            # 性能断言
            assert execution_time < 60, f"{name}回测时间过长: {execution_time:.2f}秒"

        # 打印性能报告
        print("\n回测性能测试结果:")
        print("=" * 70)
        for name, metrics in results.items():
            print(f"{name}:")
            print(f"  股票数: {metrics['stocks']}")
            print(f"  K线数: {metrics['bars']}")
            print(f"  耗时: {metrics['time']:.2f}秒")
            print(f"  速度: {metrics['bars_per_second']:.0f} bars/秒")
            print()

    def test_backtest_multi_strategy_performance(self, profiler):
        """测试多策略并行回测性能"""
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH", "000001.SZ", "600036.SH"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        strategies = [
            BuyAndHoldStrategy(),
            MovingAverageCrossStrategy(5, 20),
            MovingAverageCrossStrategy(10, 30),
            MovingAverageCrossStrategy(15, 40),
        ]

        # 串行执行
        with profiler.profile("串行回测"):
            for strategy in strategies:
                engine.run(strategy, data)

        serial_time = profiler.get_last_duration()

        # 性能基准
        assert serial_time < 30, f"串行回测时间过长: {serial_time:.2f}秒"

        print(f"\n多策略回测性能:")
        print(f"  策略数量: {len(strategies)}")
        print(f"  总耗时: {serial_time:.2f}秒")
        print(f"  平均耗时: {serial_time/len(strategies):.2f}秒/策略")


@pytest.mark.performance
class TestIndicatorPerformance:
    """技术指标计算性能测试"""

    def test_indicator_calculation_performance(self, profiler):
        """测试技术指标计算性能"""
        # 生成测试数据
        sizes = [1000, 5000, 10000, 50000]

        results = {}

        for size in sizes:
            data = pd.Series(np.random.randn(size).cumsum() + 100)

            # 测试各种指标
            indicators = {
                "SMA(20)": lambda: SMA(data, 20),
                "EMA(20)": lambda: EMA(data, 20),
                "RSI(14)": lambda: RSI(data, 14),
                "MACD": lambda: MACD(data),
                "BOLLINGER": lambda: BOLLINGER_BANDS(data),
                "ATR(14)": lambda: ATR(
                    pd.DataFrame(
                        {
                            "high": data + 1,
                            "low": data - 1,
                            "close": data,
                        }
                    ),
                    14,
                ),
            }

            size_results = {}

            for name, calc_func in indicators.items():
                with profiler.profile(f"{name}-{size}"):
                    result = calc_func()

                duration = profiler.get_last_duration()
                size_results[name] = {
                    "time": duration,
                    "points_per_second": size / duration if duration > 0 else 0,
                }

                # 性能断言
                assert duration < 5, f"{name}计算时间过长 ({size}点): {duration:.2f}秒"

            results[size] = size_results

        # 打印性能报告
        print("\n技术指标计算性能:")
        print("=" * 70)
        for size, size_results in results.items():
            print(f"\n数据规模: {size} 点")
            for name, metrics in size_results.items():
                print(
                    f"  {name:20s}: {metrics['time']*1000:6.2f}ms ({metrics['points_per_second']:8.0f} points/s)"
                )

    def test_batch_indicator_calculation(self, profiler):
        """测试批量指标计算性能"""
        size = 10000
        data = pd.DataFrame(
            {
                "open": np.random.randn(size).cumsum() + 100,
                "high": np.random.randn(size).cumsum() + 102,
                "low": np.random.randn(size).cumsum() + 98,
                "close": np.random.randn(size).cumsum() + 100,
                "volume": np.random.randint(1000000, 10000000, size),
            }
        )

        # 批量计算
        with profiler.profile("批量指标计算"):
            data["sma_20"] = SMA(data["close"], 20)
            data["ema_20"] = EMA(data["close"], 20)
            data["rsi"] = RSI(data["close"], 14)
            macd_line, signal_line, histogram = MACD(data["close"])
            data["macd"] = macd_line
            upper, middle, lower = BOLLINGER_BANDS(data["close"])
            data["bb_upper"] = upper
            data["bb_middle"] = middle
            data["bb_lower"] = lower

        duration = profiler.get_last_duration()

        # 性能基准
        assert duration < 10, f"批量指标计算时间过长: {duration:.2f}秒"

        print(f"\n批量指标计算性能:")
        print(f"  数据点数: {size}")
        print(f"  指标数量: 7")
        print(f"  总耗时: {duration:.2f}秒")
        print(f"  平均耗时: {duration/7*1000:.2f}ms/指标")


@pytest.mark.performance
class TestMemoryPerformance:
    """内存使用性能测试"""

    def test_memory_usage_during_backtest(self, get_memory_usage):
        """测试回测过程中的内存使用"""
        initial_memory = get_memory_usage()

        engine = BacktestEngine(initial_cash=1000000)

        # 生成大量数据
        data = engine.generate_mock_data(
            symbols=[f"{600000+i}.SH" for i in range(50)],  # 50只股票
            start_date="2020-01-01",
            end_date="2023-12-31",  # 4年数据
        )

        memory_after_data = get_memory_usage()

        # 运行回测
        strategy = BuyAndHoldStrategy()
        results = engine.run(strategy, data)

        memory_after_backtest = get_memory_usage()

        # 计算内存增长
        data_memory = memory_after_data - initial_memory
        backtest_memory = memory_after_backtest - memory_after_data

        print(f"\n内存使用分析:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  数据加载后: {memory_after_data:.2f}MB (+{data_memory:.2f}MB)")
        print(f"  回测完成后: {memory_after_backtest:.2f}MB (+{backtest_memory:.2f}MB)")

        # 内存使用应该在合理范围内
        assert memory_after_backtest < 2000, "内存使用超过2GB"

    def test_memory_leak_check(self, get_memory_usage):
        """检查内存泄漏"""
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH", "000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        initial_memory = get_memory_usage()
        memory_snapshots = []

        # 运行多次回测
        for i in range(10):
            strategy = BuyAndHoldStrategy()
            engine.run(strategy, data)
            memory_snapshots.append(get_memory_usage())

        final_memory = memory_snapshots[-1]

        # 检查内存是否持续增长
        memory_growth = final_memory - initial_memory

        print(f"\n内存泄漏检查:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  最终内存: {final_memory:.2f}MB")
        print(f"  内存增长: {memory_growth:.2f}MB")

        # 内存增长不应超过100MB
        assert memory_growth < 100, f"可能存在内存泄漏，增长{memory_growth:.2f}MB"


@pytest.mark.performance
class TestIOPerformance:
    """I/O性能测试"""

    def test_data_loading_performance(self, profiler, tmp_path):
        """测试数据加载性能"""
        engine = BacktestEngine(initial_cash=1000000)

        # 生成测试数据
        data = engine.generate_mock_data(
            symbols=[f"{600000+i}.SH" for i in range(20)],
            start_date="2020-01-01",
            end_date="2023-12-31",
        )

        # 测试CSV保存/加载
        csv_file = tmp_path / "test_data.csv"

        with profiler.profile("CSV保存"):
            # 保存第一只股票的数据
            symbol = list(data.keys())[0]
            data[symbol].to_csv(csv_file, index=False)

        save_time = profiler.get_last_duration()

        with profiler.profile("CSV加载"):
            loaded_data = pd.read_csv(csv_file)

        load_time = profiler.get_last_duration()

        print(f"\nCSV I/O性能:")
        print(f"  保存时间: {save_time*1000:.2f}ms")
        print(f"  加载时间: {load_time*1000:.2f}ms")

        assert save_time < 1, "CSV保存时间过长"
        assert load_time < 1, "CSV加载时间过长"

    def test_pickle_performance(self, profiler, tmp_path):
        """测试pickle序列化性能"""
        engine = BacktestEngine(initial_cash=1000000)

        data = engine.generate_mock_data(
            symbols=[f"{600000+i}.SH" for i in range(10)],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        pickle_file = tmp_path / "test_data.pkl"

        with profiler.profile("Pickle保存"):
            import pickle

            with open(pickle_file, "wb") as f:
                pickle.dump(data, f)

        save_time = profiler.get_last_duration()

        with profiler.profile("Pickle加载"):
            with open(pickle_file, "rb") as f:
                loaded_data = pickle.load(f)

        load_time = profiler.get_last_duration()

        print(f"\nPickle I/O性能:")
        print(f"  保存时间: {save_time*1000:.2f}ms")
        print(f"  加载时间: {load_time*1000:.2f}ms")


@pytest.mark.performance
class TestScalabilityPerformance:
    """扩展性性能测试"""

    def test_linear_scalability(self, profiler):
        """测试线性扩展性"""
        engine = BacktestEngine(initial_cash=1000000)

        # 测试不同规模的扩展性
        scales = [1, 5, 10, 20]
        times = []

        for scale in scales:
            data = engine.generate_mock_data(
                symbols=[f"{600000+i}.SH" for i in range(scale)],
                start_date="2023-01-01",
                end_date="2023-06-30",
            )

            with profiler.profile(f"规模-{scale}"):
                strategy = BuyAndHoldStrategy()
                engine.run(strategy, data)

            times.append(profiler.get_last_duration())

        # 计算扩展性
        base_time = times[0]
        speedup = [base_time / t for t in times]

        print(f"\n扩展性测试结果:")
        print(f"{'规模':<10}{'时间(秒)':<15}{'加速比':<10}")
        print("-" * 35)
        for s, t, sp in zip(scales, times, speedup):
            print(f"{s:<10}{t:<15.2f}{sp:<10.2f}")

        # 验证扩展性
        # 规模增加10倍，时间应增加约10倍（允许一定误差）
        expected_ratio = scales[-1] / scales[0]
        actual_ratio = times[-1] / times[0]

        efficiency = (expected_ratio / actual_ratio) * 100 if actual_ratio > 0 else 0

        print(f"\n扩展效率: {efficiency:.1f}%")
        assert efficiency > 50, f"扩展效率过低: {efficiency:.1f}%"


@pytest.mark.performance
class TestConcurrencyPerformance:
    """并发性能测试"""

    def test_multiprocessing_performance(self, profiler):
        """测试多进程性能"""
        import multiprocessing as mp

        def run_backtest(symbol):
            engine = BacktestEngine(initial_cash=1000000)
            data = engine.generate_mock_data(
                symbols=[symbol],
                start_date="2023-01-01",
                end_date="2023-12-31",
            )
            strategy = BuyAndHoldStrategy()
            return engine.run(strategy, data)

        symbols = [f"{600000+i}.SH" for i in range(8)]

        # 串行执行
        with profiler.profile("串行执行"):
            serial_results = [run_backtest(s) for s in symbols]

        serial_time = profiler.get_last_duration()

        # 并行执行
        with profiler.profile(f"并行执行-{mp.cpu_count()}核"):
            with mp.Pool(processes=mp.cpu_count()) as pool:
                parallel_results = pool.map(run_backtest, symbols)

        parallel_time = profiler.get_last_duration()

        # 计算加速比
        speedup = serial_time / parallel_time if parallel_time > 0 else 0

        print(f"\n并发性能测试:")
        print(f"  串行时间: {serial_time:.2f}秒")
        print(f"  并行时间: {parallel_time:.2f}秒")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  并发效率: {speedup/mp.cpu_count()*100:.1f}%")

        # 并行应该更快
        assert parallel_time < serial_time, "并行执行没有带来性能提升"


@pytest.mark.performance
class TestPerformanceProfiling:
    """性能剖析测试"""

    def test_hotspot_analysis(self, profiler):
        """热点分析"""
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH", "000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # 运行并收集性能数据
        with profiler.profile("完整回测"):
            strategy = MovingAverageCrossStrategy(5, 20)
            results = engine.run(strategy, data)

        # 生成性能报告
        report = profiler.generate_report()

        print("\n" + "=" * 70)
        print("性能剖析报告")
        print("=" * 70)
        print(report)

        # 验证报告包含必要信息
        assert "完整回测" in report
        assert "耗时" in report or "time" in report.lower()


@pytest.fixture(scope="session", autouse=True)
def print_performance_summary(request):
    """打印性能测试汇总"""
    yield

    if request.config.getoption("verbose") > 0:
        print("\n" + "=" * 70)
        print("性能测试完成")
        print("=" * 70)
        print("提示: 使用 pytest -m performance --verbose 运行性能测试")
        print("      使用 pytest -m performance -v 查看详细输出")
