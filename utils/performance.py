"""
性能分析工具
提供性能测量、剖析和优化建议
"""

import cProfile
import functools
import io
import pstats
import sys
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, float] = {}
        self.current_profile: Optional[str] = None

    @contextmanager
    def profile(self, name: str, enable_profiling: bool = False):
        """性能分析上下文管理器

        Args:
            name: 分析块名称
            enable_profiling: 是否启用详细剖析

        Example:
            >>> profiler = PerformanceProfiler()
            >>> with profiler.profile("data_loading"):
            ...     load_data()
        """
        self.current_profile = name

        # 记录开始时间
        start_time = time.perf_counter()

        # 可选：启用cProfile
        if enable_profiling:
            pr = cProfile.Profile()
            pr.enable()

        try:
            yield

        finally:
            # 记录结束时间
            end_time = time.perf_counter()
            duration = end_time - start_time

            # 保存时间
            if name in self.timings:
                if isinstance(self.timings[name], list):
                    self.timings[name].append(duration)
                else:
                    self.timings[name] = [self.timings[name], duration]
            else:
                self.timings[name] = duration

            # 如果启用剖析
            if enable_profiling:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
                ps.print_stats(20)  # 打印前20个
                self.timings[f"{name}_profile"] = s.getvalue()

            self.current_profile = None

    def get_last_duration(self, name: Optional[str] = None) -> float:
        """获取最后一次测量的持续时间

        Args:
            name: 分析块名称，如果为None则返回最后一次

        Returns:
            持续时间（秒）
        """
        if name:
            value = self.timings.get(name)
        else:
            # �回最后一次记录
            name = list(self.timings.keys())[-1]
            value = self.timings[name]

        # 如果是列表，返回最后一次
        if isinstance(value, list):
            return value[-1]
        return value

    def get_average_duration(self, name: str) -> Optional[float]:
        """获取平均持续时间"""
        value = self.timings.get(name)
        if value is None:
            return None

        if isinstance(value, list):
            return sum(value) / len(value)
        return value

    def get_total_duration(self, name: str) -> Optional[float]:
        """获取总持续时间"""
        value = self.timings.get(name)
        if value is None:
            return None

        if isinstance(value, list):
            return sum(value)
        return value

    def get_call_count(self, name: str) -> int:
        """获取调用次数"""
        value = self.timings.get(name)
        if value is None:
            return 0

        if isinstance(value, list):
            return len(value)
        return 1

    def reset(self):
        """重置所有统计数据"""
        self.timings.clear()
        self.memory_snapshots.clear()

    def generate_report(self, sort_by: str = "time") -> str:
        """生成性能报告

        Args:
            sort_by: 排序方式 ('time', 'name', 'calls')

        Returns:
            格式化的报告字符串
        """
        if not self.timings:
            return "没有性能数据"

        lines = []
        lines.append("=" * 70)
        lines.append("性能分析报告")
        lines.append("=" * 70)
        lines.append("")

        # 收集数据
        data = []
        for name, value in self.timings.items():
            if name.endswith("_profile"):
                continue

            if isinstance(value, list):
                total = sum(value)
                count = len(value)
                avg = total / count
            else:
                total = value
                count = 1
                avg = value

            data.append(
                {
                    "name": name,
                    "total": total,
                    "avg": avg,
                    "count": count,
                }
            )

        # 排序
        if sort_by == "time":
            data.sort(key=lambda x: x["total"], reverse=True)
        elif sort_by == "name":
            data.sort(key=lambda x: x["name"])
        elif sort_by == "calls":
            data.sort(key=lambda x: x["count"], reverse=True)

        # 打印报告
        lines.append(f"{'名称':<30} {'总耗时(秒)':<12} {'平均耗时(秒)':<12} {'调用次数':<10}")
        lines.append("-" * 70)

        total_time = 0
        total_calls = 0

        for item in data:
            lines.append(
                f"{item['name']:<30} "
                f"{item['total']:<12.4f} "
                f"{item['avg']:<12.4f} "
                f"{item['count']:<10}"
            )
            total_time += item["total"]
            total_calls += item["count"]

        lines.append("-" * 70)
        lines.append(f"{'总计':<30} {total_time:<12.4f} {'':<12} {total_calls:<10}")
        lines.append("")

        return "\n".join(lines)

    def print_report(self, sort_by: str = "time"):
        """打印性能报告到标准输出"""
        print(self.generate_report(sort_by))

    def get_slowest_operations(self, n: int = 5) -> List[tuple]:
        """获取最慢的操作

        Args:
            n: 返回前N个最慢的操作

        Returns:
            [(name, duration), ...] 列表
        """
        data = []
        for name, value in self.timings.items():
            if name.endswith("_profile"):
                continue

            if isinstance(value, list):
                duration = sum(value)
            else:
                duration = value

            data.append((name, duration))

        data.sort(key=lambda x: x[1], reverse=True)
        return data[:n]

    def identify_bottlenecks(self, threshold: float = 1.0) -> List[tuple]:
        """识别性能瓶颈

        Args:
            threshold: 时间阈值（秒），超过此值的操作被视为瓶颈

        Returns:
            [(name, duration), ...] 列表
        """
        bottlenecks = []
        for name, value in self.timings.items():
            if name.endswith("_profile"):
                continue

            if isinstance(value, list):
                duration = sum(value)
            else:
                duration = value

            if duration > threshold:
                bottlenecks.append((name, duration))

        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks


def benchmark(func: Optional[Callable] = None, *, name: Optional[str] = None, repeats: int = 1):
    """性能测试装饰器

    Args:
        func: 被装饰的函数
        name: 测试名称（如果为None，使用函数名）
        repeats: 重复次数

    Example:
        @benchmark(repeats=10)
        def my_function():
            # code
            pass
    """

    def decorator(f):
        benchmark_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            timings = []

            for _ in range(repeats):
                start = time.perf_counter()
                result = f(*args, **kwargs)
                end = time.perf_counter()
                timings.append(end - start)

            # 统计
            total = sum(timings)
            avg = total / len(timings)
            min_time = min(timings)
            max_time = max(timings)

            print(f"\n{benchmark_name} 性能测试结果:")
            print(f"  重复次数: {repeats}")
            print(f"  总耗时: {total:.4f}秒")
            print(f"  平均耗时: {avg:.4f}秒")
            print(f"  最小耗时: {min_time:.4f}秒")
            print(f"  最大耗时: {max_time:.4f}秒")

            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def measure_time(func: Callable) -> Callable:
    """测量函数执行时间的装饰器（简单版本）

    Example:
        @measure_time
        def my_function():
            time.sleep(1)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()

        print(f"{func.__name__} 耗时: {end - start:.4f}秒")
        return result

    return wrapper


def profile_detailed(func: Callable) -> Callable:
    """详细的性能剖析装饰器

    使用cProfile进行详细的性能剖析

    Example:
        @profile_detailed
        def my_function():
            # complex code
            pass
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()

        result = func(*args, **kwargs)

        pr.disable()

        # 打印结果
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(30)  # 打印前30个
        print(f"\n{func.__name__} 详细剖析:")
        print(s.getvalue())

        return result

    return wrapper


class PerformanceMonitor:
    """实时性能监控器"""

    def __init__(self, interval: float = 1.0):
        """初始化监控器

        Args:
            interval: 采样间隔（秒）
        """
        self.interval = interval
        self.measurements: List[Dict] = []

    def start(self):
        """开始监控"""
        import threading

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """停止监控"""
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()

    def _monitor_loop(self):
        """监控循环"""
        import os

        import psutil

        process = psutil.Process(os.getpid())

        while self.running:
            # 收集指标
            measurement = {
                "timestamp": time.time(),
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "threads": process.num_threads(),
            }

            self.measurements.append(measurement)
            time.sleep(self.interval)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.measurements:
            return {}

        cpu_values = [m["cpu_percent"] for m in self.measurements]
        memory_values = [m["memory_mb"] for m in self.measurements]

        return {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_avg": sum(memory_values) / len(memory_values),
            "memory_max": max(memory_values),
            "samples": len(self.measurements),
        }
