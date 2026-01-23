"""
绩效可视化模块测试
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from backtest.visualization import InteractiveVisualizer, PerformanceVisualizer


@pytest.fixture
def sample_results():
    """生成示例回测结果"""
    equity_values = np.array([1000000 + i * 1000 + np.random.randn() * 5000 for i in range(100)])

    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq="D")

    return {
        "total_return": 0.15,
        "annual_return": 0.18,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "max_drawdown": -0.08,
        "win_rate": 0.55,
        "profit_factor": 1.8,
        "avg_win": 5000,
        "avg_loss": -3000,
        "best_trade": 15000,
        "worst_trade": -10000,
        "equity_curve": equity_values,
    }


@pytest.fixture
def sample_equity_curve():
    """生成示例权益曲线"""
    np.random.seed(42)
    values = np.array([1000000 + i * 1000 + np.random.randn() * 5000 for i in range(100)])
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq="D")
    return pd.Series(values, index=dates)


@pytest.fixture
def sample_benchmark_curve():
    """生成示例基准曲线"""
    np.random.seed(43)
    values = np.array([1000000 + i * 800 + np.random.randn() * 4000 for i in range(100)])
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq="D")
    return pd.Series(values, index=dates)


@pytest.mark.backtest
class TestPerformanceVisualizer:
    """测试绩效可视化器"""

    def test_visualizer_initialization(self, sample_results, sample_equity_curve):
        """测试初始化"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        assert visualizer.results == sample_results
        assert len(visualizer.equity_curve) == 100
        assert visualizer.benchmark_curve is None
        assert len(visualizer.returns) == 100
        assert len(visualizer.drawdown) == 100

    def test_visualizer_with_benchmark(
        self, sample_results, sample_equity_curve, sample_benchmark_curve
    ):
        """测试带基准的可视化器"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
            benchmark_curve=sample_benchmark_curve,
        )

        assert visualizer.benchmark_curve is not None
        assert len(visualizer.benchmark_curve) == 100

    def test_calculate_drawdown(self, sample_results, sample_equity_curve):
        """测试回撤计算"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        # 回撤应该 <= 0
        assert (visualizer.drawdown <= 0).all()

        # 回撤长度应该与权益曲线相同
        assert len(visualizer.drawdown) == len(sample_equity_curve)

    def test_plot_equity_curve(self, sample_results, sample_equity_curve, tmp_path):
        """测试绘制权益曲线"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "equity_curve.png"
        fig = visualizer.plot_equity_curve(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_equity_curve_with_benchmark(
        self, sample_results, sample_equity_curve, sample_benchmark_curve, tmp_path
    ):
        """测试绘制带基准的权益曲线"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
            benchmark_curve=sample_benchmark_curve,
        )

        save_path = tmp_path / "equity_curve_benchmark.png"
        fig = visualizer.plot_equity_curve(show_benchmark=True, save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_drawdown(self, sample_results, sample_equity_curve, tmp_path):
        """测试绘制回撤图"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "drawdown.png"
        fig = visualizer.plot_drawdown(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_returns_distribution(self, sample_results, sample_equity_curve, tmp_path):
        """测试绘制收益分布"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "returns_dist.png"
        fig = visualizer.plot_returns_distribution(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_monthly_returns(self, sample_results, sample_equity_curve, tmp_path):
        """测试绘制月度收益"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        # 需要足够长的数据
        long_equity = pd.concat([sample_equity_curve] * 8, ignore_index=True)
        long_equity.index = pd.date_range(
            start=datetime.now() - timedelta(days=800), periods=800, freq="D"
        )

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=long_equity,
        )

        save_path = tmp_path / "monthly_returns.png"
        fig = visualizer.plot_monthly_returns(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_rolling_metrics(self, sample_results, sample_equity_curve, tmp_path):
        """测试绘制滚动指标"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "rolling_metrics.png"
        fig = visualizer.plot_rolling_metrics(window=30, save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_performance_dashboard(self, sample_results, sample_equity_curve, tmp_path):
        """测试绘制绩效仪表板"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "dashboard.png"
        fig = visualizer.plot_performance_dashboard(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_create_summary_table(self, sample_results, sample_equity_curve, tmp_path):
        """测试创建摘要表格"""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not installed")

        visualizer = PerformanceVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        # 不保存文件
        summary_df = visualizer.create_summary_table()

        assert isinstance(summary_df, pd.DataFrame)
        assert len(summary_df) == 11  # 11个指标
        assert "Metric" in summary_df.columns
        assert "Value" in summary_df.columns

        # 保存文件
        save_path = tmp_path / "summary.csv"
        summary_df = visualizer.create_summary_table(save_path=str(save_path))

        assert save_path.exists()

    def test_visualizer_requires_matplotlib(self, sample_results, sample_equity_curve):
        """测试可视化器需要matplotlib"""
        if MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib is installed")

        with pytest.raises(ImportError):
            visualizer = PerformanceVisualizer(
                results=sample_results,
                equity_curve=sample_equity_curve,
            )


@pytest.mark.backtest
class TestInteractiveVisualizer:
    """测试交互式可视化器"""

    def test_interactive_visualizer_initialization(self, sample_results, sample_equity_curve):
        """测试初始化"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")

        visualizer = InteractiveVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        assert visualizer.results == sample_results
        assert len(visualizer.equity_curve) == 100

    def test_plot_interactive_equity_curve(self, sample_results, sample_equity_curve, tmp_path):
        """测试创建交互式权益曲线"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")

        visualizer = InteractiveVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "interactive_equity.html"
        fig = visualizer.plot_interactive_equity_curve(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_plot_interactive_dashboard(self, sample_results, sample_equity_curve, tmp_path):
        """测试创建交互式仪表板"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")

        visualizer = InteractiveVisualizer(
            results=sample_results,
            equity_curve=sample_equity_curve,
        )

        save_path = tmp_path / "interactive_dashboard.html"
        fig = visualizer.plot_interactive_dashboard(save_path=str(save_path))

        assert fig is not None
        assert save_path.exists()

    def test_interactive_visualizer_requires_plotly(self, sample_results, sample_equity_curve):
        """测试交互式可视化器需要plotly"""
        if PLOTLY_AVAILABLE:
            pytest.skip("plotly is installed")

        with pytest.raises(ImportError):
            visualizer = InteractiveVisualizer(
                results=sample_results,
                equity_curve=sample_equity_curve,
            )
