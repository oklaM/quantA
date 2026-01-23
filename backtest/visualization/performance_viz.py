"""
回测绩效可视化模块
提供丰富的图表和交互式可视化功能
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    # 创建一个假的go模块用于类型注解
    class DummyFigure:
        pass

    class DummyGo:
        Figure = DummyFigure

    go = DummyGo()
    PLOTLY_AVAILABLE = False

from utils.logging import get_logger

logger = get_logger(__name__)

# 设置样式
if MATPLOTLIB_AVAILABLE:
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")


class PerformanceVisualizer:
    """
    绩效可视化器

    创建各种图表来展示回测绩效
    """

    def __init__(
        self,
        results: Dict[str, Any],
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
    ):
        """
        Args:
            results: 回测结果字典
            equity_curve: 权益曲线
            benchmark_curve: 基准曲线
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib未安装，请运行: pip install matplotlib")

        self.results = results
        self.equity_curve = equity_curve
        self.benchmark_curve = benchmark_curve

        # 计算额外指标
        self.returns = self.equity_curve.pct_change().fillna(0)
        self.cumulative_returns = (1 + self.returns).cumprod()

        if benchmark_curve is not None:
            self.benchmark_returns = benchmark_curve.pct_change().fillna(0)

        # 计算回撤
        self.drawdown = self._calculate_drawdown()

    def _calculate_drawdown(self) -> pd.Series:
        """计算回撤"""
        cumulative = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cumulative) / cumulative
        return drawdown

    def plot_equity_curve(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
        show_benchmark: bool = True,
    ) -> plt.Figure:
        """
        绘制权益曲线

        Args:
            figsize: 图表大小
            save_path: 保存路径
            show_benchmark: 是否显示基准曲线

        Returns:
            Figure对象
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制权益曲线
        ax.plot(
            self.equity_curve.index,
            self.equity_curve.values,
            label="Strategy",
            linewidth=2,
        )

        # 绘制基准曲线
        if show_benchmark and self.benchmark_curve is not None:
            ax.plot(
                self.benchmark_curve.index,
                self.benchmark_curve.values,
                label="Benchmark",
                linewidth=2,
                alpha=0.7,
            )

        ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"权益曲线已保存到: {save_path}")

        return fig

    def plot_drawdown(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制回撤图

        Args:
            figsize: 图表大小
            save_path: 保存路径

        Returns:
            Figure对象
        """
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制回撤
        ax.fill_between(
            self.drawdown.index,
            self.drawdown.values,
            0,
            alpha=0.3,
            color="red",
        )
        ax.plot(
            self.drawdown.index,
            self.drawdown.values,
            color="red",
            linewidth=1.5,
        )

        ax.set_title("Drawdown", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Drawdown", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 格式化为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"回撤图已保存到: {save_path}")

        return fig

    def plot_returns_distribution(
        self,
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制收益分布图

        Args:
            figsize: 图表大小
            save_path: 保存路径

        Returns:
            Figure对象
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 直方图
        axes[0].hist(
            self.returns.values,
            bins=50,
            alpha=0.7,
            color="blue",
            edgecolor="black",
        )
        axes[0].axvline(
            self.returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.returns.mean():.4f}",
        )
        axes[0].set_title("Returns Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Returns", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Q-Q图
        from scipy import stats

        stats.probplot(self.returns.values, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot", fontsize=12, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"收益分布图已保存到: {save_path}")

        return fig

    def plot_monthly_returns(
        self,
        figsize: tuple = (14, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制月度收益热力图

        Args:
            figsize: 图表大小
            save_path: 保存路径

        Returns:
            Figure对象
        """
        # 计算月度收益
        monthly_returns = self.returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

        # 创建年月矩阵
        monthly_returns_df = pd.DataFrame(
            {
                "year": monthly_returns.index.year,
                "month": monthly_returns.index.month,
                "returns": monthly_returns.values,
            }
        )

        # 透视表
        pivot_table = monthly_returns_df.pivot(
            index="year",
            columns="month",
            values="returns",
        )

        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Returns"},
            ax=ax,
        )

        ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Year", fontsize=12)

        # 设置月份标签
        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        ax.set_xticklabels(month_labels, rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"月度收益热力图已保存到: {save_path}")

        return fig

    def plot_rolling_metrics(
        self,
        window: int = 60,
        figsize: tuple = (12, 8),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制滚动指标

        Args:
            window: 滚动窗口
            figsize: 图表大小
            save_path: 保存路径

        Returns:
            Figure对象
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # 滚动夏普比率
        rolling_sharpe = self.returns.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )

        axes[0].plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            linewidth=2,
            label=f"{window}-day Rolling Sharpe",
        )
        axes[0].axhline(
            y=1.0,
            color="red",
            linestyle="--",
            linewidth=1,
            label="Target (1.0)",
        )
        axes[0].set_title("Rolling Sharpe Ratio", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Sharpe Ratio", fontsize=10)
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        # 滚动波动率
        rolling_volatility = self.returns.rolling(window).std() * np.sqrt(252)

        axes[1].plot(
            rolling_volatility.index,
            rolling_volatility.values,
            linewidth=2,
            label=f"{window}-day Rolling Volatility",
            color="orange",
        )
        axes[1].set_title("Rolling Volatility", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Date", fontsize=10)
        axes[1].set_ylabel("Volatility (Annualized)", fontsize=10)
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        # 格式化x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"滚动指标图已保存到: {save_path}")

        return fig

    def plot_performance_dashboard(
        self,
        figsize: tuple = (16, 12),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制综合绩效仪表板

        Args:
            figsize: 图表大小
            save_path: 保存路径

        Returns:
            Figure对象
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 权益曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.equity_curve.index, self.equity_curve.values, linewidth=2, label="Strategy")

        if self.benchmark_curve is not None:
            ax1.plot(
                self.benchmark_curve.index,
                self.benchmark_curve.values,
                linewidth=2,
                alpha=0.7,
                label="Benchmark",
            )

        ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Portfolio Value")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # 2. 回撤
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.fill_between(self.drawdown.index, self.drawdown.values, 0, alpha=0.3, color="red")
        ax2.plot(self.drawdown.index, self.drawdown.values, color="red", linewidth=1.5)
        ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # 3. 收益分布
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(self.returns.values, bins=50, alpha=0.7, color="blue", edgecolor="black")
        ax3.axvline(
            self.returns.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {self.returns.mean():.4f}",
        )
        ax3.set_title("Returns Distribution", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Returns")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 滚动夏普比率
        ax4 = fig.add_subplot(gs[2, 0])
        rolling_sharpe = self.returns.rolling(60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, label="60-day")
        ax4.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Target (1.0)")
        ax4.set_title("Rolling Sharpe Ratio", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.legend(loc="best")
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

        # 5. 累计收益
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(
            self.cumulative_returns.index,
            self.cumulative_returns.values,
            linewidth=2,
            label="Strategy",
        )

        if self.benchmark_curve is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            ax5.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                linewidth=2,
                alpha=0.7,
                label="Benchmark",
            )

        ax5.set_title("Cumulative Returns", fontsize=12, fontweight="bold")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Cumulative Returns")
        ax5.legend(loc="best")
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2%}"))
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)

        fig.suptitle("Performance Dashboard", fontsize=16, fontweight="bold", y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"绩效仪表板已保存到: {save_path}")

        return fig

    def create_summary_table(
        self,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        创建绩效摘要表格

        Args:
            save_path: 保存路径

        Returns:
            摘要DataFrame
        """
        summary_data = {
            "Metric": [
                "Total Return",
                "Annual Return",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
                "Win Rate",
                "Profit Factor",
                "Avg Win",
                "Avg Loss",
                "Best Trade",
                "Worst Trade",
            ],
            "Value": [
                f"{self.results.get('total_return', 0):.2%}",
                f"{self.results.get('annual_return', 0):.2%}",
                f"{self.results.get('sharpe_ratio', 0):.2f}",
                f"{self.results.get('sortino_ratio', 0):.2f}",
                f"{self.results.get('max_drawdown', 0):.2%}",
                f"{self.results.get('win_rate', 0):.2%}",
                f"{self.results.get('profit_factor', 0):.2f}",
                f"{self.results.get('avg_win', 0):.2f}",
                f"{self.results.get('avg_loss', 0):.2f}",
                f"{self.results.get('best_trade', 0):.2f}",
                f"{self.results.get('worst_trade', 0):.2f}",
            ],
        }

        summary_df = pd.DataFrame(summary_data)

        if save_path:
            summary_df.to_csv(save_path, index=False)
            logger.info(f"绩效摘要已保存到: {save_path}")

        return summary_df


class InteractiveVisualizer:
    """
    交互式可视化器

    使用Plotly创建交互式图表
    """

    def __init__(
        self,
        results: Dict[str, Any],
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
    ):
        """
        Args:
            results: 回测结果
            equity_curve: 权益曲线
            benchmark_curve: 基准曲线
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly未安装，请运行: pip install plotly")

        self.results = results
        self.equity_curve = equity_curve
        self.benchmark_curve = benchmark_curve
        self.returns = equity_curve.pct_change().fillna(0)

    def plot_interactive_equity_curve(self, save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式权益曲线

        Args:
            save_path: 保存路径

        Returns:
            Plotly Figure对象
        """
        fig = go.Figure()

        # 策略曲线
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve.values,
                mode="lines",
                name="Strategy",
                line=dict(width=2),
            )
        )

        # 基准曲线
        if self.benchmark_curve is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.benchmark_curve.index,
                    y=self.benchmark_curve.values,
                    mode="lines",
                    name="Benchmark",
                    line=dict(width=2),
                )
            )

        fig.update_layout(
            title="Interactive Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            template="plotly_dark",
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"交互式权益曲线已保存到: {save_path}")

        return fig

    def plot_interactive_dashboard(self, save_path: Optional[str] = None) -> go.Figure:
        """
        创建交互式仪表板

        Args:
            save_path: 保存路径

        Returns:
            Plotly Figure对象
        """
        # 创建子图
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Equity Curve", "Drawdown", "Returns", "Rolling Sharpe"),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 权益曲线
        fig.add_trace(
            go.Scatter(
                x=self.equity_curve.index,
                y=self.equity_curve.values,
                name="Equity",
                line=dict(width=2),
            ),
            row=1,
            col=1,
        )

        # 回撤
        drawdown = (self.equity_curve - self.equity_curve.cummax()) / self.equity_curve.cummax()
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        # 收益
        fig.add_trace(
            go.Scatter(
                x=self.returns.index,
                y=self.returns.values,
                mode="markers",
                name="Returns",
                marker=dict(size=3),
            ),
            row=2,
            col=1,
        )

        # 滚动夏普
        rolling_sharpe = self.returns.rolling(60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                name="Rolling Sharpe",
                line=dict(width=2),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title_text="Interactive Performance Dashboard",
            showlegend=True,
            template="plotly_dark",
            height=800,
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"交互式仪表板已保存到: {save_path}")

        return fig


__all__ = [
    "PerformanceVisualizer",
    "InteractiveVisualizer",
]
