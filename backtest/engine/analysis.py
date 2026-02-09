"""
绩效分析模块
提供回测绩效分析和可视化功能
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """
    绩效分析器

    计算各种绩效指标
    """

    def __init__(self):
        """初始化分析器"""
        self.metrics: Dict[str, Any] = {}

    def calculate_returns(
        self,
        equity_curve: pd.Series,
    ) -> pd.Series:
        """
        计算收益率序列

        Args:
            equity_curve: 权益曲线

        Returns:
            收益率序列
        """
        return equity_curve.pct_change().fillna(0)

    def calculate_total_return(
        self,
        equity_curve: pd.Series,
    ) -> float:
        """
        计算总收益率

        Args:
            equity_curve: 权益曲线

        Returns:
            总收益率
        """
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    def calculate_annual_return(
        self,
        equity_curve: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        计算年化收益率

        Args:
            equity_curve: 权益曲线
            periods_per_year: 每年交易周期数

        Returns:
            年化收益率
        """
        total_return = self.calculate_total_return(equity_curve)
        n_periods = len(equity_curve) - 1
        if n_periods <= 0:
            return 0.0

        return (1 + total_return) ** (periods_per_year / n_periods) - 1

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.03,
        periods_per_year: int = 252,
    ) -> float:
        """
        计算夏普比率

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易周期数

        Returns:
            夏普比率
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

    def calculate_max_drawdown(
        self,
        equity_curve: pd.Series,
    ) -> float:
        """
        计算最大回撤

        Args:
            equity_curve: 权益曲线

        Returns:
            最大回撤
        """
        if len(equity_curve) < 2:
            return 0.0

        cumulative = equity_curve
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max

        return drawdown.min()

    def calculate_win_rate(
        self,
        trades: pd.DataFrame,
    ) -> float:
        """
        计算胜率

        Args:
            trades: 交易记录

        Returns:
            胜率
        """
        if len(trades) == 0:
            return 0.0

        winning_trades = trades[trades["pnl"] > 0]
        return len(winning_trades) / len(trades)

    def analyze(
        self,
        equity_curve: pd.Series,
        trades: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        综合分析

        Args:
            equity_curve: 权益曲线
            trades: 交易记录
            benchmark_returns: 基准收益率

        Returns:
            分析结果字典
        """
        returns = self.calculate_returns(equity_curve)

        metrics = {
            "total_return": self.calculate_total_return(equity_curve),
            "annual_return": self.calculate_annual_return(equity_curve),
            "sharpe_ratio": self.calculate_sharpe_ratio(returns),
            "max_drawdown": self.calculate_max_drawdown(equity_curve),
            "final_equity": equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
        }

        # 交易相关指标
        if trades is not None and len(trades) > 0:
            metrics["win_rate"] = self.calculate_win_rate(trades)
            metrics["total_trades"] = len(trades)
            metrics["avg_pnl"] = trades["pnl"].mean()

        # 基准相关指标
        if benchmark_returns is not None:
            metrics["benchmark_return"] = benchmark_returns.sum()
            # 计算超额收益
            metrics["excess_return"] = metrics["total_return"] - metrics["benchmark_return"]

        self.metrics = metrics
        return metrics


__all__ = ["PerformanceAnalyzer"]
