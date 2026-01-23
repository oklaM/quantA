"""
绩效分析模块
计算回测的各种性能指标
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.helpers import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
)
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """绩效指标数据类"""

    # 收益指标
    total_return: float = 0.0  # 总收益率
    annual_return: float = 0.0  # 年化收益率
    daily_return_mean: float = 0.0  # 日均收益率

    # 风险指标
    volatility: float = 0.0  # 波动率（年化）
    max_drawdown: float = 0.0  # 最大回撤
    downside_risk: float = 0.0  # 下行风险

    # 风险调整收益
    sharpe_ratio: float = 0.0  # 夏普比率
    sortino_ratio: float = 0.0  # 索提诺比率
    calmar_ratio: float = 0.0  # 卡玛比率

    # 交易统计
    total_trades: int = 0  # 总交易次数
    win_rate: float = 0.0  # 胜率
    profit_factor: float = 0.0  # 盈亏比
    avg_win: float = 0.0  # 平均盈利
    avg_loss: float = 0.0  # 平均亏损

    # 持仓统计
    avg_holding_period: float = 0.0  # 平均持仓天数

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "收益指标": {
                "总收益率": f"{self.total_return * 100:.2f}%",
                "年化收益率": f"{self.annual_return * 100:.2f}%",
                "日均收益率": f"{self.daily_return_mean * 100:.4f}%",
            },
            "风险指标": {
                "年化波动率": f"{self.volatility * 100:.2f}%",
                "最大回撤": f"{self.max_drawdown * 100:.2f}%",
                "下行风险": f"{self.downside_risk * 100:.2f}%",
            },
            "风险调整收益": {
                "夏普比率": f"{self.sharpe_ratio:.2f}",
                "索提诺比率": f"{self.sortino_ratio:.2f}",
                "卡玛比率": f"{self.calmar_ratio:.2f}",
            },
            "交易统计": {
                "总交易次数": self.total_trades,
                "胜率": f"{self.win_rate * 100:.2f}%",
                "盈亏比": f"{self.profit_factor:.2f}",
                "平均盈利": f"{self.avg_win:.2f}",
                "平均亏损": f"{self.avg_loss:.2f}",
            },
            "持仓统计": {
                "平均持仓天数": f"{self.avg_holding_period:.2f}",
            },
        }


class PerformanceAnalyzer:
    """绩效分析器"""

    def __init__(self, risk_free_rate: float = 0.03):
        """
        Args:
            risk_free_rate: 无风险利率（年化）
        """
        self.risk_free_rate = risk_free_rate
        self.metrics = PerformanceMetrics()

    def calculate(
        self,
        equity_curve: pd.DataFrame,
        positions: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
    ) -> PerformanceMetrics:
        """
        计算绩效指标

        Args:
            equity_curve: 净值曲线
            positions: 持仓列表
            trades: 交易列表

        Returns:
            绩效指标
        """
        if equity_curve.empty:
            logger.warning("净值曲线为空")
            return self.metrics

        # 计算收益率序列
        returns = equity_curve["total_value"].pct_change().dropna()

        # 收益指标
        self._calculate_return_metrics(equity_curve, returns)

        # 风险指标
        self._calculate_risk_metrics(equity_curve, returns)

        # 风险调整收益
        self._calculate_risk_adjusted_metrics(returns)

        # 交易统计
        if trades:
            self._calculate_trade_metrics(trades)

        return self.metrics

    def _calculate_return_metrics(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.Series,
    ):
        """计算收益指标"""
        # 总收益率
        initial_value = equity_curve["total_value"].iloc[0]
        final_value = equity_curve["total_value"].iloc[-1]
        self.metrics.total_return = (final_value - initial_value) / initial_value

        # 年化收益率
        days = len(equity_curve)
        years = days / 252  # 假设一年252个交易日
        if years > 0:
            self.metrics.annual_return = (1 + self.metrics.total_return) ** (1 / years) - 1

        # 日均收益率
        self.metrics.daily_return_mean = returns.mean()

    def _calculate_risk_metrics(
        self,
        equity_curve: pd.DataFrame,
        returns: pd.Series,
    ):
        """计算风险指标"""
        # 波动率（年化）
        self.metrics.volatility = calculate_volatility(returns)

        # 最大回撤
        self.metrics.max_drawdown = calculate_max_drawdown(equity_curve["total_value"])

        # 下行风险
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            self.metrics.downside_risk = negative_returns.std() * np.sqrt(252)

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series):
        """计算风险调整收益指标"""
        # 夏普比率
        self.metrics.sharpe_ratio = calculate_sharpe_ratio(
            returns,
            self.risk_free_rate,
        )

        # 索提诺比率
        self.metrics.sortino_ratio = calculate_sortino_ratio(
            returns,
            self.risk_free_rate,
        )

        # 卡玛比率 = 年化收益 / 最大回撤
        if self.metrics.max_drawdown != 0:
            self.metrics.calmar_ratio = abs(self.metrics.annual_return / self.metrics.max_drawdown)
        else:
            self.metrics.calmar_ratio = 0.0

    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]):
        """计算交易统计"""
        self.metrics.total_trades = len(trades)

        # 分析每笔交易的盈亏
        profits = []
        losses = []

        # 按股票分组计算盈亏
        symbol_trades = {}
        for trade in trades:
            symbol = trade.get("symbol")
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)

        # 计算每笔完整交易的盈亏
        for symbol, symbol_trade_list in symbol_trades.items():
            buys = [t for t in symbol_trade_list if t.get("side") == "buy"]
            sells = [t for t in symbol_trade_list if t.get("side") == "sell"]

            # 简化处理：计算总买入和总卖出
            total_buy_qty = sum(t.get("quantity", 0) for t in buys)
            total_buy_amount = sum(t.get("quantity", 0) * t.get("price", 0) for t in buys)
            total_sell_qty = sum(t.get("quantity", 0) for t in sells)
            total_sell_amount = sum(t.get("quantity", 0) * t.get("price", 0) for t in sells)

            # 计算盈亏
            if total_sell_qty > 0:
                avg_buy_price = total_buy_amount / total_buy_qty if total_buy_qty > 0 else 0
                avg_sell_price = total_sell_amount / total_sell_qty
                pnl = (avg_sell_price - avg_buy_price) * total_sell_qty

                if pnl > 0:
                    profits.append(pnl)
                else:
                    losses.append(abs(pnl))

        # 胜率
        total_trades_with_result = len(profits) + len(losses)
        if total_trades_with_result > 0:
            self.metrics.win_rate = len(profits) / total_trades_with_result

        # 盈亏比
        if losses and profits:
            self.metrics.profit_factor = sum(profits) / sum(losses)

        # 平均盈亏
        if profits:
            self.metrics.avg_win = np.mean(profits)
        if losses:
            self.metrics.avg_loss = np.mean(losses)


def create_performance_report(
    equity_curve: pd.DataFrame,
    positions: List[Dict[str, Any]],
    trades: List[Dict[str, Any]],
    risk_free_rate: float = 0.03,
) -> str:
    """
    创建绩效报告

    Args:
        equity_curve: 净值曲线
        positions: 持仓列表
        trades: 交易列表
        risk_free_rate: 无风险利率

    Returns:
        格式化的绩效报告字符串
    """
    analyzer = PerformanceAnalyzer(risk_free_rate)
    metrics = analyzer.calculate(equity_curve, positions, trades)
    metrics_dict = metrics.to_dict()

    # 构建报告
    report_lines = [
        "=" * 60,
        "回测绩效报告".center(60),
        "=" * 60,
    ]

    for category, values in metrics_dict.items():
        report_lines.append(f"\n【{category}】")
        for name, value in values.items():
            report_lines.append(f"  {name}: {value}")

    report_lines.append("\n" + "=" * 60)

    return "\n".join(report_lines)


__all__ = [
    "PerformanceMetrics",
    "PerformanceAnalyzer",
    "create_performance_report",
]
