"""
完整回测引擎
整合所有组件，提供完整的回测功能
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.engine.a_share_rules import AShareRulesEngine
from backtest.engine.data_handler import DataHandler, SimpleDataHandler
from backtest.engine.event_engine import BarEvent, EventQueue
from backtest.engine.execution import ExecutionHandler
from backtest.engine.portfolio import Portfolio
from backtest.engine.strategy import Strategy
from utils.logging import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    完整回测引擎
    整合所有回测组件
    """

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: Strategy,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0001,
    ):
        """
        Args:
            data: {symbol: DataFrame} 格式的数据
            strategy: 策略实例
            initial_cash: 初始资金
            commission_rate: 佣金率
            slippage_rate: 滑点率
        """
        # 创建组件
        self.data_handler = SimpleDataHandler(data)
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
        )

        # 创建执行处理器
        from backtest.engine.execution import SimulationExecutionHandler

        self.execution_handler = SimulationExecutionHandler(
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
        )

        # 策略
        self.strategy = strategy

        # 创建事件队列
        self.strategy.event_queue = EventQueue()

        # 连接组件
        self.strategy.set_data_handler(self.data_handler)
        self.strategy.set_portfolio(self.portfolio)

        # 统计
        self.stats = {
            "total_bars": 0,
            "total_orders": 0,
            "total_fills": 0,
        }

        logger.info("回测引擎初始化完成")

    def run(self) -> Dict[str, Any]:
        """
        运行回测

        Returns:
            回测结果
        """
        logger.info("开始回测...")

        # 获取第一个股票的数据长度作为回测长度
        if not self.data_handler.symbols:
            logger.error("没有数据")
            return {}

        first_symbol = self.data_handler.symbols[0]

        # 重置
        self.data_handler.reset()
        self.portfolio.reset()

        # 主循环
        while True:
            # 更新所有股票的数据
            has_data = False

            for symbol in self.data_handler.symbols:
                bar = self.data_handler.get_next_bar(symbol)

                if bar is not None:
                    has_data = True

                    # 更新执行处理器的当前数据
                    self.execution_handler.update_bar(
                        {
                            "symbol": bar.symbol,
                            "open": bar.open,
                            "high": bar.high,
                            "low": bar.low,
                            "close": bar.close,
                            "volume": bar.volume,
                            "prev_close": bar.data.get("prev_close", bar.close),
                        }
                    )

                    # 更新投资组合的当前价格
                    self.portfolio.update_current_price(symbol, bar.close)

            if not has_data:
                break

            # 处理事件
            self._process_events()

            # 更新投资组合
            self.portfolio.update_total_value()

            # 获取当前时间（使用第一个股票的时间）
            current_bar = self.data_handler.get_current_bar(first_symbol)
            if current_bar:
                self.portfolio.add_to_equity_curve(current_bar.datetime)

            self.stats["total_bars"] += 1

        logger.info("回测完成")
        return self._get_results()

    def _process_events(self):
        """处理事件"""
        # 触发策略
        for symbol in self.data_handler.symbols:
            bar = self.data_handler.get_current_bar(symbol)
            if bar:
                # 触发策略
                self.strategy.on_bar(bar)

        # 处理订单队列
        if self.strategy.event_queue:
            while not self.strategy.event_queue.empty():
                event = self.strategy.event_queue.get()

                if hasattr(event, "side"):  # 订单事件
                    # 更新事件时间
                    if event.datetime is None:
                        current_bar = self.data_handler.get_current_bar(event.symbol)
                        if current_bar:
                            event.datetime = current_bar.datetime

                    # 执行订单
                    fill_event = self.execution_handler.execute_order(event)

                    if fill_event:
                        # 更新投资组合
                        self.portfolio.update_fill(fill_event)
                        self.stats["total_fills"] += 1
                    else:
                        self.stats["total_orders"] += 1

    def _get_results(self) -> Dict[str, Any]:
        """获取回测结果"""
        account_info = self.portfolio.get_account_info()
        equity_curve = self.portfolio.get_equity_curve()

        # 计算性能指标
        if not equity_curve.empty:
            returns = equity_curve["return"].pct_change().dropna()

            from utils.helpers import (
                calculate_max_drawdown,
                calculate_sharpe_ratio,
                calculate_volatility,
            )

            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(equity_curve["total_value"])
            volatility = calculate_volatility(returns)
        else:
            sharpe = 0.0
            max_dd = 0.0
            volatility = 0.0

        return {
            "account": account_info,
            "equity_curve": equity_curve,
            "positions": self.portfolio.get_all_positions(),
            "stats": self.stats,
            "performance": {
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "volatility": volatility,
            },
        }


def run_backtest(
    data: Dict[str, pd.DataFrame],
    strategy: Strategy,
    initial_cash: float = 1000000.0,
    commission_rate: float = 0.0003,
    slippage_rate: float = 0.0001,
) -> Dict[str, Any]:
    """
    便捷函数：运行回测

    Args:
        data: {symbol: DataFrame} 数据
        strategy: 策略
        initial_cash: 初始资金
        commission_rate: 佣金率
        slippage_rate: 滑点率

    Returns:
        回测结果
    """
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    return engine.run()


__all__ = [
    "BacktestEngine",
    "run_backtest",
]
