"""
组合回测模块
支持多策略、多资产组合回测
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import Strategy
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """持仓信息"""

    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # 在组合中的权重

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "weight": self.weight,
        }


@dataclass
class StrategyAllocation:
    """策略配置"""

    strategy: Strategy
    symbols: List[str]
    weight: float  # 资金分配权重
    max_position: Optional[float] = None  # 最大持仓比例

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy.__class__.__name__,
            "symbols": self.symbols,
            "weight": self.weight,
            "max_position": self.max_position,
        }


class Portfolio:
    """
    投资组合

    管理多个策略和资产的持仓
    """

    def __init__(
        self,
        initial_cash: float,
        strategies: List[StrategyAllocation],
    ):
        """
        Args:
            initial_cash: 初始资金
            strategies: 策略配置列表
        """
        self.initial_cash = initial_cash
        self.strategies = strategies

        # 验证权重
        total_weight = sum(s.weight for s in strategies)
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(f"策略权重总和必须为1.0，当前为{total_weight}")

        # 资金分配
        self.cash_allocations = {
            i: initial_cash * strategy.weight for i, strategy in enumerate(strategies)
        }

        # 持仓
        self.positions: Dict[str, Position] = {}  # symbol -> Position

        # 现金（按策略）
        self.strategy_cash: Dict[int, float] = self.cash_allocations.copy()

        # 历史记录
        self.equity_curve: List[float] = []
        self.dates: List[pd.Timestamp] = []

        logger.info(f"组合初始化: 初始资金={initial_cash:,.2f}, 策略数={len(strategies)}")

    def get_total_value(self, prices: Dict[str, float]) -> float:
        """
        获取组合总价值

        Args:
            prices: 当前价格字典 {symbol: price}

        Returns:
            总价值
        """
        total_value = sum(self.strategy_cash.values())

        for symbol, position in self.positions.items():
            current_price = prices.get(symbol, position.current_price)
            total_value += position.quantity * current_price

        return total_value

    def get_strategy_values(self, prices: Dict[str, float]) -> Dict[int, float]:
        """
        获取各策略的价值

        Args:
            prices: 当前价格字典

        Returns:
            {strategy_id: value}
        """
        values = {}

        # 按策略汇总持仓价值
        strategy_positions = defaultdict(list)
        for symbol, position in self.positions.items():
            # 找到该股票属于哪个策略
            for i, strategy_alloc in enumerate(self.strategies):
                if symbol in strategy_alloc.symbols:
                    current_price = prices.get(symbol, position.current_price)
                    strategy_positions[i].append(position.quantity * current_price)
                    break

        # 计算各策略价值
        for i in range(len(self.strategies)):
            position_value = sum(strategy_positions[i])
            cash_value = self.strategy_cash[i]
            values[i] = position_value + cash_value

        return values

    def update_position(
        self,
        strategy_id: int,
        symbol: str,
        quantity: int,
        price: float,
    ):
        """
        更新持仓

        Args:
            strategy_id: 策略ID
            symbol: 股票代码
            quantity: 数量（正数买入，负数卖出）
            price: 价格
        """
        current_cash = self.strategy_cash[strategy_id]

        # 检查资金是否足够
        required_cash = abs(quantity) * price
        if quantity > 0 and current_cash < required_cash:
            logger.warning(
                f"策略{strategy_id}资金不足: 需要{required_cash:,.2f}, 可用{current_cash:,.2f}"
            )
            return

        # 更新持仓
        if symbol in self.positions:
            position = self.positions[symbol]
            old_quantity = position.quantity
            position.quantity += quantity

            # 如果清仓，删除持仓
            if position.quantity == 0:
                del self.positions[symbol]
                logger.info(f"策略{strategy_id}清仓 {symbol}")
            else:
                # 更新成本价
                if quantity > 0:
                    total_cost = old_quantity * position.entry_price + quantity * price
                    position.entry_price = total_cost / position.quantity

                position.current_price = price
                position.market_value = position.quantity * price
                position.unrealized_pnl = (price - position.entry_price) * position.quantity

                logger.info(
                    f"策略{strategy_id}更新持仓 {symbol}: {old_quantity} -> {position.quantity}"
                )
        else:
            # 新建持仓
            if quantity > 0:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    market_value=quantity * price,
                    unrealized_pnl=0.0,
                    weight=0.0,  # 稍后计算
                )
                logger.info(f"策略{strategy_id}新建持仓 {symbol}: {quantity}股 @{price:.2f}")

        # 更新现金
        self.strategy_cash[strategy_id] -= quantity * price

    def record_equity(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        记录权益

        Args:
            date: 日期
            prices: 价格字典
        """
        total_value = self.get_total_value(prices)
        self.equity_curve.append(total_value)
        self.dates.append(date)

        # 更新持仓权重
        for position in self.positions.values():
            current_price = prices.get(position.symbol, position.current_price)
            position.market_value = position.quantity * current_price
            position.weight = position.market_value / total_value if total_value > 0 else 0

    def get_summary(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        获取组合摘要

        Args:
            prices: 当前价格字典

        Returns:
            摘要字典
        """
        total_value = self.get_total_value(prices)
        total_return = (total_value - self.initial_cash) / self.initial_cash

        # 策略价值
        strategy_values = self.get_strategy_values(prices)

        # 持仓统计
        long_positions = [p for p in self.positions.values() if p.quantity > 0]
        short_positions = [p for p in self.positions.values() if p.quantity < 0]

        summary = {
            "total_value": total_value,
            "total_return": total_return,
            "cash": sum(self.strategy_cash.values()),
            "cash_ratio": sum(self.strategy_cash.values()) / total_value if total_value > 0 else 0,
            "num_positions": len(self.positions),
            "long_positions": len(long_positions),
            "short_positions": len(short_positions),
            "strategy_values": strategy_values,
            "positions": [p.to_dict() for p in self.positions.values()],
        }

        return summary


class PortfolioBacktestEngine:
    """
    组合回测引擎

    支持多个策略同时运行
    """

    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategies: List[StrategyAllocation],
        initial_cash: float = 10000000.0,
        commission_rate: float = 0.0003,
    ):
        """
        Args:
            data_dict: 数据字典 {symbol: DataFrame}
            strategies: 策略配置列表
            initial_cash: 初始资金
            commission_rate: 手续费率
        """
        self.data_dict = data_dict
        self.strategies = strategies
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate

        # 创建组合
        self.portfolio = Portfolio(initial_cash, strategies)

        # 回测结果
        self.results: Dict[str, Any] = {}

        logger.info(f"组合回测引擎初始化: 股票数={len(data_dict)}, 策略数={len(strategies)}")

    def run(self) -> Dict[str, Any]:
        """
        运行组合回测

        Returns:
            回测结果
        """
        logger.info("开始组合回测...")

        # 获取所有日期
        all_dates = set()
        for df in self.data_dict.values():
            all_dates.update(df["datetime"])
        all_dates = sorted(all_dates)

        # 对每个策略创建独立的回测引擎
        strategy_engines = {}
        for i, strategy_alloc in enumerate(self.strategies):
            strategy_data = {
                symbol: self.data_dict[symbol]
                for symbol in strategy_alloc.symbols
                if symbol in self.data_dict
            }

            # 这里简化处理：为每个股票创建独立的引擎
            # 实际应用中可能需要更复杂的处理
            strategy_engines[i] = {
                "strategy": strategy_alloc.strategy,
                "symbols": strategy_alloc.symbols,
            }

        # 逐日回测
        for date in all_dates:
            # 获取当日价格
            daily_prices = {}
            for symbol, df in self.data_dict.items():
                daily_data = df[df["datetime"] == date]
                if not daily_data.empty:
                    daily_prices[symbol] = daily_data.iloc[0]["close"]

            # 对每个策略生成交易信号
            for i, strategy_alloc in enumerate(self.strategies):
                strategy = strategy_alloc.strategy

                for symbol in strategy_alloc.symbols:
                    if symbol not in self.data_dict:
                        continue

                    # 获取该股票的历史数据
                    symbol_data = self.data_dict[symbol]
                    historical_data = symbol_data[symbol_data["datetime"] <= date]

                    if len(historical_data) < 2:
                        continue

                    # 创建临时事件
                    from backtest.engine.event_engine import BarEvent

                    bar_event = BarEvent(
                        datetime=date,
                        symbol=symbol,
                        open=historical_data.iloc[-1]["open"],
                        high=historical_data.iloc[-1]["high"],
                        low=historical_data.iloc[-1]["low"],
                        close=historical_data.iloc[-1]["close"],
                        volume=int(historical_data.iloc[-1]["volume"]),
                    )

                    # 获取策略信号
                    strategy.on_bar(bar_event)

                    # 检查是否有待执行的订单
                    if hasattr(strategy, "pending_orders") and strategy.pending_orders:
                        for order in strategy.pending_orders:
                            if order.symbol == symbol:
                                # 计算交易数量（简化处理）
                                available_cash = self.portfolio.strategy_cash[i]
                                price = daily_prices.get(symbol, historical_data.iloc[-1]["close"])

                                if order.direction == "BUY":
                                    max_quantity = int(
                                        available_cash / price / (1 + self.commission_rate)
                                    )
                                    quantity = (
                                        min(order.quantity, max_quantity)
                                        if order.quantity > 0
                                        else max_quantity
                                    )
                                    quantity = max(0, quantity)

                                    if quantity > 0:
                                        self.portfolio.update_position(i, symbol, quantity, price)
                                elif order.direction == "SELL":
                                    # 卖出逻辑需要根据当前持仓
                                    if symbol in self.portfolio.positions:
                                        position = self.portfolio.positions[symbol]
                                        quantity = (
                                            min(abs(order.quantity), position.quantity)
                                            if order.quantity < 0
                                            else position.quantity
                                        )
                                        self.portfolio.update_position(i, symbol, -quantity, price)

                        # 清空已处理订单
                        strategy.pending_orders = []

            # 记录当日权益
            self.portfolio.record_equity(date, daily_prices)

        # 计算回测结果
        self.results = self._calculate_results()

        logger.info("组合回测完成！")

        return self.results

    def _calculate_results(self) -> Dict[str, Any]:
        """计算回测结果"""
        if not self.portfolio.equity_curve:
            return {}

        equity_series = pd.Series(
            self.portfolio.equity_curve,
            index=self.portfolio.dates,
        )

        returns = equity_series.pct_change().fillna(0)

        # 基本指标
        total_return = (equity_series.iloc[-1] - self.initial_cash) / self.initial_cash
        n_days = len(equity_series)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # 风险指标
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (
            returns.mean() / (returns.std() + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
        )

        # 回撤
        cumulative = equity_series.cummax()
        drawdown = (equity_series - cumulative) / cumulative
        max_drawdown = drawdown.min()

        # 各策略收益
        final_prices = {symbol: df.iloc[-1]["close"] for symbol, df in self.data_dict.items()}
        strategy_values = self.portfolio.get_strategy_values(final_prices)

        results = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "equity_curve": equity_series.values.tolist(),
            "dates": equity_series.index.tolist(),
            "strategy_values": strategy_values,
            "final_summary": self.portfolio.get_summary(final_prices),
        }

        return results


__all__ = [
    "Portfolio",
    "Position",
    "StrategyAllocation",
    "PortfolioBacktestEngine",
]
