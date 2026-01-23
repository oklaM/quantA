"""
组合回测模块测试
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.portfolio import (
    Portfolio,
    PortfolioBacktestEngine,
    Position,
    StrategyAllocation,
)


@pytest.fixture
def sample_symbols():
    """示例股票代码"""
    return ["000001.SZ", "000002.SZ", "600000.SH"]


@pytest.fixture
def sample_data_dict(sample_symbols):
    """生成示例数据字典"""
    np.random.seed(42)
    data_dict = {}

    for symbol in sample_symbols:
        n = 100
        returns = np.random.normal(0.0005, 0.02, n)
        prices = 100.0 * (1 + returns).cumprod()

        dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq="D")

        data = []
        for date, close in zip(dates, prices):
            high = close * (1 + abs(np.random.normal(0, 0.015)))
            low = close * (1 - abs(np.random.normal(0, 0.015)))
            open_price = close * (1 + np.random.normal(0, 0.008))

            data.append(
                {
                    "datetime": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(high, open_price, close),
                    "low": min(low, open_price, close),
                    "close": close,
                    "volume": np.random.randint(1000000, 10000000),
                }
            )

        data_dict[symbol] = pd.DataFrame(data)

    return data_dict


@pytest.fixture
def sample_prices(sample_symbols):
    """示例价格字典"""
    return {symbol: 100.0 + i * 10.0 for i, symbol in enumerate(sample_symbols)}


@pytest.mark.backtest
class TestPosition:
    """测试Position类"""

    def test_position_creation(self):
        """测试创建持仓"""
        position = Position(
            symbol="000001.SZ",
            quantity=1000,
            entry_price=10.0,
            current_price=11.0,
            market_value=11000.0,
            unrealized_pnl=1000.0,
            weight=0.5,
        )

        assert position.symbol == "000001.SZ"
        assert position.quantity == 1000
        assert position.entry_price == 10.0
        assert position.unrealized_pnl == 1000.0

    def test_position_to_dict(self):
        """测试转换为字典"""
        position = Position(
            symbol="000001.SZ",
            quantity=1000,
            entry_price=10.0,
            current_price=11.0,
            market_value=11000.0,
            unrealized_pnl=1000.0,
            weight=0.5,
        )

        pos_dict = position.to_dict()

        assert isinstance(pos_dict, dict)
        assert pos_dict["symbol"] == "000001.SZ"
        assert pos_dict["quantity"] == 1000


@pytest.mark.backtest
class TestStrategyAllocation:
    """测试StrategyAllocation类"""

    def test_strategy_allocation_creation(self, sample_symbols):
        """测试创建策略配置"""
        strategy = BuyAndHoldStrategy(symbol=sample_symbols[0])
        allocation = StrategyAllocation(
            strategy=strategy,
            symbols=sample_symbols,
            weight=0.5,
            max_position=0.3,
        )

        assert allocation.strategy == strategy
        assert allocation.symbols == sample_symbols
        assert allocation.weight == 0.5
        assert allocation.max_position == 0.3

    def test_strategy_allocation_to_dict(self, sample_symbols):
        """测试转换为字典"""
        strategy = BuyAndHoldStrategy(symbol=sample_symbols[0])
        allocation = StrategyAllocation(
            strategy=strategy,
            symbols=sample_symbols,
            weight=0.5,
        )

        alloc_dict = allocation.to_dict()

        assert isinstance(alloc_dict, dict)
        assert alloc_dict["strategy_name"] == "BuyAndHoldStrategy"
        assert alloc_dict["symbols"] == sample_symbols


@pytest.mark.backtest
class TestPortfolio:
    """测试Portfolio类"""

    def test_portfolio_initialization(self, sample_symbols):
        """测试初始化"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        assert portfolio.initial_cash == 1000000.0
        assert len(portfolio.strategies) == len(sample_symbols)
        assert len(portfolio.positions) == 0
        assert len(portfolio.equity_curve) == 0

    def test_portfolio_weight_validation(self, sample_symbols):
        """测试权重验证"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=0.6,  # 总和将为1.2，不为1.0
            )
            for symbol in sample_symbols[:2]
        ]

        with pytest.raises(ValueError):
            portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

    def test_get_total_value(self, sample_symbols, sample_prices):
        """测试获取总价值"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        total_value = portfolio.get_total_value(sample_prices)

        # 初始应该全是现金
        assert total_value == 1000000.0

    def test_update_position_buy(self, sample_symbols, sample_prices):
        """测试买入更新持仓"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        # 买入
        portfolio.update_position(
            strategy_id=0,
            symbol=sample_symbols[0],
            quantity=1000,
            price=sample_prices[sample_symbols[0]],
        )

        # 检查持仓
        assert sample_symbols[0] in portfolio.positions
        assert portfolio.positions[sample_symbols[0]].quantity == 1000

        # 检查现金减少
        assert portfolio.strategy_cash[0] < portfolio.cash_allocations[0]

    def test_update_position_sell(self, sample_symbols, sample_prices):
        """测试卖出更新持仓"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        # 先买入
        portfolio.update_position(
            strategy_id=0,
            symbol=sample_symbols[0],
            quantity=1000,
            price=sample_prices[sample_symbols[0]],
        )

        # 卖出
        portfolio.update_position(
            strategy_id=0,
            symbol=sample_symbols[0],
            quantity=-500,
            price=sample_prices[sample_symbols[0]],
        )

        # 检查持仓减少
        assert portfolio.positions[sample_symbols[0]].quantity == 500

    def test_update_position_close(self, sample_symbols, sample_prices):
        """测试平仓"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        # 买入
        portfolio.update_position(
            strategy_id=0,
            symbol=sample_symbols[0],
            quantity=1000,
            price=sample_prices[sample_symbols[0]],
        )

        # 全部卖出
        portfolio.update_position(
            strategy_id=0,
            symbol=sample_symbols[0],
            quantity=-1000,
            price=sample_prices[sample_symbols[0]],
        )

        # 检查持仓被删除
        assert sample_symbols[0] not in portfolio.positions

    def test_record_equity(self, sample_symbols, sample_prices):
        """测试记录权益"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        # 记录权益
        date = pd.Timestamp(datetime.now())
        portfolio.record_equity(date, sample_prices)

        assert len(portfolio.equity_curve) == 1
        assert len(portfolio.dates) == 1
        assert portfolio.equity_curve[0] == portfolio.get_total_value(sample_prices)

    def test_get_summary(self, sample_symbols, sample_prices):
        """测试获取摘要"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        portfolio = Portfolio(initial_cash=1000000.0, strategies=strategies)

        # 买入一个股票
        portfolio.update_position(
            strategy_id=0,
            symbol=sample_symbols[0],
            quantity=1000,
            price=sample_prices[sample_symbols[0]],
        )

        summary = portfolio.get_summary(sample_prices)

        assert "total_value" in summary
        assert "total_return" in summary
        assert "num_positions" in summary
        assert summary["num_positions"] == 1


@pytest.mark.backtest
class TestPortfolioBacktestEngine:
    """测试PortfolioBacktestEngine类"""

    def test_engine_initialization(self, sample_data_dict, sample_symbols):
        """测试初始化"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        engine = PortfolioBacktestEngine(
            data_dict=sample_data_dict,
            strategies=strategies,
            initial_cash=1000000.0,
        )

        assert engine.data_dict == sample_data_dict
        assert len(engine.strategies) == len(sample_symbols)

    def test_engine_run(self, sample_data_dict, sample_symbols):
        """测试运行回测"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        engine = PortfolioBacktestEngine(
            data_dict=sample_data_dict,
            strategies=strategies,
            initial_cash=1000000.0,
        )

        results = engine.run()

        # 检查结果
        assert "total_return" in results
        assert "annual_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "equity_curve" in results
        assert "dates" in results
        assert len(results["equity_curve"]) > 0

    def test_strategy_values(self, sample_data_dict, sample_symbols):
        """测试各策略价值"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        engine = PortfolioBacktestEngine(
            data_dict=sample_data_dict,
            strategies=strategies,
            initial_cash=1000000.0,
        )

        results = engine.run()

        # 检查各策略都有价值
        assert "strategy_values" in results
        assert len(results["strategy_values"]) == len(sample_symbols)

        for strategy_id, value in results["strategy_values"].items():
            assert value >= 0

    def test_portfolio_summary(self, sample_data_dict, sample_symbols):
        """测试组合摘要"""
        strategies = [
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=1.0 / len(sample_symbols),
            )
            for symbol in sample_symbols
        ]

        engine = PortfolioBacktestEngine(
            data_dict=sample_data_dict,
            strategies=strategies,
            initial_cash=1000000.0,
        )

        results = engine.run()

        # 检查摘要
        assert "final_summary" in results
        summary = results["final_summary"]

        assert "total_value" in summary
        assert "cash" in summary
        assert "num_positions" in summary
        assert "positions" in summary
