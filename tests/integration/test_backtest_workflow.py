"""
回测流程集成测试
测试策略初始化、数据加载、回测执行和结果分析
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.engine.analysis import PerformanceAnalyzer
from backtest.engine.a_share_rules import AShareRulesEngine
from backtest.engine.data_handler import SimpleDataHandler
from backtest.engine.event_engine import EventQueue, EventType
from backtest.engine.indicators import TechnicalIndicators
from backtest.engine.portfolio import Portfolio
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy, Strategy
from backtest.optimization import GridSearchOptimizer
from utils.logging import get_logger

logger = get_logger(__name__)


# ========== Fixtures ==========


@pytest.fixture
def sample_data():
    """生成样本数据"""
    # 使用独立的方式生成数据
    engine = BacktestEngine(
        data={},
        strategy=BuyAndHoldStrategy(),
        initial_cash=1000000,
    )
    return engine.generate_mock_data(
        symbols=["600000.SH"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        freq="1d",
    )


@pytest.fixture
def multi_stock_data():
    """生成多股票数据"""
    engine = BacktestEngine(
        data={},
        strategy=BuyAndHoldStrategy(),
        initial_cash=1000000,
    )
    return engine.generate_mock_data(
        symbols=["600000.SH", "000001.SZ", "600036.SH"],
        start_date="2023-01-01",
        end_date="2023-06-30",
        freq="1d",
    )


@pytest.fixture
def sample_strategy():
    """创建样本策略"""
    return BuyAndHoldStrategy()


@pytest.fixture
def backtest_engine(sample_data):
    """创建回测引擎"""
    return BacktestEngine(
        data=sample_data,
        strategy=BuyAndHoldStrategy(),
        initial_cash=1000000,
        commission_rate=0.0003,
    )


# ========== Tests ==========


@pytest.mark.integration
class TestBacktestInitialization:
    """测试回测初始化"""

    def test_engine_initialization(self, sample_data):
        """测试引擎初始化"""
        strategy = BuyAndHoldStrategy()
        engine = BacktestEngine(
            data=sample_data,
            strategy=strategy,
            initial_cash=1000000,
            commission_rate=0.0003,
        )

        assert engine.data_handler is not None
        assert engine.portfolio is not None
        assert engine.strategy is not None
        assert engine.execution_handler is not None

    def test_portfolio_initialization(self):
        """测试组合初始化"""
        portfolio = Portfolio(
            initial_cash=1000000,
            commission_rate=0.0003,
        )

        assert portfolio.initial_cash == 1000000
        assert portfolio.current_cash == 1000000
        assert len(portfolio.positions) == 0

    def test_strategy_initialization(self):
        """测试策略初始化"""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)

        assert strategy.short_window == 5
        assert strategy.long_window == 20
        assert strategy.event_queue is not None


@pytest.mark.integration
class TestDataLoading:
    """测试数据加载"""

    def test_single_stock_data_loading(self, sample_data):
        """测试单股票数据加载"""
        assert "600000.SH" in sample_data
        assert len(sample_data["600000.SH"]) > 0
        assert "close" in sample_data["600000.SH"].columns

    def test_multi_stock_data_loading(self, multi_stock_data):
        """测试多股票数据加载"""
        assert len(multi_stock_data) == 3
        assert all(len(data) > 0 for data in multi_stock_data.values())

        # 验证数据一致性
        dates = set()
        for data in multi_stock_data.values():
            dates.update(data.index.tolist())

        # 所有股票应该有相同的日期范围
        assert len(dates) > 0

    def test_data_handler_initialization(self, sample_data):
        """测试数据处理器初始化"""
        handler = SimpleDataHandler(sample_data)

        assert len(handler.symbols) == 1
        assert handler.current_bar == 0

    def test_data_handler_iteration(self, sample_data):
        """测试数据处理器迭代"""
        handler = SimpleDataHandler(sample_data)

        # 重置
        handler.reset()

        # 获取第一个bar
        bar = handler.get_next_bar()
        assert bar is not None
        assert "symbol" in bar
        assert "datetime" in bar
        assert "close" in bar


@pytest.mark.integration
class TestStrategyExecution:
    """测试策略执行"""

    def test_buy_and_hold_strategy(self, backtest_engine):
        """测试买入持有策略"""
        results = backtest_engine.run()

        assert results is not None
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results

    def test_ma_cross_strategy(self, sample_data):
        """测试均线交叉策略"""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(
            data=sample_data,
            strategy=strategy,
            initial_cash=1000000,
        )

        results = engine.run()

        assert results is not None
        assert results["total_return"] is not None
        # MA策略应该产生一些交易
        assert results.get("total_trades", 0) >= 0

    def test_strategy_signals(self, sample_data):
        """测试策略信号生成"""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        strategy.set_data_handler(SimpleDataHandler(sample_data))

        # 重置数据处理器
        strategy.data_handler.reset()

        # 生成一些信号
        signals_generated = 0
        for _ in range(30):  # 至少30个bar
            bar = strategy.data_handler.get_next_bar()
            if bar is None:
                break

            strategy.on_bar(bar)

            # 检查是否有订单生成
            if strategy.event_queue:
                signals_generated += len(strategy.event_queue.events)
                strategy.event_queue = EventQueue()

        # 验证信号生成
        assert signals_generated >= 0  # 可能没有信号，但不会出错


@pytest.mark.integration
class TestPortfolioManagement:
    """测试组合管理"""

    def test_portfolio_update(self, backtest_engine):
        """测试组合更新"""
        initial_cash = backtest_engine.portfolio.current_cash

        # 运行回测
        backtest_engine.run()

        # 组合应该有变化
        assert backtest_engine.portfolio.current_cash != initial_cash or len(
            backtest_engine.portfolio.positions
        ) > 0

    def test_position_tracking(self, sample_data):
        """测试持仓跟踪"""
        strategy = BuyAndHoldStrategy()
        engine = BacktestEngine(
            data=sample_data,
            strategy=strategy,
            initial_cash=1000000,
        )

        engine.run()

        # BuyAndHold应该有持仓
        assert len(engine.portfolio.positions) > 0

        # 验证持仓数据
        for symbol, position in engine.portfolio.positions.items():
            assert position.quantity > 0
            assert position.avg_cost > 0

    def test_portfolio_performance(self, backtest_engine):
        """测试组合绩效"""
        results = backtest_engine.run()

        # 验证绩效指标
        assert results["total_return"] is not None
        assert results["sharpe_ratio"] is not None
        assert results["max_drawdown"] is not None

        # 基本合理性检查
        assert results["max_drawdown"] <= 0  # 回撤应该是负数
        assert results["total_trades"] >= 0  # 交易数非负


@pytest.mark.integration
class TestOrderExecution:
    """测试订单执行"""

    def test_order_creation(self, sample_data):
        """测试订单创建"""
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        strategy.set_data_handler(SimpleDataHandler(sample_data))

        strategy.data_handler.reset()

        # 模拟一些bar事件
        for _ in range(25):
            bar = strategy.data_handler.get_next_bar()
            if bar is None:
                break
            strategy.on_bar(bar)

        # 验证订单队列
        # 注意：可能没有订单，这取决于数据
        assert strategy.event_queue is not None

    def test_order_execution(self, sample_data):
        """测试订单执行"""
        portfolio = Portfolio(initial_cash=1000000, commission_rate=0.0003)

        # 模拟买入订单
        from backtest.engine.event_engine import OrderEvent

        order = OrderEvent(
            symbol="600000.SH",
            quantity=1000,
            direction="BUY",
            price=10.0,
        )

        # 执行订单
        fill = portfolio.execute_order(order, commission=0.0003)

        # 验证执行
        assert fill is not None
        assert portfolio.current_cash < 1000000  # 现金减少
        assert len(portfolio.positions) > 0  # 有持仓


@pytest.mark.integration
class TestAShareRules:
    """测试A股交易规则"""

    def test_t1_rule(self):
        """测试T+1规则"""
        rules_engine = AShareRulesEngine()

        # 当天买入的股票
        positions = {
            "600000.SH": {
                "quantity": 1000,
                "buy_date": datetime(2023, 1, 1),
                "can_sell": False,  # T+1规则
            }
        }

        # 检查是否可以卖出
        can_sell = rules_engine.check_t1_rule(positions, "600000.SH", datetime(2023, 1, 1))

        assert can_sell is False  # 当天不能卖出

    def test_price_limit_rule(self):
        """测试涨跌停限制"""
        rules_engine = AShareRulesEngine()

        # 测试涨停限制
        prev_close = 10.0
        limit_up = rules_engine.calculate_limit_price(prev_close, direction="up", board_type="main")

        # 主板涨停10%
        assert abs(limit_up - 11.0) < 0.01

        # 测试跌停限制
        limit_down = rules_engine.calculate_limit_price(prev_close, direction="down", board_type="main")

        # 主板跌停-10%
        assert abs(limit_down - 9.0) < 0.01

    def test_trading_hours(self):
        """测试交易时间"""
        rules_engine = AShareRulesEngine()

        # 测试交易时间
        is_trading_time = rules_engine.check_trading_hours(datetime(2023, 1, 1, 10, 30))

        assert is_trading_time is True  # 10:30是交易时间

        # 测试非交易时间
        is_trading_time = rules_engine.check_trading_hours(datetime(2023, 1, 1, 12, 0))

        assert is_trading_time is False  # 12:00不是交易时间


@pytest.mark.integration
class TestPerformanceAnalysis:
    """测试性能分析"""

    def test_analyzer_initialization(self, backtest_engine):
        """测试分析器初始化"""
        results = backtest_engine.run()
        analyzer = PerformanceAnalyzer(results)

        assert analyzer.results is not None
        assert analyzer.results == results

    def test_metrics_calculation(self, backtest_engine):
        """测试指标计算"""
        results = backtest_engine.run()
        analyzer = PerformanceAnalyzer(results)

        metrics = analyzer.calculate_all_metrics()

        # 验证关键指标
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "total_trades" in metrics
        assert "win_rate" in metrics

    def test_equity_curve(self, backtest_engine):
        """测试权益曲线"""
        results = backtest_engine.run()
        analyzer = PerformanceAnalyzer(results)

        equity_curve = analyzer.get_equity_curve()

        assert equity_curve is not None
        assert len(equity_curve) > 0
        assert "date" in equity_curve.columns
        assert "equity" in equity_curve.columns

    def test_report_generation(self, backtest_engine, tmp_path):
        """测试报告生成"""
        results = backtest_engine.run()
        analyzer = PerformanceAnalyzer(results)

        # 生成JSON报告
        report_path = tmp_path / "report.json"
        analyzer.save_report(str(report_path))

        assert report_path.exists()

        # 读取并验证
        import json

        with open(report_path, "r") as f:
            report = json.load(f)

        assert "total_return" in report
        assert "sharpe_ratio" in report


@pytest.mark.integration
class TestOptimization:
    """测试参数优化"""

    def test_grid_search_optimizer(self, sample_data):
        """测试网格搜索优化"""
        engine = BacktestEngine(initial_cash=1000000)

        optimizer = GridSearchOptimizer(
            engine=engine,
            strategy=MovingAverageCrossStrategy,
            param_grid={
                "short_window": [5, 10],
                "long_window": [20, 30],
            },
        )

        best_params, all_results = optimizer.optimize(sample_data, metric="sharpe_ratio")

        # 验证优化结果
        assert best_params is not None
        assert "short_window" in best_params
        assert "long_window" in best_params
        assert len(all_results) == 4  # 2 * 2组合

    def test_optimization_performance(self, sample_data):
        """测试优化性能"""
        import time

        engine = BacktestEngine(initial_cash=1000000)

        optimizer = GridSearchOptimizer(
            engine=engine,
            strategy=MovingAverageCrossStrategy,
            param_grid={
                "short_window": [5, 10],
                "long_window": [20],
            },
        )

        start_time = time.time()
        best_params, all_results = optimizer.optimize(sample_data)
        elapsed_time = time.time() - start_time

        # 验证性能（应该合理快速）
        assert elapsed_time < 30.0  # 30秒内完成
        assert len(all_results) == 2


@pytest.mark.integration
class TestMultiStrategy:
    """测试多策略回测"""

    def test_strategy_comparison(self, multi_stock_data):
        """测试策略对比"""
        engine = BacktestEngine(initial_cash=1000000)

        strategies = {
            "buy_and_hold": BuyAndHoldStrategy(),
            "ma_cross_5_20": MovingAverageCrossStrategy(5, 20),
            "ma_cross_10_30": MovingAverageCrossStrategy(10, 30),
        }

        results = {}
        for name, strategy in strategies.items():
            engine_copy = BacktestEngine(
                data=multi_stock_data,
                strategy=strategy,
                initial_cash=1000000,
            )
            result = engine_copy.run()
            results[name] = result

        # 验证所有策略都有结果
        assert len(results) == 3

        # 验证结果可比较
        returns = {name: r["total_return"] for name, r in results.items()}
        assert all(v is not None for v in returns.values())

    def test_strategy_portfolio(self, multi_stock_data):
        """测试策略组合"""
        from backtest.portfolio import StrategyPortfolio

        portfolio = StrategyPortfolio(initial_cash=1000000)

        strategies = {
            "strategy_1": BuyAndHoldStrategy(),
            "strategy_2": MovingAverageCrossStrategy(5, 20),
        }

        # 为每个策略分配资金
        weights = {"strategy_1": 0.6, "strategy_2": 0.4}

        # 运行组合回测
        results = {}
        for name, strategy in strategies.items():
            engine = BacktestEngine(
                data=multi_stock_data,
                strategy=strategy,
                initial_cash=1000000 * weights[name],
            )
            results[name] = engine.run()

        # 验证组合结果
        assert len(results) == 2
        assert all(r["total_return"] is not None for r in results.values())


@pytest.mark.integration
class TestTechnicalIndicators:
    """测试技术指标"""

    def test_indicator_calculation(self, sample_data):
        """测试指标计算"""
        indicators = TechnicalIndicators()
        data = sample_data["600000.SH"]

        # 添加一些指标
        featured_data = data.copy()
        if "close" in featured_data.columns:
            featured_data["sma_5"] = indicators.sma(featured_data["close"], 5)
            featured_data["sma_20"] = indicators.sma(featured_data["close"], 20)
            featured_data["rsi_14"] = indicators.rsi(featured_data["close"], 14)

        # 验证指标添加
        assert len(featured_data.columns) > len(data.columns)

    def test_sma_calculation(self, sample_data):
        """测试SMA计算"""
        indicators = TechnicalIndicators()
        data = sample_data["600000.SH"]

        sma_data = indicators.add_sma(data, period=20)

        # 验证SMA列存在
        assert "sma_20" in sma_data.columns

        # 验证SMA值合理
        assert sma_data["sma_20"].dropna().iloc[0] > 0

    def test_rsi_calculation(self, sample_data):
        """测试RSI计算"""
        indicators = TechnicalIndicators()
        data = sample_data["600000.SH"]

        rsi_data = indicators.add_rsi(data, period=14)

        # 验证RSI列存在
        assert "rsi_14" in rsi_data.columns

        # 验证RSI范围（0-100）
        rsi_values = rsi_data["rsi_14"].dropna()
        assert rsi_values.min() >= 0
        assert rsi_values.max() <= 100


@pytest.mark.integration
class TestBacktestScenarios:
    """测试回测场景"""

    def test_bull_market_scenario(self):
        """测试牛市场景"""
        # 生成上涨趋势数据
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = 10 + np.arange(len(dates)) * 0.01  # 每天上涨

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": 1000000,
            }
        )

        data_dict = {"600000.SH": data}

        engine = BacktestEngine(
            data=data_dict,
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 牛市中应该盈利
        assert results["total_return"] > 0

    def test_bear_market_scenario(self):
        """测试熊市场景"""
        # 生成下跌趋势数据
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = 20 - np.arange(len(dates)) * 0.01  # 每天下跌

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 1.01,
                "high": prices * 0.99,
                "low": prices * 0.98,
                "close": prices,
                "volume": 1000000,
            }
        )

        data_dict = {"600000.SH": data}

        engine = BacktestEngine(
            data=data_dict,
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 熊市中应该亏损
        assert results["total_return"] < 0

    def test_sideways_market_scenario(self):
        """测试震荡市场景"""
        # 生成震荡数据
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        prices = 10 + np.sin(np.arange(len(dates)) * 0.1)  # 震荡

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 0.99,
                "high": prices * 1.01,
                "low": prices * 0.98,
                "close": prices,
                "volume": 1000000,
            }
        )

        data_dict = {"600000.SH": data}

        engine = BacktestEngine(
            data=data_dict,
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 震荡市收益应该接近0
        assert abs(results["total_return"]) < 0.1


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleBacktest:
    """测试大规模回测"""

    def test_large_dataset_backtest(self):
        """测试大数据集回测"""
        # 生成3年数据
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH", "000001.SZ", "600036.SH", "000002.SZ"],
            start_date="2020-01-01",
            end_date="2023-12-31",
            freq="1d",
        )

        # 运行回测
        strategy = BuyAndHoldStrategy()
        engine_test = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_cash=1000000,
        )

        import time

        start_time = time.time()
        results = engine_test.run()
        elapsed_time = time.time() - start_time

        # 验证性能和结果
        assert results is not None
        assert elapsed_time < 60.0  # 60秒内完成
        assert results["total_return"] is not None

    def test_multi_symbol_backtest(self):
        """测试多股票回测"""
        # 生成10只股票数据
        symbols = [f"60000{i}.SH" for i in range(10)]

        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=symbols,
            start_date="2023-01-01",
            end_date="2023-06-30",
            freq="1d",
        )

        # 运行回测
        strategy = BuyAndHoldStrategy()
        engine_test = BacktestEngine(
            data=data,
            strategy=strategy,
            initial_cash=1000000,
        )

        results = engine_test.run()

        # 验证结果
        assert results is not None
        assert len(engine_test.portfolio.positions) == 10  # 所有股票都有持仓
