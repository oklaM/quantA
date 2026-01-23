"""
测试完整回测引擎的集成
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.engine.backtest import BacktestEngine, run_backtest
from backtest.engine.strategy import BuyAndHoldStrategy


@pytest.mark.integration
def test_backtest_engine_initialization(sample_price_data):
    """测试回测引擎初始化"""
    # 转换数据格式
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
    )

    assert engine is not None
    assert engine.data_handler is not None
    assert engine.portfolio is not None
    assert engine.execution_handler is not None


@pytest.mark.integration
def test_backtest_engine_run(sample_price_data):
    """测试运行回测"""
    # 准备数据
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0,  # 关闭滑点以简化测试
    )

    # 运行回测
    results = engine.run()

    # 检查结果
    assert results is not None
    assert "account" in results
    assert "equity_curve" in results
    assert "positions" in results
    assert "stats" in results
    assert "performance" in results


@pytest.mark.integration
def test_backtest_buy_and_hold(sample_price_data):
    """测试买入持有策略"""
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0,
    )

    results = engine.run()

    # 检查账户信息
    account = results["account"]
    assert account["initial_cash"] == 1000000.0
    assert account["cash"] < 1000000.0  # 应该花了钱买股票
    assert account["num_positions"] == 1

    # 检查持仓
    positions = results["positions"]
    assert len(positions) == 1
    assert positions[0]["symbol"] == "000001.SZ"
    assert positions[0]["quantity"] == 100

    # 检查统计信息
    stats = results["stats"]
    assert stats["total_bars"] > 0
    assert stats["total_fills"] > 0


@pytest.mark.integration
def test_backtest_empty_data():
    """测试空数据"""
    data = {}
    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    results = engine.run()

    # 空数据应该返回空结果
    assert results == {}


@pytest.mark.integration
def test_backtest_multiple_symbols(sample_multi_symbol_data):
    """测试多股票回测"""
    data = sample_multi_symbol_data.copy()

    # 只交易第一个股票
    first_symbol = list(data.keys())[0]
    strategy = BuyAndHoldStrategy(symbol=first_symbol, quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0,
    )

    results = engine.run()

    assert results is not None
    assert "account" in results
    assert "positions" in results


@pytest.mark.integration
def test_backtest_equity_curve(sample_price_data):
    """测试净值曲线"""
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0,
    )

    results = engine.run()

    equity_curve = results["equity_curve"]

    assert equity_curve is not None
    assert not equity_curve.empty
    assert "datetime" in equity_curve.columns
    assert "total_value" in equity_curve.columns
    assert "return" in equity_curve.columns
    assert "cash" in equity_curve.columns


@pytest.mark.integration
def test_backtest_performance_metrics(sample_price_data):
    """测试性能指标"""
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0,
    )

    results = engine.run()

    performance = results["performance"]

    assert "sharpe_ratio" in performance
    assert "max_drawdown" in performance
    assert "volatility" in performance


@pytest.mark.integration
def test_run_backtest_convenience_function(sample_price_data):
    """测试便捷函数"""
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0,
    )

    assert results is not None
    assert "account" in results
    assert "equity_curve" in results


@pytest.mark.integration
def test_backtest_reset(sample_price_data):
    """测试回测重置"""
    data = {"000001.SZ": sample_price_data.copy()}

    strategy = BuyAndHoldStrategy(symbol="000001.SZ", quantity=100)

    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    # 第一次运行
    results1 = engine.run()
    assert results1 is not None

    # 第二次运行
    results2 = engine.run()
    assert results2 is not None

    # 两次运行结果应该相似（但可能因为策略内部状态不同）
    assert "account" in results2
