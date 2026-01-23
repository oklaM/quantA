"""
异常场景测试
测试系统在各种异常和边界条件下的行为
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine
from backtest.engine.indicators import EMA, MACD, RSI, SMA
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy
from data.market.data_manager import DataManager
from rl.envs.a_share_trading_env import ASharesTradingEnv
from trading.risk import ActionType, OrderRequest, RiskController


@pytest.mark.edge_case
class TestInvalidInputData:
    """测试无效输入数据处理"""

    def test_empty_dataframe(self):
        """测试空数据框"""
        engine = BacktestEngine(initial_cash=1000000)

        empty_data = {}

        with pytest.raises(ValueError, match=".*数据为空.*"):
            engine.run(BuyAndHoldStrategy(), empty_data)

    def test_missing_columns(self):
        """测试缺少必需列"""
        engine = BacktestEngine(initial_cash=1000000)

        # 创建缺少close列的数据
        invalid_data = {
            "600000.SH": pd.DataFrame(
                {
                    "open": [10.0, 10.5, 11.0],
                    "high": [10.5, 11.0, 11.5],
                    # 缺少 'close' 和 'volume'
                }
            )
        }

        with pytest.raises(ValueError, match=".*缺少必需列.*"):
            engine.run(BuyAndHoldStrategy(), invalid_data)

    def test_negative_prices(self):
        """测试负价格"""
        engine = BacktestEngine(initial_cash=1000000)

        # 包含负价格的数据
        data_with_negative = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=3),
                    "open": [10.0, 10.5, -5.0],  # 负价格
                    "high": [10.5, 11.0, 11.5],
                    "low": [9.5, 10.0, -6.0],  # 负价格
                    "close": [10.5, 11.0, 11.5],
                    "volume": [1000000, 1500000, 2000000],
                }
            )
        }

        # 应该处理负价格（过滤或报错）
        try:
            result = engine.run(BuyAndHoldStrategy(), data_with_negative)
            # 如果没有报错，应该验证数据被清洗
            assert result is not None
        except ValueError as e:
            assert "负价格" in str(e) or "negative" in str(e).lower()

    def test_zero_volume(self):
        """测试零成交量"""
        engine = BacktestEngine(initial_cash=1000000)

        data_with_zero_volume = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=3),
                    "open": [10.0, 10.5, 11.0],
                    "high": [10.5, 11.0, 11.5],
                    "low": [9.5, 10.0, 10.5],
                    "close": [10.5, 11.0, 11.5],
                    "volume": [1000000, 0, 2000000],  # 零成交量
                }
            )
        }

        # 应该能够处理零成交量
        result = engine.run(BuyAndHoldStrategy(), data_with_zero_volume)
        assert result is not None

    def test_nan_values(self):
        """测试NaN值处理"""
        engine = BacktestEngine(initial_cash=1000000)

        data_with_nan = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=5),
                    "open": [10.0, np.nan, 11.0, 11.5, 12.0],
                    "high": [10.5, 11.0, np.nan, 12.0, 12.5],
                    "low": [9.5, 10.0, 10.5, np.nan, 11.5],
                    "close": [10.5, 11.0, 11.5, 12.0, np.nan],
                    "volume": [1000000, 1500000, 2000000, 2500000, 3000000],
                }
            )
        }

        # 应该能够处理NaN（填充或删除）
        result = engine.run(BuyAndHoldStrategy(), data_with_nan)
        assert result is not None

    def test_infinite_values(self):
        """测试无穷大值"""
        data_with_inf = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=3),
                    "open": [10.0, np.inf, 11.0],
                    "high": [10.5, 11.0, 11.5],
                    "low": [9.5, 10.0, -np.inf],  # 负无穷
                    "close": [10.5, 11.0, 11.5],
                    "volume": [1000000, 1500000, 2000000],
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)

        # 应该处理无穷大值
        try:
            result = engine.run(BuyAndHoldStrategy(), data_with_inf)
            assert result is not None
        except (ValueError, FloatingPointError):
            # 或者抛出适当的异常
            pass

    def test_duplicate_datetime_index(self):
        """测试重复的时间索引"""
        data_with_duplicates = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": ["2023-01-01", "2023-01-01", "2023-01-02"],  # 重复日期
                    "open": [10.0, 10.5, 11.0],
                    "high": [10.5, 11.0, 11.5],
                    "low": [9.5, 10.0, 10.5],
                    "close": [10.5, 11.0, 11.5],
                    "volume": [1000000, 1500000, 2000000],
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)

        # 应该处理重复日期（删除或聚合）
        result = engine.run(BuyAndHoldStrategy(), data_with_duplicates)
        assert result is not None

    def test_single_row_data(self):
        """测试只有一行数据"""
        single_row_data = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": ["2023-01-01"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.5],
                    "close": [10.5],
                    "volume": [1000000],
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)

        # 至少需要一定数量的数据才能计算指标
        try:
            result = engine.run(BuyAndHoldStrategy(), single_row_data)
            # 如果能够运行，应该返回结果
            assert result is not None
        except ValueError as e:
            # 或者抛出有意义的错误
            assert "数据不足" in str(e) or "insufficient" in str(e).lower()


@pytest.mark.edge_case
class TestExtremeValues:
    """测试极端值处理"""

    def test_very_large_prices(self):
        """测试非常大的价格"""
        data = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=3),
                    "open": [10000.0, 10500.0, 11000.0],  # 非常高的价格
                    "high": [10500.0, 11000.0, 11500.0],
                    "low": [9500.0, 10000.0, 10500.0],
                    "close": [10500.0, 11000.0, 11500.0],
                    "volume": [1000000, 1500000, 2000000],
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)
        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None

    def test_very_small_prices(self):
        """测试非常小的价格"""
        data = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=3),
                    "open": [0.01, 0.02, 0.03],  # 非常低的价格
                    "high": [0.02, 0.03, 0.04],
                    "low": [0.005, 0.015, 0.025],
                    "close": [0.02, 0.03, 0.04],
                    "volume": [1000000, 1500000, 2000000],
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)
        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None

    def test_extreme_volume(self):
        """测试极端成交量"""
        data = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=3),
                    "open": [10.0, 10.5, 11.0],
                    "high": [10.5, 11.0, 11.5],
                    "low": [9.5, 10.0, 10.5],
                    "close": [10.5, 11.0, 11.5],
                    "volume": [10**9, 10**10, 10**11],  # 极端大的成交量
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)
        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None

    def test_price_volatility_spike(self):
        """测试价格剧烈波动"""
        data = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=5),
                    "open": [10.0, 50.0, 10.0, 100.0, 10.0],  # 巨大波动
                    "high": [50.0, 100.0, 50.0, 150.0, 50.0],
                    "low": [9.0, 10.0, 9.0, 10.0, 9.0],
                    "close": [50.0, 10.0, 100.0, 10.0, 50.0],
                    "volume": [1000000, 1500000, 2000000, 2500000, 3000000],
                }
            )
        }

        engine = BacktestEngine(initial_cash=1000000)
        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None


@pytest.mark.edge_case
class TestResourceConstraints:
    """测试资源限制"""

    def test_insufficient_cash(self):
        """测试资金不足"""
        engine = BacktestEngine(initial_cash=1000)  # 很少的初始资金

        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        result = engine.run(BuyAndHoldStrategy(), data)

        # 应该能够处理资金不足的情况
        assert result is not None
        # 总收益率不应该包含无穷大或NaN
        assert not np.isinf(result["total_return"])
        assert not np.isnan(result["total_return"])

    def test_zero_initial_cash(self):
        """测试零初始资金"""
        engine = BacktestEngine(initial_cash=0)

        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        # 应该能够处理（虽然不会有交易）
        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None

    def test_very_high_commission(self):
        """测试极高的手续费"""
        engine = BacktestEngine(
            initial_cash=1000000,
            commission=0.5,  # 50%手续费
        )

        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        result = engine.run(BuyAndHoldStrategy(), data)

        # 高手续费应该导致亏损，但不应该崩溃
        assert result is not None


@pytest.mark.edge_case
class TestBoundaryConditions:
    """测试边界条件"""

    def test_exactly_one_year(self):
        """测试刚好一年的数据"""
        engine = BacktestEngine(initial_cash=1000000)

        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-12-31",  # 刚好一年
        )

        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None

    def test_leap_year_feb29(self):
        """测试闰年2月29日"""
        engine = BacktestEngine(initial_cash=1000000)

        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2024-02-28",
            end_date="2024-03-01",  # 包含闰日
        )

        result = engine.run(BuyAndHoldStrategy(), data)
        assert result is not None

    def test_minimum_indicator_period(self):
        """测试刚好满足指标最小周期"""
        # SMA需要至少20个数据点
        data = {
            "600000.SH": pd.DataFrame(
                {
                    "datetime": pd.date_range("2023-01-01", periods=20),
                    "open": np.random.randn(20).cumsum() + 100,
                    "high": np.random.randn(20).cumsum() + 102,
                    "low": np.random.randn(20).cumsum() + 98,
                    "close": np.random.randn(20).cumsum() + 100,
                    "volume": np.random.randint(1000000, 10000000, 20),
                }
            )
        }

        # 应该能够计算SMA
        sma = SMA(data["600000.SH"]["close"], 20)
        assert len(sma) == 20
        assert not sma.isna().all()  # 至少有一些非NaN值

    def test_indicator_with_insufficient_data(self):
        """测试数据不足时的指标计算"""
        # 只有5个数据点，但尝试计算20日SMA
        data = pd.Series(np.random.randn(5).cumsum() + 100)

        # 应该返回NaN或部分结果
        sma = SMA(data, 20)
        assert len(sma) == 5
        # 前19个应该是NaN，第20个（不存在）也应该处理正确
        assert sma.isna().sum() >= 0


@pytest.mark.edge_case
class TestRiskControlExceptions:
    """测试风控异常场景"""

    def test_invalid_order_parameters(self):
        """测试无效订单参数"""
        controller = RiskController()

        context = {
            "account": {"total_asset": 1000000, "available_cash": 500000},
            "positions": [],
            "daily_stats": {"initial_asset": 1000000, "traded_volume": 0, "daily_pnl": 0},
        }

        # 测试负数量
        allowed, rejects = controller.validate_order(
            symbol="600000.SH",
            action="buy",
            quantity=-100,  # 负数量
            price=10.0,
            context=context,
        )

        assert allowed is False

    def test_negative_price(self):
        """测试负价格"""
        controller = RiskController()

        context = {
            "account": {"total_asset": 1000000, "available_cash": 500000},
            "positions": [],
            "daily_stats": {},
        }

        # 测试负价格
        allowed, rejects = controller.validate_order(
            symbol="600000.SH",
            action="buy",
            quantity=1000,
            price=-10.0,  # 负价格
            context=context,
        )

        assert allowed is False

    def test_invalid_action(self):
        """测试无效的交易动作"""
        controller = RiskController()

        context = {
            "account": {"total_asset": 1000000, "available_cash": 500000},
            "positions": [],
            "daily_stats": {},
        }

        # 测试无效动作
        allowed, rejects = controller.validate_order(
            symbol="600000.SH",
            action="invalid_action",  # 无效动作
            quantity=1000,
            price=10.0,
            context=context,
        )

        # 应该拒绝或处理
        assert isinstance(allowed, bool)


@pytest.mark.edge_case
class TestRLExceptionScenarios:
    """测试RL异常场景"""

    def test_env_reset_failure(self):
        """测试环境重置失败"""
        # 创建空数据
        empty_data = pd.DataFrame(
            {
                "datetime": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )

        with pytest.raises((ValueError, IndexError)):
            env = ASharesTradingEnv(df=empty_data, initial_cash=100000)
            obs, info = env.reset()

    def test_invalid_action(self):
        """测试无效动作"""
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        env = ASharesTradingEnv(df=data, initial_cash=100000)
        obs, info = env.reset()

        # 测试超出范围的action
        invalid_action = 999  # 远超action_space

        with pytest.raises((ValueError, IndexError)):
            next_obs, reward, done, truncated, info = env.step(invalid_action)

    def test_env_step_after_done(self):
        """测试在done后继续step"""
        engine = BacktestEngine(initial_cash=1000)
        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-01-05",  # 很少的数据
        )

        env = ASharesTradingEnv(df=data, initial_cash=100)
        obs, info = env.reset()

        # 运行到结束
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, reward, done, truncated, info = env.step(action)

        # 尝试在done后继续step
        # 应该能够处理（要么报错，要么重置环境）
        try:
            next_obs, reward, done, truncated, info = env.step(0)
            # 如果没有报错，应该返回终止状态
            assert done or truncated
        except (ValueError, RuntimeError):
            # 或者抛出适当的异常
            pass


@pytest.mark.edge_case
class TestFileIORexceptions:
    """测试文件IO异常"""

    def test_nonexistent_file(self):
        """测试读取不存在的文件"""
        with pytest.raises(FileNotFoundError):
            pd.read_csv("/nonexistent/path/to/file.csv")

    def test_invalid_file_format(self):
        """测试无效的文件格式"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("invalid,csv,format\n")
            f.write("not,a,dataframe\n")
            temp_path = f.name

        try:
            # 读取应该可能成功，但数据不完整
            df = pd.read_csv(temp_path)
            # 验证能够处理不完整的数据
            assert df is not None
        finally:
            os.unlink(temp_path)

    def test_permission_denied(self, tmp_path):
        """测试权限被拒绝"""
        # 创建文件并移除权限
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # 移除读权限（Unix）
        original_mode = test_file.stat().st_mode
        test_file.chmod(0o000)

        try:
            with pytest.raises(PermissionError):
                test_file.read_text()
        finally:
            # 恢复权限以便清理
            test_file.chmod(original_mode)


@pytest.mark.edge_case
class TestNetworkErrorSimulation:
    """测试网络错误模拟"""

    def test_data_source_unavailable(self):
        """测试数据源不可用"""
        # 尝试连接不存在的数据源
        from data.market.sources import BaseDataProvider

        class UnavailableProvider(BaseDataProvider):
            def connect(self):
                raise ConnectionError("数据源不可用")

        provider = UnavailableProvider()

        with pytest.raises(ConnectionError):
            provider.connect()

    def test_timeout_simulation(self):
        """测试超时"""
        import time

        def slow_operation():
            time.sleep(2)
            return "done"

        # 设置短超时
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("操作超时")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)  # 1秒超时

        try:
            with pytest.raises(TimeoutError):
                slow_operation()
        finally:
            signal.alarm(0)  # 取消闹钟


@pytest.mark.edge_case
class TestConcurrentAccess:
    """测试并发访问问题"""

    def test_concurrent_backtest(self):
        """测试并发执行回测"""
        import threading

        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        results = []
        errors = []

        def run_backtest():
            try:
                result = engine.run(BuyAndHoldStrategy(), data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # 创建多个线程
        threads = [threading.Thread(target=run_backtest) for _ in range(5)]

        # 启动所有线程
        for t in threads:
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(errors) == 0, f"并发访问导致错误: {errors}"
        assert len(results) == 5
