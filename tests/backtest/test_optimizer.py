"""
参数优化模块测试
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.engine.strategies import BollingerBandsStrategy
from backtest.optimization import (
    BayesianOptimizer,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    create_optimizer,
)


@pytest.fixture
def sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n = 200
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n)
    prices = base_price * (1 + returns).cumprod()

    dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq="D")

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.015)))
        low = close * (1 - abs(np.random.normal(0, 0.015)))
        open_price = close * (1 + np.random.normal(0, 0.008))

        data.append(
            {
                "datetime": date,
                "symbol": "000001.SZ",
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": np.random.randint(1000000, 10000000),
            }
        )

    return pd.DataFrame(data)


@pytest.mark.backtest
class TestGridSearchOptimizer:
    """测试网格搜索优化器"""

    def test_grid_search_initialization(self, sample_data):
        """测试初始化"""
        optimizer = GridSearchOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        assert optimizer.data is not None
        assert optimizer.strategy_class == BollingerBandsStrategy
        assert len(optimizer.results) == 0

    def test_grid_search_small_space(self, sample_data):
        """测试小参数空间的网格搜索"""
        optimizer = GridSearchOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        param_space = {
            "period": [10, 20],
            "std_dev": [1.5, 2.0],
        }

        result = optimizer.optimize(
            param_space=param_space,
            optimization_target="sharpe_ratio",
        )

        # 应该有4个结果 (2 * 2)
        assert len(optimizer.results) <= 4

        # 检查最佳结果
        assert result.params is not None
        assert "period" in result.params
        assert "std_dev" in result.params
        assert isinstance(result.metrics["sharpe_ratio"], float)

    def test_grid_search_with_n_trials(self, sample_data):
        """测试限制试验次数的网格搜索"""
        optimizer = GridSearchOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        param_space = {
            "period": [10, 20, 30, 40],
            "std_dev": [1.5, 2.0, 2.5],
        }

        # 限制只运行3次
        result = optimizer.optimize(
            param_space=param_space,
            optimization_target="sharpe_ratio",
            n_trials=3,
        )

        # 应该最多有3个结果
        assert len(optimizer.results) <= 3


@pytest.mark.backtest
class TestRandomSearchOptimizer:
    """测试随机搜索优化器"""

    def test_random_search_initialization(self, sample_data):
        """测试初始化"""
        optimizer = RandomSearchOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        assert optimizer.data is not None
        assert optimizer.strategy_class == BollingerBandsStrategy

    def test_random_search_with_ranges(self, sample_data):
        """测试使用范围的随机搜索"""
        optimizer = RandomSearchOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        param_space = {
            "period": ["int", 5, 30],  # 整数范围
            "std_dev": ["uniform", 1.0, 3.0],  # 均匀分布
        }

        result = optimizer.optimize(
            param_space=param_space,
            optimization_target="sharpe_ratio",
            n_trials=5,
        )

        # 应该有5个结果
        assert len(optimizer.results) == 5

        # 检查参数在合理范围内
        assert 5 <= result.params["period"] <= 30
        assert 1.0 <= result.params["std_dev"] <= 3.0

    def test_random_search_sampling(self, sample_data):
        """测试参数采样"""
        optimizer = RandomSearchOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        param_space = {
            "period": [10, 20, 30],  # 离散值列表
            "std_dev": ["uniform", 1.5, 2.5],
        }

        # 测试多次采样
        params_list = []
        for _ in range(10):
            params = optimizer._sample_params(param_space)
            params_list.append(params)

        # 检查所有参数都在范围内
        for params in params_list:
            assert params["period"] in [10, 20, 30]
            assert 1.5 <= params["std_dev"] <= 2.5


@pytest.mark.backtest
class TestBayesianOptimizer:
    """测试贝叶斯优化器"""

    def test_bayesian_optimizer_requires_optuna(self, sample_data):
        """测试贝叶斯优化器需要optuna"""
        try:
            import optuna

            optuna_available = True
        except ImportError:
            optuna_available = False

        if not optuna_available:
            with pytest.raises(ImportError):
                optimizer = BayesianOptimizer(
                    data=sample_data,
                    strategy_class=BollingerBandsStrategy,
                )

    def test_bayesian_optimizer_initialization(self, sample_data):
        """测试初始化"""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        optimizer = BayesianOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        assert optimizer.data is not None

    def test_bayesian_optimization_small(self, sample_data):
        """测试小规模贝叶斯优化"""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        optimizer = BayesianOptimizer(
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        param_space = {
            "period": ("int", 10, 30),
            "std_dev": ("float", 1.5, 2.5),
        }

        result = optimizer.optimize(
            param_space=param_space,
            optimization_target="sharpe_ratio",
            n_trials=5,
        )

        # 检查结果
        assert result.params is not None
        assert 10 <= result.params["period"] <= 30
        assert 1.5 <= result.params["std_dev"] <= 2.5


@pytest.mark.backtest
class TestOptimizerFactory:
    """测试优化器工厂函数"""

    def test_create_grid_optimizer(self, sample_data):
        """测试创建网格搜索优化器"""
        optimizer = create_optimizer(
            optimizer_type="grid",
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        assert isinstance(optimizer, GridSearchOptimizer)

    def test_create_random_optimizer(self, sample_data):
        """测试创建随机搜索优化器"""
        optimizer = create_optimizer(
            optimizer_type="random",
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        assert isinstance(optimizer, RandomSearchOptimizer)

    def test_create_bayesian_optimizer(self, sample_data):
        """测试创建贝叶斯优化器"""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")

        optimizer = create_optimizer(
            optimizer_type="bayesian",
            data=sample_data,
            strategy_class=BollingerBandsStrategy,
        )

        assert isinstance(optimizer, BayesianOptimizer)

    def test_create_invalid_optimizer(self, sample_data):
        """测试创建无效的优化器"""
        with pytest.raises(ValueError):
            create_optimizer(
                optimizer_type="invalid",
                data=sample_data,
                strategy_class=BollingerBandsStrategy,
            )


@pytest.mark.backtest
class TestOptimizationResult:
    """测试优化结果"""

    def test_optimization_result_creation(self):
        """测试优化结果创建"""
        from backtest.optimization import OptimizationResult

        result = OptimizationResult(
            params={"period": 20, "std_dev": 2.0},
            metrics={"sharpe_ratio": 1.5, "annual_return": 0.2},
            backtest_results={},
        )

        assert result.params == {"period": 20, "std_dev": 2.0}
        assert result.metrics["sharpe_ratio"] == 1.5
        assert isinstance(result.timestamp, str)

    def test_optimization_result_timestamp(self):
        """测试时间戳自动生成"""
        from backtest.optimization import OptimizationResult

        result1 = OptimizationResult(
            params={},
            metrics={},
            backtest_results={},
        )

        import time

        time.sleep(0.01)

        result2 = OptimizationResult(
            params={},
            metrics={},
            backtest_results={},
        )

        # result2的时间戳应该更晚
        assert result2.timestamp > result1.timestamp
