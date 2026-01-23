"""
回测参数优化模块
提供网格搜索、随机搜索、贝叶斯优化等参数优化方法
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import RandomSampler, TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    # 创建假的optuna模块用于类型注解
    class DummyTrial:
        pass

    class DummyOptuna:
        Trial = DummyTrial

    optuna = DummyOptuna()
    OPTUNA_AVAILABLE = False

try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import Strategy
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""

    params: Dict[str, Any]
    metrics: Dict[str, float]
    backtest_results: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ParameterOptimizer:
    """
    参数优化器基类
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_class: type,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
    ):
        """
        Args:
            data: 回测数据
            strategy_class: 策略类
            initial_cash: 初始资金
            commission_rate: 手续费率
        """
        self.data = data
        self.strategy_class = strategy_class
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate

        self.results: List[OptimizationResult] = []

    def _run_backtest(self, params: Dict[str, Any], symbol: str = "000001.SZ") -> Dict[str, float]:
        """
        运行单次回测

        Args:
            params: 策略参数
            symbol: 标的代码

        Returns:
            回测指标
        """
        # 创建策略实例
        strategy = self.strategy_class(symbol=symbol, **params)

        # 创建回测引擎
        engine = BacktestEngine(
            data=self.data,
            strategy=strategy,
            initial_cash=self.initial_cash,
            commission_rate=self.commission_rate,
        )

        # 运行回测
        results = engine.run()

        # 提取关键指标
        metrics = {
            "total_return": results["total_return"],
            "annual_return": results.get("annual_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "win_rate": results.get("win_rate", 0),
            "profit_factor": results.get("profit_factor", 0),
        }

        return metrics

    def optimize(
        self,
        param_space: Dict[str, Any],
        optimization_target: str = "sharpe_ratio",
        n_trials: Optional[int] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        执行优化

        Args:
            param_space: 参数空间
            optimization_target: 优化目标指标
            n_trials: 试验次数
            **kwargs: 其他参数

        Returns:
            最佳优化结果
        """
        raise NotImplementedError("子类必须实现optimize方法")


class GridSearchOptimizer(ParameterOptimizer):
    """
    网格搜索优化器

    遍历所有参数组合，寻找最优解
    """

    def optimize(
        self,
        param_space: Dict[str, Any],
        optimization_target: str = "sharpe_ratio",
        n_trials: Optional[int] = None,
        **kwargs,
    ) -> OptimizationResult:
        """
        网格搜索优化

        Args:
            param_space: 参数空间，例如 {'period': [5, 10, 20], 'std_dev': [1.5, 2.0, 2.5]}
            optimization_target: 优化目标
            n_trials: 最大试验次数（None表示全部）
            **kwargs: 其他参数

        Returns:
            最佳结果
        """
        logger.info("开始网格搜索优化...")
        logger.info(f"参数空间: {param_space}")
        logger.info(f"优化目标: {optimization_target}")

        # 生成参数组合
        param_names = list(param_space.keys())
        param_values = list(param_space.values())

        all_combinations = list(product(*param_values))
        total_combinations = len(all_combinations)

        logger.info(f"总参数组合数: {total_combinations}")

        if n_trials is not None and n_trials < total_combinations:
            # 随机采样n_trials个组合
            indices = np.random.choice(total_combinations, n_trials, replace=False)
            combinations = [all_combinations[i] for i in indices]
        else:
            combinations = all_combinations

        self.results = []
        best_score = -np.inf
        best_params = None
        best_metrics = None

        for i, combination in enumerate(combinations):
            # 构建参数字典
            params = dict(zip(param_names, combination))

            logger.info(f"\n[{i+1}/{len(combinations)}] 测试参数: {params}")

            try:
                # 运行回测
                metrics = self._run_backtest(params)
                score = metrics.get(optimization_target, 0)

                logger.info(f"  {optimization_target}: {score:.4f}")

                # 保存结果
                result = OptimizationResult(
                    params=params.copy(),
                    metrics=metrics,
                    backtest_results={},
                )
                self.results.append(result)

                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    logger.info(f"  ★ 新的最佳参数！{optimization_target}={score:.4f}")

            except Exception as e:
                logger.error(f"  回测失败: {e}")
                continue

        logger.info(f"\n网格搜索完成！")
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳{optimization_target}: {best_score:.4f}")

        return OptimizationResult(
            params=best_params,
            metrics=best_metrics,
            backtest_results={},
        )


class RandomSearchOptimizer(ParameterOptimizer):
    """
    随机搜索优化器

    在参数空间中随机采样
    """

    def optimize(
        self,
        param_space: Dict[str, Any],
        optimization_target: str = "sharpe_ratio",
        n_trials: int = 50,
        **kwargs,
    ) -> OptimizationResult:
        """
        随机搜索优化

        Args:
            param_space: 参数空间，支持范围和离散值
                例如: {
                    'period': ('int', 5, 50),  # 均匀分布整数
                    'std_dev': ('uniform', 1.0, 3.0),  # 均匀分布浮点数
                    'threshold': ('log_uniform', 0.001, 0.1),  # 对数均匀分布
                    'method': ['categorical', 'sma', 'ema'],  # 分类变量
                }
            optimization_target: 优化目标
            n_trials: 随机试验次数
            **kwargs: 其他参数

        Returns:
            最佳结果
        """
        logger.info("开始随机搜索优化...")
        logger.info(f"参数空间: {param_space}")
        logger.info(f"优化目标: {optimization_target}")
        logger.info(f"试验次数: {n_trials}")

        self.results = []
        best_score = -np.inf
        best_params = None
        best_metrics = None

        for i in range(n_trials):
            # 采样参数
            params = self._sample_params(param_space)

            logger.info(f"\n[{i+1}/{n_trials}] 测试参数: {params}")

            try:
                # 运行回测
                metrics = self._run_backtest(params)
                score = metrics.get(optimization_target, 0)

                logger.info(f"  {optimization_target}: {score:.4f}")

                # 保存结果
                result = OptimizationResult(
                    params=params.copy(),
                    metrics=metrics,
                    backtest_results={},
                )
                self.results.append(result)

                # 更新最佳结果
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                    logger.info(f"  ★ 新的最佳参数！{optimization_target}={score:.4f}")

            except Exception as e:
                logger.error(f"  回测失败: {e}")
                continue

        logger.info(f"\n随机搜索完成！")
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳{optimization_target}: {best_score:.4f}")

        return OptimizationResult(
            params=best_params,
            metrics=best_metrics,
            backtest_results={},
        )

    def _sample_params(self, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """从参数空间采样"""
        params = {}

        for key, value in param_space.items():
            if isinstance(value, list):
                # 离散值列表
                if len(value) == 3 and value[0] in ["int", "uniform", "log_uniform", "categorical"]:
                    # 参数范围定义
                    param_type = value[0]

                    if param_type == "int":
                        params[key] = np.random.randint(value[1], value[2] + 1)
                    elif param_type == "uniform":
                        params[key] = np.random.uniform(value[1], value[2])
                    elif param_type == "log_uniform":
                        params[key] = np.exp(np.random.uniform(np.log(value[1]), np.log(value[2])))
                    elif param_type == "categorical":
                        params[key] = np.random.choice(value[1])
                else:
                    # 普通列表
                    params[key] = np.random.choice(value)
            else:
                params[key] = value

        return params


class BayesianOptimizer(ParameterOptimizer):
    """
    贝叶斯优化器

    使用Optuna进行贝叶斯优化
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请运行: pip install optuna")

    def optimize(
        self,
        param_space: Dict[str, Any],
        optimization_target: str = "sharpe_ratio",
        n_trials: int = 100,
        study_name: Optional[str] = None,
        direction: str = "maximize",
        **kwargs,
    ) -> OptimizationResult:
        """
        贝叶斯优化

        Args:
            param_space: Optuna参数空间定义
                例如: {
                    'period': ('int', 5, 50),
                    'std_dev': ('float', 1.0, 3.0),
                    'method': ['categorical', ['sma', 'ema']],
                }
            optimization_target: 优化目标
            n_trials: 试验次数
            study_name: study名称
            direction: 优化方向 ('maximize' or 'minimize')
            **kwargs: 其他参数

        Returns:
            最佳结果
        """
        logger.info("开始贝叶斯优化...")
        logger.info(f"优化目标: {optimization_target}")
        logger.info(f"试验次数: {n_trials}")

        # 创建study
        if study_name is None:
            study_name = f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(),
        )

        # 定义目标函数
        def objective(trial):
            # 采样参数
            params = self._suggest_params(trial, param_space)

            logger.info(f"测试参数: {params}")

            try:
                # 运行回测
                metrics = self._run_backtest(params)
                score = metrics.get(optimization_target, 0)

                logger.info(f"  {optimization_target}: {score:.4f}")

                # 保存结果
                result = OptimizationResult(
                    params=params.copy(),
                    metrics=metrics,
                    backtest_results={},
                )
                self.results.append(result)

                return score

            except Exception as e:
                logger.error(f"  回测失败: {e}")
                # 返回极差值
                return -1e10 if direction == "maximize" else 1e10

        # 运行优化
        study.optimize(objective, n_trials=n_trials)

        # 获取最佳结果
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value

        # 运行最佳参数的回测以获取完整指标
        best_metrics = self._run_backtest(best_params)

        logger.info(f"\n贝叶斯优化完成！")
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳{optimization_target}: {best_score:.4f}")

        return OptimizationResult(
            params=best_params,
            metrics=best_metrics,
            backtest_results={},
        )

    def _suggest_params(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """从trial采样参数"""
        params = {}

        for key, value in param_space.items():
            if isinstance(value, list) and len(value) >= 3:
                param_type = value[0]

                if param_type == "int":
                    params[key] = trial.suggest_int(key, value[1], value[2])
                elif param_type == "float":
                    params[key] = trial.suggest_float(key, value[1], value[2])
                elif param_type == "log_float":
                    params[key] = trial.suggest_float(key, value[1], value[2], log=True)
                elif param_type == "categorical":
                    params[key] = trial.suggest_categorical(key, value[1])
            else:
                params[key] = value

        return params


class MultiObjectiveOptimizer(ParameterOptimizer):
    """
    多目标优化器

    同时优化多个目标（如收益和风险）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请运行: pip install optuna")

    def optimize(
        self,
        param_space: Dict[str, Any],
        objectives: Dict[str, str],
        n_trials: int = 100,
        study_name: Optional[str] = None,
        **kwargs,
    ) -> List[OptimizationResult]:
        """
        多目标优化

        Args:
            param_space: 参数空间
            objectives: 目标字典，例如 {'sharpe_ratio': 'maximize', 'max_drawdown': 'minimize'}
            n_trials: 试验次数
            study_name: study名称
            **kwargs: 其他参数

        Returns:
            Pareto前沿结果列表
        """
        logger.info("开始多目标优化...")
        logger.info(f"优化目标: {objectives}")
        logger.info(f"试验次数: {n_trials}")

        # 创建多目标study
        if study_name is None:
            study_name = f"multi_objective_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        study = optuna.create_study(
            study_name=study_name,
            directions=[objectives[obj] for obj in objectives],
        )

        # 定义目标函数
        def objective(trial):
            # 采样参数
            params = self._suggest_params(trial, param_space)

            logger.info(f"测试参数: {params}")

            try:
                # 运行回测
                metrics = self._run_backtest(params)

                # 提取目标值
                values = [metrics.get(obj, 0) for obj in objectives]

                logger.info(f"  目标值: {values}")

                # 保存结果
                result = OptimizationResult(
                    params=params.copy(),
                    metrics=metrics,
                    backtest_results={},
                )
                self.results.append(result)

                return tuple(values)

            except Exception as e:
                logger.error(f"  回测失败: {e}")
                # 返回极差值
                return tuple([-1e10 if obj == "maximize" else 1e10 for obj in objectives])

        # 运行优化
        study.optimize(objective, n_trials=n_trials)

        # 获取Pareto最优解
        pareto_results = []
        for trial in study.best_trials:
            params = trial.params
            metrics = self._run_backtest(params)

            result = OptimizationResult(
                params=params,
                metrics=metrics,
                backtest_results={},
            )
            pareto_results.append(result)

        logger.info(f"\n多目标优化完成！")
        logger.info(f"找到 {len(pareto_results)} 个Pareto最优解")

        return pareto_results

    def _suggest_params(self, trial: optuna.Trial, param_space: Dict[str, Any]) -> Dict[str, Any]:
        """从trial采样参数"""
        params = {}

        for key, value in param_space.items():
            if isinstance(value, list) and len(value) >= 3:
                param_type = value[0]

                if param_type == "int":
                    params[key] = trial.suggest_int(key, value[1], value[2])
                elif param_type == "float":
                    params[key] = trial.suggest_float(key, value[1], value[2])
                elif param_type == "log_float":
                    params[key] = trial.suggest_float(key, value[1], value[2], log=True)
                elif param_type == "categorical":
                    params[key] = trial.suggest_categorical(key, value[1])
            else:
                params[key] = value

        return params


def create_optimizer(
    optimizer_type: str, data: pd.DataFrame, strategy_class: type, **kwargs
) -> ParameterOptimizer:
    """
    创建优化器的工厂函数

    Args:
        optimizer_type: 优化器类型 ('grid', 'random', 'bayesian', 'multi_objective')
        data: 回测数据
        strategy_class: 策略类
        **kwargs: 其他参数

    Returns:
        优化器实例
    """
    optimizer_classes = {
        "grid": GridSearchOptimizer,
        "random": RandomSearchOptimizer,
        "bayesian": BayesianOptimizer,
        "multi_objective": MultiObjectiveOptimizer,
    }

    optimizer_class = optimizer_classes.get(optimizer_type.lower())
    if optimizer_class is None:
        raise ValueError(f"未知的优化器类型: {optimizer_type}")

    return optimizer_class(data, strategy_class, **kwargs)


__all__ = [
    "ParameterOptimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "BayesianOptimizer",
    "MultiObjectiveOptimizer",
    "OptimizationResult",
    "create_optimizer",
]
