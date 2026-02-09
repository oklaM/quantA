"""
强化学习超参数优化模块
提供多种优化算法用于RL模型超参数调优
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import optuna
    from optuna.pruners import HyperbandPruner, MedianPruner
    from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

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
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

    # Create a dummy BaseCallback for type annotations
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0

        def _on_step(self) -> bool:
            return True

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TuningResult:
    """调优结果"""

    trial_number: int
    params: Dict[str, Any]
    mean_reward: float
    std_reward: float
    best_mean_reward: float
    datetime: str = field(default_factory=lambda: datetime.now().isoformat())


class TrialEvalCallback(BaseCallback):
    """
    试验评估回调

    在训练过程中定期评估模型
    """

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int = 5,
        eval_freq: int = 1000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        每eval_freq步评估一次

        Returns:
            是否继续训练
        """
        if self.n_calls % self.eval_freq == 0:
            # 评估模型
            episode_rewards = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                episode_reward = 0

                while True:
                    action = self.model.predict(obs, deterministic=self.deterministic)[0]
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            # 更新最佳奖励
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            # 报告中间结果给Optuna
            if self.parent is not None and hasattr(self.parent, "report"):
                self.parent.report(mean_reward, self.n_calls)

            # 早停判断
            if self.is_pruned:
                return False

        return True


class RLHyperparameterTuner:
    """
    RL超参数调优器

    使用Optuna进行超参数优化
    """

    def __init__(
        self,
        env,
        eval_env,
        algorithm: str = "ppo",
        n_trials: int = 50,
        n_startup_trials: int = 10,
        n_eval_episodes: int = 10,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        optimization_metric: str = "mean_reward",
    ):
        """
        Args:
            env: 训练环境
            eval_env: 评估环境
            algorithm: 算法名称 (ppo, dqn, a2c, sac, td3)
            n_trials: 试验次数
            n_startup_trials: 随机探索试验次数
            n_eval_episodes: 评估回合数
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            optimization_metric: 优化指标
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安装，请运行: pip install optuna")
        if not SB3_AVAILABLE:
            raise ImportError("stable_baselines3未安装，请运行: pip install stable-baselines3")

        self.env = env
        self.eval_env = eval_env
        self.algorithm = algorithm.lower()
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.n_eval_episodes = n_eval_episodes
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.optimization_metric = optimization_metric

        self.study = None
        self.results: List[TuningResult] = []

        logger.info(f"RL超参数调优器初始化: 算法={algorithm}")

    def _get_algorithm_class(self):
        """获取算法类"""
        algorithms = {
            "ppo": PPO,
            "dqn": DQN,
            "a2c": A2C,
            "sac": SAC,
            "td3": TD3,
        }

        algo_class = algorithms.get(self.algorithm)
        if algo_class is None:
            raise ValueError(f"不支持的算法: {self.algorithm}")

        return algo_class

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        采样超参数

        Args:
            trial: Optuna trial对象

        Returns:
            超参数字典
        """
        params = {}

        if self.algorithm == "ppo":
            # PPO超参数
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            params["n_steps"] = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
            params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
            params["n_epochs"] = trial.suggest_int("n_epochs", 5, 20)
            params["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
            params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)
            params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.4)
            params["ent_coef"] = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)

        elif self.algorithm == "dqn":
            # DQN超参数
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            params["buffer_size"] = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
            params["learning_starts"] = trial.suggest_categorical(
                "learning_starts", [1000, 5000, 10000]
            )
            params["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            params["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
            params["exploration_fraction"] = trial.suggest_float("exploration_fraction", 0.05, 0.3)
            params["exploration_final_eps"] = trial.suggest_float(
                "exploration_final_eps", 0.001, 0.05
            )
            params["target_network_update_freq"] = trial.suggest_categorical(
                "target_network_update_freq", [100, 500, 1000]
            )

        elif self.algorithm == "a2c":
            # A2C超参数
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            params["n_steps"] = trial.suggest_categorical("n_steps", [5, 10, 20, 30])
            params["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
            params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 1.0)
            params["ent_coef"] = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)

        elif self.algorithm == "sac":
            # SAC超参数
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            params["buffer_size"] = trial.suggest_categorical(
                "buffer_size", [100000, 500000, 1000000]
            )
            params["learning_starts"] = trial.suggest_categorical(
                "learning_starts", [100, 1000, 5000]
            )
            params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
            params["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)
            params["ent_coef"] = trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1])

        elif self.algorithm == "td3":
            # TD3超参数
            params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            params["buffer_size"] = trial.suggest_categorical(
                "buffer_size", [100000, 500000, 1000000]
            )
            params["learning_starts"] = trial.suggest_categorical(
                "learning_starts", [100, 1000, 5000]
            )
            params["batch_size"] = trial.suggest_categorical("batch_size", [64, 100, 256])
            params["gamma"] = trial.suggest_float("gamma", 0.9, 0.9999)

        return params

    def _objective(self, trial: optuna.Trial) -> float:
        """
        优化目标函数

        Args:
            trial: Optuna trial对象

        Returns:
            优化目标值
        """
        # 采样超参数
        hyperparams = self._sample_hyperparameters(trial)

        logger.info(f"\nTrial {trial.number}: 测试超参数")
        logger.info(f"参数: {hyperparams}")

        try:
            # 创建模型
            Algorithm = self._get_algorithm_class()
            model = Algorithm("MlpPolicy", self.env, verbose=0, **hyperparams)

            # 创建评估回调
            eval_callback = TrialEvalCallback(
                eval_env=self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=self.eval_freq,
                deterministic=True,
                verbose=0,
            )

            # 训练模型
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=eval_callback,
                reset_num_timesteps=True,
            )

            # 评估模型
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                episode_reward = 0

                while True:
                    action = model.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            logger.info(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

            # 保存结果
            result = TuningResult(
                trial_number=trial.number,
                params=hyperparams.copy(),
                mean_reward=mean_reward,
                std_reward=std_reward,
                best_mean_reward=eval_callback.best_mean_reward,
            )
            self.results.append(result)

            return mean_reward

        except Exception as e:
            logger.error(f"试验失败: {e}")
            return -1e10

    def optimize(
        self,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        sampler: Optional[str] = "tpe",
        pruner: Optional[str] = "median",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        执行超参数优化

        Args:
            study_name: study名称
            storage: 存储路径（用于持久化）
            sampler: 采样器 (tpe, cmaes, random)
            pruner: 剪枝器 (median, hyperband, none)
            **kwargs: 其他参数

        Returns:
            优化结果字典
        """
        logger.info("=" * 70)
        logger.info(f"开始{self.algorithm.upper()}超参数优化")
        logger.info("=" * 70)
        logger.info(f"试验次数: {self.n_trials}")
        logger.info(f"训练步数: {self.total_timesteps}")
        logger.info(f"评估回合数: {self.n_eval_episodes}")

        # 创建study名称
        if study_name is None:
            study_name = f"{self.algorithm}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 选择采样器
        if sampler == "tpe":
            sampler_obj = TPESampler(n_startup_trials=self.n_startup_trials, multivariate=True)
        elif sampler == "cmaes":
            sampler_obj = CmaEsSampler()
        elif sampler == "random":
            sampler_obj = RandomSampler()
        else:
            sampler_obj = TPESampler()

        # 选择剪枝器
        if pruner == "median":
            pruner_obj = MedianPruner(
                n_startup_trials=self.n_startup_trials, n_warmup_steps=self.eval_freq * 2
            )
        elif pruner == "hyperband":
            pruner_obj = HyperbandPruner()
        else:
            pruner_obj = MedianPruner()

        # 创建study
        self.study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler_obj,
            pruner=pruner_obj,
            storage=storage,
            load_if_exists=True,
        )

        # 运行优化
        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        # 获取最佳结果
        best_trial = self.study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value

        logger.info("\n" + "=" * 70)
        logger.info("优化完成！")
        logger.info("=" * 70)
        logger.info(f"最佳试验: {best_trial.number}")
        logger.info(f"最佳参数: {best_params}")
        logger.info(f"最佳{self.optimization_metric}: {best_value:.4f}")

        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_trial": best_trial.number,
            "n_trials": len(self.study.trials),
            "study_name": study_name,
        }

    def get_best_model(self, save_path: Optional[str] = None):
        """
        使用最佳参数训练模型

        Args:
            save_path: 模型保存路径

        Returns:
            训练好的模型
        """
        if self.study is None:
            raise ValueError("请先运行optimize()方法")

        best_params = self.study.best_params

        logger.info("使用最佳参数训练模型...")
        logger.info(f"参数: {best_params}")

        # 创建模型
        Algorithm = self._get_algorithm_class()
        model = Algorithm("MlpPolicy", self.env, verbose=1, **best_params)

        # 训练
        model.learn(total_timesteps=self.total_timesteps * 2)  # 更长的训练

        # 保存
        if save_path:
            model.save(save_path)
            logger.info(f"模型已保存到: {save_path}")

        return model

    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        绘制优化历史

        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt

            if self.study is None:
                raise ValueError("请先运行optimize()方法")

            fig = optuna.visualization.matplotlib.plot_optimization_history(self.study)
            plt.title("Optimization History")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"优化历史图已保存到: {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"绘制优化历史失败: {e}")

    def plot_param_importances(self, save_path: Optional[str] = None):
        """
        绘制参数重要性

        Args:
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt

            if self.study is None:
                raise ValueError("请先运行optimize()方法")

            fig = optuna.visualization.matplotlib.plot_param_importances(self.study)
            plt.title("Parameter Importances")

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"参数重要性图已保存到: {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"绘制参数重要性失败: {e}")

    def save_results(self, path: str):
        """
        保存调优结果

        Args:
            path: 保存路径
        """
        if self.study is None:
            raise ValueError("请先运行optimize()方法")

        results_data = {
            "algorithm": self.algorithm,
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": str(t.state),
                }
                for t in self.study.trials
            ],
            "datetime": datetime.now().isoformat(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"调优结果已保存到: {path}")


__all__ = [
    "RLHyperparameterTuner",
    "TuningResult",
    "TrialEvalCallback",
]
