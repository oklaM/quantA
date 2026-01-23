"""
强化学习训练器完整实现
支持PPO、DQN、A2C等多种算法的训练
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None
    DQN = None
    A2C = None

from rl.rewards.reward_functions import BaseRewardFunction, create_reward_function
from utils.logging import get_logger

logger = get_logger(__name__)


class TrainingCallback(BaseCallback):
    """自定义训练回调"""

    def __init__(
        self,
        eval_freq: int = 1000,
        eval_env=None,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.eval_results = []

    def _on_step(self) -> bool:
        """每n步调用一次"""
        if self.eval_env is None:
            return True

        if self.n_calls % self.eval_freq == 0:
            # 评估模型
            episode_rewards = []
            episode_lengths = []

            for _ in range(self.n_eval_episodes):
                obs, info = self.eval_env.reset()
                episode_reward = 0
                episode_length = 0

                while True:
                    action = self.model.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)

            self.eval_results.append(
                {
                    "timestep": self.num_timesteps,
                    "mean_reward": mean_reward,
                    "std_reward": std_reward,
                    "mean_length": mean_length,
                }
            )

            if self.verbose > 0:
                logger.info(
                    f"Evaluation at {self.num_timesteps} steps: "
                    f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )

            # 保存最佳模型
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    logger.info(f"New best mean reward: {self.best_mean_reward:.2f}")

                # 保存最佳模型
                self.model.save("models/best_model")

        return True


class RLTrainer:
    """
    RL训练器

    支持多种算法的训练、评估和保存
    """

    def __init__(
        self,
        env,
        algorithm: str = "ppo",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        tensorboard_log: str = "./logs/",
    ):
        """
        Args:
            env: 训练环境
            algorithm: 算法名称 (ppo, dqn, a2c, sac, td3)
            learning_rate: 学习率
            n_steps: 每次更新的步数
            tensorboard_log: TensorBoard日志目录
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable_baselines3未安装，请运行: pip install stable-baselines3")

        self.env = env
        self.algorithm = algorithm.lower()
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.tensorboard_log = tensorboard_log

        self.model = None
        self.training_log = []

        logger.info(f"RL训练器初始化: 算法={algorithm}")

    def build_model(self, **kwargs) -> Union[PPO, DQN, A2C, SAC, TD3]:
        """
        构建模型

        Returns:
            模型实例
        """
        # 通用参数
        common_params = {
            "learning_rate": self.learning_rate,
            "verbose": 1,
            "tensorboard_log": self.tensorboard_log,
        }

        # 算法特定参数
        if self.algorithm == "ppo":
            params = {
                "n_steps": kwargs.get("n_steps", self.n_steps),
                "batch_size": kwargs.get("batch_size", 64),
                "n_epochs": kwargs.get("n_epochs", 10),
                "gamma": kwargs.get("gamma", 0.99),
                "gae_lambda": kwargs.get("gae_lambda", 0.95),
                "clip_range": kwargs.get("clip_range", 0.2),
                "ent_coef": kwargs.get("ent_coef", 0.01),
            }
            model = PPO("MlpPolicy", self.env, **common_params, **params)

        elif self.algorithm == "dqn":
            params = {
                "buffer_size": kwargs.get("buffer_size", 100000),
                "learning_starts": kwargs.get("learning_starts", 1000),
                "batch_size": kwargs.get("batch_size", 32),
                "gamma": kwargs.get("gamma", 0.99),
                "exploration_fraction": kwargs.get("exploration_fraction", 0.1),
                "exploration_final_eps": kwargs.get("exploration_final_eps", 0.01),
                "target_network_update_freq": kwargs.get("target_network_update_freq", 500),
            }
            model = DQN("MlpPolicy", self.env, **common_params, **params)

        elif self.algorithm == "a2c":
            params = {
                "n_steps": kwargs.get("n_steps", self.n_steps),
                "gamma": kwargs.get("gamma", 0.99),
                "gae_lambda": kwargs.get("gae_lambda", 1.0),
            }
            model = A2C("MlpPolicy", self.env, **common_params, **params)

        elif self.algorithm == "sac":
            params = {
                "buffer_size": kwargs.get("buffer_size", 1000000),
                "learning_starts": kwargs.get("learning_starts", 100),
                "batch_size": kwargs.get("batch_size", 256),
                "gamma": kwargs.get("gamma", 0.99),
                "ent_coef": kwargs.get("ent_coef", "auto"),
            }
            model = SAC("MlpPolicy", self.env, **common_params, **params)

        elif self.algorithm == "td3":
            params = {
                "buffer_size": kwargs.get("buffer_size", 1000000),
                "learning_starts": kwargs.get("learning_starts", 100),
                "batch_size": kwargs.get("batch_size", 100),
                "gamma": kwargs.get("gamma", 0.99),
            }
            model = TD3("MlpPolicy", self.env, **common_params, **params)

        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

        self.model = model
        logger.info(f"模型构建完成: {self.algorithm}")
        return model

    def train(
        self,
        total_timesteps: int = 100000,
        save_freq: int = 10000,
        eval_env=None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        **kwargs,
    ):
        """
        训练模型

        Args:
            total_timesteps: 总训练步数
            save_freq: 保存频率
            eval_env: 评估环境
            eval_freq: 评估频率
            n_eval_episodes: 评估回合数
            **kwargs: 其他参数
        """
        if self.model is None:
            self.build_model(**kwargs)

        logger.info(f"开始训练 {self.algorithm}...")
        logger.info(f"总步数: {total_timesteps}")

        # 创建回调
        callbacks = []

        # 检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq, save_path="./checkpoints/", name_prefix=f"{self.algorithm}_model"
        )
        callbacks.append(checkpoint_callback)

        # 评估回调
        if eval_env is not None:
            eval_callback = TrainingCallback(
                eval_freq=eval_freq,
                eval_env=eval_env,
                n_eval_episodes=n_eval_episodes,
                verbose=1,
            )
            callbacks.append(eval_callback)

        callback_list = CallbackList(callbacks)

        # 训练
        start_time = datetime.now()
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            reset_num_timesteps=False,
        )
        end_time = datetime.now()

        training_duration = (end_time - start_time).total_seconds()
        logger.info(f"训练完成！耗时: {training_duration:.2f}秒")

        # 保存最终模型
        self.save(f"models/{self.algorithm}_final_model")

        return self.model

    def evaluate(
        self,
        env=None,
        n_episodes: int = 100,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            env: 评估环境（默认使用训练环境）
            n_episodes: 评估回合数
            deterministic: 是否使用确定性策略

        Returns:
            评估结果字典
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")

        eval_env = env or self.env

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0

            while True:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        results = {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "n_episodes": n_episodes,
        }

        logger.info(f"评估结果:")
        logger.info(f"  平均奖励: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        logger.info(f"  奖励范围: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        logger.info(f"  平均长度: {results['mean_length']:.2f}")

        return results

    def save(self, path: str):
        """
        保存模型

        Args:
            path: 保存路径
        """
        if self.model is None:
            raise ValueError("没有模型可保存")

        # 创建目录
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.model.save(path)
        logger.info(f"模型已保存到: {path}")

    def load(self, path: str):
        """
        加载模型

        Args:
            path: 模型路径
        """
        if self.algorithm == "ppo":
            self.model = PPO.load(path)
        elif self.algorithm == "dqn":
            self.model = DQN.load(path)
        elif self.algorithm == "a2c":
            self.model = A2C.load(path)
        elif self.algorithm == "sac":
            self.model = SAC.load(path)
        elif self.algorithm == "td3":
            self.model = TD3.load(path)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

        logger.info(f"模型已从 {path} 加载")

    def predict(self, observation, deterministic: bool = True):
        """
        预测动作

        Args:
            observation: 观察
            deterministic: 是否使用确定性策略

        Returns:
            动作
        """
        if self.model is None:
            raise ValueError("模型尚未加载")

        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def plot_training_curve(self, save_path: Optional[str] = None):
        """
        绘制训练曲线

        Args:
            save_path: 保存路径（可选）
        """
        try:
            # 加载TensorBoard日志
            x, y = ts2xy(load_results(self.tensorboard_log), "train")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, y)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Episode Reward")
            ax.set_title("Training Curve")
            ax.grid(True)

            if save_path:
                plt.savefig(save_path)
                logger.info(f"训练曲线已保存到: {save_path}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            logger.error(f"绘制训练曲线失败: {e}")


class MultiAgentTrainer:
    """
    多智能体训练器

    支持同时训练多个智能体
    """

    def __init__(self, envs: List, algorithms: List[str]):
        """
        Args:
            envs: 环境列表
            algorithms: 算法列表
        """
        if len(envs) != len(algorithms):
            raise ValueError("环境和算法数量必须相同")

        self.trainers = []

        for env, algorithm in zip(envs, algorithms):
            trainer = RLTrainer(env, algorithm=algorithm)
            self.trainers.append(trainer)

        logger.info(f"多智能体训练器初始化: {len(self.trainers)}个智能体")

    def train_all(self, total_timesteps: int = 100000, **kwargs):
        """
        训练所有智能体

        Args:
            total_timesteps: 总训练步数
            **kwargs: 其他参数
        """
        results = {}

        for i, trainer in enumerate(self.trainers):
            logger.info(f"训练智能体 {i+1}/{len(self.trainers)}...")

            model = trainer.train(total_timesteps=total_timesteps, **kwargs)
            results[i] = model

        return results

    def evaluate_all(self, **kwargs) -> List[Dict]:
        """评估所有智能体"""
        results = []

        for i, trainer in enumerate(self.trainers):
            logger.info(f"评估智能体 {i+1}/{len(self.trainers)}...")
            result = trainer.evaluate(**kwargs)
            results.append({"agent_id": i, "algorithm": trainer.algorithm, **result})

        return results


def create_trainer(env, algorithm: str = "ppo", **kwargs) -> RLTrainer:
    """
    创建训练器的工厂函数

    Args:
        env: 环境
        algorithm: 算法名称
        **kwargs: 其他参数

    Returns:
        RLTrainer实例
    """
    return RLTrainer(env, algorithm=algorithm, **kwargs)


__all__ = [
    "RLTrainer",
    "MultiAgentTrainer",
    "TrainingCallback",
    "create_trainer",
]
