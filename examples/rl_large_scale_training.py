"""
RL模型大规模训练示例

展示如何进行大规模强化学习模型训练
"""

import json
import multiprocessing as mp
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from stable_baselines3 import A2C, DQN, PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.results_plotter import load_results, ts2xy
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("警告: stable-baselines3未安装")

from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.evaluation.model_evaluator import ModelComparator, ModelEvaluator
from rl.models.model_manager import ModelManager
from rl.training.trainer import RLTrainer
from utils.logging import get_logger

logger = get_logger(__name__)


class TrainingProgressCallback(BaseCallback):
    """训练进度回调"""

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.start_time = None
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        self.start_time = time.time()
        logger.info("=" * 70)
        logger.info("开始大规模RL训练")
        logger.info("=" * 70)

    def _on_rollout_end(self):
        if self.verbose > 0:
            logger.info(f"Rollout完成: {self.num_timesteps} 步")

    def _on_step(self):
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            elapsed = time.time() - self.start_time
            speed = self.num_timesteps / elapsed if elapsed > 0 else 0
            logger.info(f"训练进度: {self.num_timesteps} 步, 速度: {speed:.0f} 步/秒")

        return True


def generate_large_dataset(
    symbols: List[str],
    days: int = 2000,
    start_date: str = "2018-01-01",
) -> Dict[str, pd.DataFrame]:
    """
    生成大规模训练数据集

    Args:
        symbols: 股票代码列表
        days: 每只股票的天数
        start_date: 起始日期

    Returns:
        股票数据字典
    """
    logger.info(f"生成大规模数据集: {len(symbols)} 只股票, {days} 天")

    datasets = {}

    for symbol in symbols:
        np.random.seed(hash(symbol) % (2**32))

        # 生成价格序列
        returns = np.random.normal(0.0005, 0.02, days)
        prices = 100.0 * (1 + returns).cumprod()

        # 生成日期
        dates = pd.date_range(start=start_date, periods=days, freq='D')

        # 生成OHLC
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.015)))
            low = close * (1 - abs(np.random.normal(0, 0.015)))
            open_price = close * (1 + np.random.normal(0, 0.008))

            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': np.random.randint(1000000, 10000000),
            })

        datasets[symbol] = pd.DataFrame(data)
        logger.info(f"生成数据完成: {symbol}, {len(data)} 条记录")

    return datasets


def create_training_env(data: pd.DataFrame, window_size: int = 60) -> ASharesTradingEnv:
    """创建训练环境"""
    return ASharesTradingEnv(
        data=data.reset_index(drop=True),
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=window_size,
    )


def train_large_scale_ppo(
    datasets: Dict[str, pd.DataFrame],
    total_timesteps: int = 1000000,
    n_envs: int = 4,
    model_path: str = "models/large_scale_ppo",
):
    """
    大规模PPO训练

    Args:
        datasets: 数据集字典
        total_timesteps: 总训练步数
        n_envs: 并行环境数
        model_path: 模型保存路径
    """
    logger.info("=" * 70)
    logger.info("大规模PPO训练")
    logger.info("=" * 70)

    # 选择主要股票进行训练
    main_symbol = list(datasets.keys())[0]
    train_data = datasets[main_symbol]

    # 创建多个环境
    logger.info(f"创建 {n_envs} 个并行环境")

    def make_env():
        return lambda: create_training_env(train_data)

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env()])

    # 创建训练器
    trainer = RLTrainer(
        env=env,
        algorithm="ppo",
        learning_rate=3e-4,
        tensorboard_log="./logs/large_scale_ppo/",
    )

    # 创建模型
    model = trainer.build_model(
        n_steps=2048 * n_envs,
        batch_size=64 * n_envs,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # 创建回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/large_scale_ppo/",
        name_prefix="ppo_model",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_ppo",
        log_path="./logs/eval_ppo",
        eval_freq=20000,
        n_eval_episodes=10,
        verbose=1,
    )

    progress_callback = TrainingProgressCallback(verbose=1)

    # 训练
    logger.info(f"开始训练: {total_timesteps} 步")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback, progress_callback],
        progress_bar=True,
    )

    end_time = time.time()
    training_duration = (end_time - start_time) / 3600

    logger.info(f"训练完成! 耗时: {training_duration:.2f} 小时")

    # 保存模型
    logger.info(f"保存模型到: {model_path}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    # 使用模型管理器
    manager = ModelManager()
    version = manager.save_model(
        model=model,
        algorithm="ppo",
        metadata={
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "training_duration_hours": training_duration,
            "dataset_size": len(train_data),
        },
    )

    logger.info(f"模型版本: {version.version_id}")

    return model, version


def multi_symbol_training(
    datasets: Dict[str, pd.DataFrame],
    total_timesteps_per_symbol: int = 100000,
):
    """
    多股票训练

    Args:
        datasets: 数据集字典
        total_timesteps_per_symbol: 每只股票的训练步数
    """
    logger.info("=" * 70)
    logger.info(f"多股票训练: {len(datasets)} 只股票")
    logger.info("=" * 70)

    results = {}

    for symbol, data in datasets.items():
        logger.info(f"\n训练股票: {symbol}")

        # 创建环境
        env = create_training_env(data)

        # 创建训练器
        trainer = RLTrainer(
            env=env,
            algorithm="ppo",
            learning_rate=3e-4,
        )

        # 训练
        model = trainer.train(
            total_timesteps=total_timesteps_per_symbol,
            save_freq=20000,
        )

        # 评估
        evaluator = ModelEvaluator(env, n_episodes=50)
        result = evaluator.evaluate(model, symbol)

        results[symbol] = {
            'mean_reward': result.mean_reward,
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0),
            'win_rate': result.metrics.get('win_rate', 0),
        }

        logger.info(f"{symbol} 训练完成: 奖励={result.mean_reward:.2f}")

    # 显示结果
    results_df = pd.DataFrame(results).T
    logger.info(f"\n多股票训练结果:\n{results_df.to_string()}")

    return results


def hyperparameter_tuning(
    data: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    超参数调优

    Args:
        data: 训练数据
        param_grid: 参数网格
        n_trials: 试验次数

    Returns:
        最佳参数
    """
    logger.info("=" * 70)
    logger.info("超参数调优")
    logger.info("=" * 70)

    best_score = -np.inf
    best_params = None

    for trial in range(n_trials):
        # 随机选择参数
        params = {
            'learning_rate': np.random.choice(param_grid['learning_rate']),
            'n_steps': np.random.choice(param_grid['n_steps']),
            'batch_size': np.random.choice(param_grid['batch_size']),
            'gamma': np.random.choice(param_grid['gamma']),
        }

        logger.info(f"\n试验 {trial + 1}/{n_trials}")
        logger.info(f"参数: {params}")

        # 创建环境
        env = create_training_env(data)

        # 创建模型
        model = PPO("MlpPolicy", env, verbose=0, **params)

        # 短暂训练
        model.learn(total_timesteps=10000, verbose=0)

        # 评估
        evaluator = ModelEvaluator(env, n_episodes=10)
        result = evaluator.evaluate(model, f"trial_{trial}")

        score = result.mean_reward
        logger.info(f"得分: {score:.2f}")

        if score > best_score:
            best_score = score
            best_params = params
            logger.info(f"✨ 新的最佳参数! 得分: {score:.2f}")

    logger.info(f"\n最佳参数: {best_params}")
    logger.info(f"最佳得分: {best_score:.2f}")

    return best_params


def evaluate_trained_model(
    model_path: str,
    test_data: pd.DataFrame,
    n_episodes: int = 100,
) -> Dict[str, Any]:
    """
    评估训练好的模型

    Args:
        model_path: 模型路径
        test_data: 测试数据
        n_episodes: 测试回合数

    Returns:
        评估结果
    """
    logger.info("=" * 70)
    logger.info("评估训练模型")
    logger.info("=" * 70)

    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    # 创建环境
    env = create_training_env(test_data)

    # 评估
    evaluator = ModelEvaluator(env, n_episodes=n_episodes)
    result = evaluator.evaluate(model, "large_scale_model")

    logger.info(f"\n评估结果:")
    logger.info(f"  平均奖励: {result.mean_reward:.2f} +/- {result.std_reward:.2f}")
    logger.info(f"  夏普比率: {result.metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  胜率: {result.metrics.get('win_rate', 0):.2%}")

    return result.to_dict()


def plot_training_results(log_dir: str = "./logs/large_scale_ppo/"):
    """绘制训练结果"""
    logger.info(f"绘制训练结果: {log_dir}")

    try:
        x, y = ts2xy(load_results(log_dir), 'train')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 奖励曲线
        ax1.plot(x, y)
        ax1.set_xlabel('Timesteps')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Reward')
        ax1.grid(True)

        # 滚动平均
        window = 100
        if len(y) >= window:
            rolling_mean = pd.Series(y).rolling(window).mean()
            ax2.plot(x, rolling_mean)
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel(f'Rolling Mean Reward (window={window})')
            ax2.set_title('Smoothed Training Curve')
            ax2.grid(True)

        plt.tight_layout()

        save_path = "logs/training_curve.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练曲线已保存: {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"绘制训练曲线失败: {e}")


def main():
    """主函数 - 大规模训练示例"""
    print("\n" + "=" * 70)
    print("quantA 大规模RL训练示例")
    print("=" * 70)

    if not SB3_AVAILABLE:
        print("\n错误: 需要安装 stable-baselines3 和 gymnasium")
        return

    # 配置
    SYMBOLS = ["000001.SZ", "000002.SZ", "000004.SZ", "600519.SH"]
    TRAINING_DAYS = 1000
    TOTAL_TIMESTEPS = 500000
    N_ENVS = 4

    try:
        # 1. 生成大规模数据
        print("\n步骤1: 生成训练数据")
        datasets = generate_large_dataset(
            symbols=SYMBOLS,
            days=TRAINING_DAYS,
            start_date="2020-01-01",
        )

        # 2. 大规模PPO训练
        print("\n步骤2: 大规模PPO训练")
        model, version = train_large_scale_ppo(
            datasets=datasets,
            total_timesteps=TOTAL_TIMESTEPS,
            n_envs=N_ENVS,
        )

        # 3. 绘制训练曲线
        print("\n步骤3: 绘制训练结果")
        plot_training_results()

        # 4. 评估模型
        print("\n步骤4: 评估模型")
        test_data = datasets[SYMBOLS[0]]
        evaluation_results = evaluate_trained_model(
            model_path="models/large_scale_ppo",
            test_data=test_data,
            n_episodes=50,
        )

        # 5. 超参数调优（可选，耗时较长）
        # print("\n步骤5: 超参数调优")
        # param_grid = {
        #     'learning_rate': [1e-4, 3e-4, 1e-3],
        #     'n_steps': [1024, 2048, 4096],
        #     'batch_size': [32, 64, 128],
        #     'gamma': [0.95, 0.99, 0.995],
        # }
        # best_params = hyperparameter_tuning(test_data, param_grid, n_trials=5)

        print("\n" + "=" * 70)
        print("大规模训练完成!")
        print("=" * 70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
