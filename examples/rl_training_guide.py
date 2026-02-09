"""
强化学习训练示例
展示如何使用quantA的RL系统进行模型训练
"""

import sys

sys.path.insert(0, '/home/rowan/Projects/quantA')

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.evaluation.model_evaluator import ModelEvaluator
from rl.optimization.hyperparameter_tuning import RLHyperparameterTuner
from rl.training.trainer import RLTrainer
from utils import logger


def generate_training_data(symbols: list, days: int = 252):
    """
    生成训练数据

    Args:
        symbols: 股票代码列表
        days: 天数

    Returns:
        dict: {symbol: DataFrame}
    """
    logger.info(f"生成{len(symbols)}只股票{days}天的训练数据...")

    data = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
        returns = np.random.normal(0.0005, 0.015, days)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            'high': prices * (1 + np.random.uniform(0, 0.02, days)),
            'low': prices * (1 - np.random.uniform(0, 0.02, days)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, days)
        })

        # 确保价格合理性
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, days))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, days))

        data[symbol] = df

    logger.info(f"数据生成完成: {sum(len(df) for df in data.values())}行")
    return data


def example_1_train_ppo():
    """示例1: 训练PPO模型"""
    logger.info("=" * 60)
    logger.info("示例1: 训练PPO模型")
    logger.info("=" * 60)

    # 生成训练数据
    symbols = ["600519.SH", "000001.SZ"]
    data = generate_training_data(symbols, days=252)  # 1年交易日

    # 创建环境
    logger.info("创建A股交易环境...")
    env = ASharesTradingEnv(
        data=data,
        symbols=symbols,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=20,
        reward_type='sharpe_ratio',  # 使用夏普比率作为奖励
    )

    # 创建训练器
    logger.info("创建PPO训练器...")
    trainer = RLTrainer(
        env=env,
        algorithm='ppo',
        model_path='models/ppo_a_share',
    )

    # 设置超参数
    hyperparams = {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
    }

    # 训练模型
    logger.info("开始训练...")
    trainer.train(
        total_timesteps=50000,
        hyperparams=hyperparams,
        save_freq=10000,
    )

    logger.info("训练完成!")
    logger.info(f"模型已保存到: {trainer.model_path}")

    # 评估模型
    logger.info("评估模型性能...")
    evaluator = ModelEvaluator(env=env)
    metrics = evaluator.evaluate(trainer.model, n_episodes=10)

    logger.info(f"\n评估结果:")
    logger.info(f"  平均总收益: {metrics['mean_total_reward']:,.2f}")
    logger.info(f"  平均收益率: {metrics['mean_return_pct']:.2f}%")
    logger.info(f"  平均夏普比率: {metrics['mean_sharpe_ratio']:.2f}")
    logger.info(f"  平均最大回撤: {metrics['mean_max_drawdown']:.2%}")

    return trainer, evaluator


def example_2_train_dqn():
    """示例2: 训练DQN模型"""
    logger.info("\n" + "=" * 60)
    logger.info("示例2: 训练DQN模型")
    logger.info("=" * 60)

    # 生成训练数据
    symbols = ["600519.SH"]
    data = generate_training_data(symbols, days=126)  # 半年

    # 创建环境（使用离散动作空间）
    logger.info("创建A股交易环境（离散动作）...")
    env = ASharesTradingEnv(
        data=data,
        symbols=symbols,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=20,
        reward_type='simple_profit',  # 使用简单收益作为奖励
        action_space='discrete',  # 离散动作空间
    )

    # 创建训练器
    logger.info("创建DQN训练器...")
    trainer = RLTrainer(
        env=env,
        algorithm='dqn',
        model_path='models/dqn_a_share',
    )

    # 设置超参数
    hyperparams = {
        'learning_rate': 1e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 32,
        'gamma': 0.99,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.05,
    }

    # 训练模型
    logger.info("开始训练...")
    trainer.train(
        total_timesteps=30000,
        hyperparams=hyperparams,
        save_freq=5000,
    )

    logger.info("训练完成!")
    logger.info(f"模型已保存到: {trainer.model_path}")

    return trainer


def example_3_hyperparameter_tuning():
    """示例3: 超参数调优"""
    logger.info("\n" + "=" * 60)
    logger.info("示例3: 超参数调优")
    logger.info("=" * 60)

    # 生成训练数据
    symbols = ["600519.SH"]
    data = generate_training_data(symbols, days=60)

    # 创建环境
    env = ASharesTradingEnv(
        data=data,
        symbols=symbols,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=10,
        reward_type='simple_profit',
    )

    # 创建调优器
    logger.info("创建超参数调优器...")
    tuner = RLHyperparameterTuner(
        env=env,
        algorithm='ppo',
        n_trials=5,  # 试验次数
        n_jobs=1,  # 并行任务数
    )

    # 定义搜索空间
    search_space = {
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'n_steps': [1024, 2048],
        'batch_size': [32, 64],
        'gamma': [0.99, 0.995],
    }

    # 运行调优
    logger.info("开始超参数调优...")
    best_params, best_score = tuner.optimize(
        search_space=search_space,
        timesteps_per_trial=10000,
        metric='sharpe_ratio',
    )

    logger.info(f"\n调优完成!")
    logger.info(f"最佳参数: {best_params}")
    logger.info(f"最佳得分: {best_score:.2f}")

    return tuner, best_params


def example_4_model_comparison():
    """示例4: 模型对比"""
    logger.info("\n" + "=" * 60)
    logger.info("示例4: 模型对比")
    logger.info("=" * 60)

    # 生成训练数据
    symbols = ["600519.SH"]
    data = generate_training_data(symbols, days=60)

    # 创建环境
    env = ASharesTradingEnv(
        data=data,
        symbols=symbols,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=10,
        reward_type='simple_profit',
    )

    # 创建评估器
    evaluator = ModelEvaluator(env=env)

    # 训练多个模型
    algorithms = ['ppo', 'dqn', 'a2c']
    models = {}
    results = {}

    for algo in algorithms:
        logger.info(f"\n训练{algo.upper()}模型...")

        trainer = RLTrainer(
            env=env,
            algorithm=algo,
            model_path=f'models/{algo}_comparison',
        )

        # 训练
        trainer.train(
            total_timesteps=10000,
            save_freq=5000,
        )

        # 评估
        metrics = evaluator.evaluate(trainer.model, n_episodes=5)

        models[algo] = trainer.model
        results[algo] = metrics

        logger.info(f"{algo.upper()}结果:")
        logger.info(f"  平均收益率: {metrics['mean_return_pct']:.2f}%")
        logger.info(f"  夏普比率: {metrics['mean_sharpe_ratio']:.2f}")

    # 对比结果
    logger.info("\n" + "=" * 60)
    logger.info("模型对比结果")
    logger.info("=" * 60)

    best_algo = max(results.keys(), key=lambda k: results[k]['mean_sharpe_ratio'])
    logger.info(f"最佳模型: {best_algo.upper()}")
    logger.info(f"夏普比率: {results[best_algo]['mean_sharpe_ratio']:.2f}")

    return models, results


def main():
    """主函数"""
    logger.info("quantA 强化学习训练指南")
    logger.info("=" * 60)

    # 运行示例
    try:
        # 基础训练示例
        example_1_train_ppo()
        example_2_train_dqn()

        # 高级示例
        example_3_hyperparameter_tuning()
        example_4_model_comparison()

        logger.info("\n" + "=" * 60)
        logger.info("所有示例运行完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"运行示例时出错: {e}", exc_info=True)


if __name__ == "__main__":
    main()
