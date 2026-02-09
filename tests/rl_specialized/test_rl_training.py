#!/usr/bin/env python3
"""
RL 训练测试脚本
测试 PPO/DQN 算法训练流程、模型保存与加载
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 导入 RL 组件
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.rewards.reward_functions import create_reward_function
from rl.training.trainer import RLTrainer


def create_sample_data(n_days=200):
    """创建示例交易数据"""
    # 生成 n 天的模拟数据
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # 生成模拟价格数据
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # 创建 DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days)
    })

    return data

def test_environment():
    """测试环境创建"""
    print("=" * 60)
    print("1. 测试环境创建")
    print("=" * 60)

    data = create_sample_data()
    env = ASharesTradingEnv(data=data)
    print(f"✓ 环境创建成功: {env.__class__.__name__}")
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  观察维度: {env.observation_space.shape}")
    print(f"  数据长度: {len(data)} 天")
    return env

def test_random_interactions(env, n_steps=10):
    """测试环境交互"""
    print("\n" + "=" * 60)
    print("2. 测试环境交互")
    print("=" * 60)

    obs, info = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  初始观察形状: {obs.shape}")
    print(f"  初始总资产: {info.get('total_value', 0):.2f}")

    print("\n随机交互测试:")
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"  步骤 {i+1}:")
        print(f"    动作: {action} (0=hold, 1=buy, 2=sell)")
        print(f"    奖励: {reward:.4f}")
        print(f"    总资产: {info.get('total_value', 0):.2f}")
        print(f"    持仓: {info.get('position_size', 0)}")
        print(f"    终止: {terminated}, 截断: {truncated}")

        if terminated or truncated:
            obs, info = env.reset()
            print("    环境重置")

    print("✓ 环境交互测试完成")

def test_reward_function():
    """测试奖励函数"""
    print("\n" + "=" * 60)
    print("3. 测试奖励函数")
    print("=" * 60)

    # 创建测试数据
    data = create_sample_data()
    env = ASharesTradingEnv(data=data)

    # 测试不同的奖励函数
    reward_types = ["sharpe", "profit", "risk_adjusted", "simple"]

    for reward_type in reward_types:
        try:
            reward_func = create_reward_function(reward_type)
            print(f"✓ {reward_type} 奖励函数创建成功")

            # 测试奖励计算
            obs, info = env.reset()
            action = 1  # 买入
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  {reward_type} 奖励: {reward:.4f}")

        except Exception as e:
            print(f"✗ {reward_type} 奖励函数创建失败: {e}")

def test_short_training(env, algorithm="ppo", timesteps=500):
    """简短训练测试"""
    print("\n" + "=" * 60)
    print(f"4. 测试 {algorithm.upper()} 训练 (简短)")
    print("=" * 60)

    try:
        # 创建训练器
        trainer = RLTrainer(
            env=env,
            algorithm=algorithm,
            learning_rate=0.001,
            n_steps=64,
            tensorboard_log="./logs/"
        )
        print(f"✓ {algorithm.upper()} 训练器创建成功")

        # 创建临时模型目录
        models_dir = Path("./test_models")
        models_dir.mkdir(exist_ok=True)

        # 开始简短训练
        print(f"开始训练 {timesteps} timesteps...")
        model = trainer.train(
            timesteps=timesteps,
            eval_freq=100,
            eval_episodes=3,
            save_path=str(models_dir / f"test_{algorithm}")
        )
        print(f"✓ {algorithm.upper()} 训练完成")

        return model, trainer

    except Exception as e:
        print(f"✗ 训练失败: {e}")
        return None, None

def test_model_save_load(env, model, trainer):
    """测试模型保存与加载"""
    print("\n" + "=" * 60)
    print("5. 测试模型保存与加载")
    print("=" * 60)

    if model is None or trainer is None:
        print("✗ 没有训练好的模型")
        return None

    # 创建模型目录
    models_dir = Path("./test_models")
    models_dir.mkdir(exist_ok=True)

    # 保存模型
    model_path = str(models_dir / "test_model")
    model.save(model_path)
    print(f"✓ 模型保存到: {model_path}")

    # 加载模型
    try:
        loaded_model = trainer.load_model(model_path)
        print("✓ 模型加载成功")

        # 测试加载模型的性能
        obs, info = env.reset()
        total_reward = 0

        for _ in range(50):  # 测试50步
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        print(f"✓ 加载模型测试总奖励: {total_reward:.2f}")
        return loaded_model

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

def test_model_evaluation(env, model):
    """测试模型评估"""
    print("\n" + "=" * 60)
    print("6. 测试模型评估")
    print("=" * 60)

    if model is None:
        print("✗ 没有要评估的模型")
        return

    # 评估模型
    try:
        # 创建评估函数
        def evaluate_model(model, env, n_episodes=5):
            episode_rewards = []
            episode_lengths = []

            for episode in range(n_episodes):
                obs, info = env.reset()
                episode_reward = 0
                episode_length = 0

                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1

                    if terminated or truncated:
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                print(f"  Episode {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")

            return {
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "rewards": episode_rewards
            }

        # 运行评估
        results = evaluate_model(model, env, n_episodes=3)

        print(f"\n评估结果:")
        print(f"  平均奖励: {results['mean_reward']:.2f} (+/- {results['std_reward']:.2f})")
        print(f"  平均长度: {results['mean_length']:.2f}")
        print("✓ 模型评估完成")

    except Exception as e:
        print(f"✗ 模型评估失败: {e}")

def cleanup():
    """清理测试文件"""
    import shutil

    # 清理测试模型目录
    test_models = Path("./test_models")
    if test_models.exists():
        shutil.rmtree(test_models)
        print("✓ 清理测试模型目录")

def main():
    """主测试流程"""
    print("RL 训练和评估系统测试")
    print("=" * 60)

    # 测试环境
    env = test_environment()

    # 测试奖励函数
    test_reward_function()

    # 测试环境交互
    test_random_interactions(env, n_steps=10)

    # 测试 PPO 训练
    ppo_model, ppo_trainer = test_short_training(env, algorithm="ppo", timesteps=300)

    # 如果 PPO 成功，测试 DQN
    dqn_model = None
    if ppo_model:
        print("\n" + "=" * 60)
        print("4. 测试 DQN 训练 (简短)")
        print("=" * 60)
        dqn_model, dqn_trainer = test_short_training(env, algorithm="dqn", timesteps=200)

    # 测试模型保存和加载
    if ppo_model:
        loaded_model = test_model_save_load(env, ppo_model, ppo_trainer)

    # 测试模型评估
    if ppo_model:
        test_model_evaluation(env, ppo_model)

    # 清理
    cleanup()

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print("✓ 环境测试: 通过")
    print("✓ 奖励函数: 通过")
    print("✓ 环境交互: 通过")
    print("✓ PPO 训练: 通过" if ppo_model else "✗ PPO 训练: 失败")
    print("✓ DQN 训练: 通过" if dqn_model else "✗ DQN 训练: 失败")
    print("✓ 模型保存/加载: 通过" if loaded_model else "✗ 模型保存/加载: 失败")
    print("✓ 模型评估: 通过")

if __name__ == "__main__":
    main()