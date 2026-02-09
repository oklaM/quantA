#!/usr/bin/env python3
"""
RL 系统演示脚本
展示 RL 系统的核心功能，不依赖 stable-baselines3
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# 导入 RL 组件
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.rewards.reward_functions import create_reward_function
from utils.logging import get_logger

logger = get_logger(__name__)

def create_demo_data(n_days=100):
    """创建演示数据"""
    np.random.seed(42)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_days)
    })

    return data

def demo_environment():
    """演示环境功能"""
    print("=== 环境功能演示 ===")
    print("创建交易环境...")

    data = create_demo_data(100)
    env = ASharesTradingEnv(data=data)

    print(f"环境信息：")
    print(f"  - 观察空间：{env.observation_space.shape[0]} 维")
    print(f"  - 动作空间：{env.action_space.n} 个动作")
    print(f"  - 数据长度：{len(data)} 天")
    print(f"  - 最大步数：{env.max_steps}")

    # 重置环境
    obs, info = env.reset()
    print(f"初始状态：现金={info.get('cash', 0):.2f}, 总资产={info.get('total_value', 0):.2f}")

    return env

def demo_reward_functions():
    """演示奖励函数"""
    print("\n=== 奖励函数演示 ===")

    # 创建奖励函数
    reward_types = ["simple", "sharpe", "risk_adjusted"]
    reward_funcs = {
        name: create_reward_function(name)
        for name in reward_types
    }

    # 演示奖励计算
    print("不同奖励函数的对比：")
    print("场景：买入100股，价格从100涨到101")

    for name, func in reward_funcs.items():
        result = func.calculate(
            action=1,
            current_price=101.0,
            portfolio_value=1010000,
            previous_value=1000000,
            position_size=1000
        )
        print(f"  {name}: {result.reward:.4f}")

    return reward_funcs

def demo_trading_strategy():
    """演示交易策略"""
    print("\n=== 交易策略演示 ===")

    data = create_demo_data(100)
    env = ASharesTradingEnv(data=data)

    # 简单的买入-持有-卖出策略
    obs, info = env.reset()
    print("初始状态：")
    print(f"  现金：{info.get('cash', 0):.2f}")
    print(f"  持仓：{info.get('position_size', 0)}")

    # 步骤1：买入
    print("\n步骤1：买入信号 -> 执行买入")
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"  买入后：现金={info.get('cash', 0):.2f}, 持仓={info.get('position_size', 0)}")

    # 步骤2：持有
    print("\n步骤2：持有信号 -> 执行持有")
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"  持有后：现金={info.get('cash', 0):.2f}, 持仓={info.get('position_size', 0)}")

    # 步骤3：卖出
    print("\n步骤3：卖出信号 -> 执行卖出")
    obs, reward, terminated, truncated, info = env.step(2)
    print(f"  卖出后：现金={info.get('cash', 0):.2f}, 持仓={info.get('position_size', 0)}")

    print(f"\n最终总资产：{info.get('total_value', 0):.2f}")
    print(f"总收益：{info.get('total_value', 0) - 1000000:.2f}")

def demo_random_agent():
    """演示随机智能体"""
    print("\n=== 随机智能体演示 ===")

    data = create_demo_data(100)
    env = ASharesTradingEnv(data=data)

    # 随机策略
    obs, info = env.reset()
    total_reward = 0
    actions = []

    print("随机策略运行 20 步：")
    print("步骤 | 动作 | 奖励 | 总资产")
    print("-" * 35)

    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        actions.append(action)

        action_name = {0: "持有", 1: "买入", 2: "卖出"}[action]
        print(f"{step+1:2d}   | {action_name}   | {reward:8.4f} | {info.get('total_value', 0):8.2f}")

        if terminated or truncated:
            break

    print(f"\n随机策略结果：")
    print(f"  总奖励：{total_reward:.4f}")
    print(f"  买入次数：{actions.count(1)}")
    print(f"  卖出次数：{actions.count(2)}")
    print(f"  持有次数：{actions.count(0)}")

def demo_feature_analysis():
    """演示特征分析"""
    print("\n=== 特征分析演示 ===")

    data = create_demo_data(100)
    env = ASharesTradingEnv(data=data)
    obs, info = env.reset()

    print("观察空间特征分析：")
    print(f"特征维度：{obs.shape[0]}")

    # 分析特征组成
    n_price_features = 5
    n_indicator_features = 15
    n_account_features = 2

    print(f"\n特征分布：")
    print("价格特征（前5维）：")
    for i in range(5):
        print(f"  特征{i}: {obs[i]:.4f}")

    print("\n技术指标（中间5维）：")
    for i in range(10, 15):
        print(f"  特征{i}: {obs[i]:.4f}")

    print("\n账户状态（最后2维）：")
    for i in range(-2, 0):
        print(f"  特征{i}: {obs[i]:.4f}")

    # 特征统计
    print(f"\n特征统计：")
    print(f"  最小值：{obs.min():.4f}")
    print(f"  最大值：{obs.max():.4f}")
    print(f"  均值：{obs.mean():.4f}")
    print(f"  标准差：{obs.std():.4f}")

def demo_training_pseudo():
    """模拟训练过程"""
    print("\n=== 训练过程模拟 ===")

    # 模拟训练数据
    episodes = 5
    episode_rewards = []
    episode_lengths = []

    print("模拟训练过程（5个episode）：")

    for episode in range(episodes):
        data = create_demo_data(100)
        env = ASharesTradingEnv(data=data)
        obs, info = env.reset()

        # 模拟训练
        episode_reward = 0
        episode_length = 0
        max_steps = 50

        while episode_length < max_steps:
            # 简单策略：如果持仓比例低则买入，高则卖出
            if info.get('position_size', 0) < 5000:
                action = 1  # 买入
            else:
                action = 2  # 卖出

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode {episode+1}: 奖励={episode_reward:.2f}, 长度={episode_length}")

    print(f"\n训练结果统计：")
    print(f"  平均奖励：{np.mean(episode_rewards):.2f}")
    print(f"  奖励标准差：{np.std(episode_rewards):.2f}")
    print(f"  平均长度：{np.mean(episode_lengths):.2f}")

    return episode_rewards

def main():
    """主函数"""
    print("quantA RL 系统演示")
    print("=" * 50)

    # 运行所有演示
    demo_environment()
    demo_reward_functions()
    demo_trading_strategy()
    demo_random_agent()
    demo_feature_analysis()
    demo_training_pseudo()

    print("\n" + "=" * 50)
    print("演示完成")
    print("\n注意：由于缺少 stable_baselines3 依赖，实际训练功能需要安装依赖后使用。")
    print("但系统的环境、奖励函数、交易逻辑等核心功能都已就绪。")

if __name__ == "__main__":
    main()