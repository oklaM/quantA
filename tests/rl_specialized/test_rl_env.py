#!/usr/bin/env python3
"""
RL 环境测试脚本
测试 Gymnasium 环境的创建、重置和交互
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta

import gymnasium as gym
import numpy as np
import pandas as pd

from rl.envs.a_share_trading_env import ASharesTradingEnv


def create_sample_data():
    """创建示例交易数据"""
    # 生成100天的模拟数据
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(100)]

    # 生成模拟价格数据
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, 100)  # 日收益率
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
        'volume': np.random.randint(100000, 1000000, 100)
    })

    return data

def test_environment():
    """测试环境的基本功能"""
    print("=" * 60)
    print("测试 Gymnasium 环境创建和交互")
    print("=" * 60)

    # 创建示例数据
    data = create_sample_data()
    print(f"✓ 创建示例数据: {len(data)} 天数据")

    # 创建环境
    env = ASharesTradingEnv(data=data)
    print(f"✓ 环境创建成功: {env.__class__.__name__}")

    # 检查环境信息
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"观察维度: {env.observation_space.shape}")
    print(f"动作类型: {type(env.action_space)}")

    # 测试重置
    obs, info = env.reset()
    print(f"✓ 重置成功")
    print(f"初始观察形状: {obs.shape}")
    print(f"初始观察值: {obs[:5]}...")  # 只显示前5个值

    # 测试几个随机步骤
    print("\n测试环境交互（5个随机步骤）:")
    print("-" * 40)

    for i in range(5):
        # 随机选择动作
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"步骤 {i+1}:")
        print(f"  动作: {action} (0=hold, 1=buy, 2=sell)")
        print(f"  奖励: {reward:.4f}")
        print(f"  终止: {terminated}, 截断: {truncated}")
        print(f"  持仓: {info.get('position', 'N/A')}")
        print(f"  现金: {info.get('cash', 'N/A'):.2f}")
        print()

        if terminated or truncated:
            print("环境结束，重置...")
            obs, info = env.reset()

    env.close()
    print("✓ 环境测试完成")

if __name__ == "__main__":
    test_environment()