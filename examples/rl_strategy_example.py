"""
强化学习策略使用示例

展示如何使用RL环境训练和测试交易策略
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("警告: stable_baselines3未安装，请运行: pip install stable-baselines3")

from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_sample_data(days: int = 500, start_price: float = 100.0):
    """
    生成示例数据用于RL训练

    Args:
        days: 天数
        start_price: 起始价格

    Returns:
        DataFrame: 包含OHLCV数据
    """
    np.random.seed(42)

    # 生成价格序列（几何布朗运动）
    returns = np.random.normal(0.0005, 0.02, days)
    prices = start_price * (1 + returns).cumprod()

    # 生成日期
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        periods=days,
        freq='D'
    )

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
            'volume': np.random.randint(1000000, 10000000)
        })

    df = pd.DataFrame(data)
    return df


def example_create_environment():
    """示例1：创建RL环境"""
    print("\n" + "="*70)
    print("示例1：创建RL交易环境")
    print("="*70)

    if not SB3_AVAILABLE:
        print("\n无法运行：需要安装 stable-baselines3")
        print("安装命令: pip install stable-baselines3")
        return None

    # 生成训练数据
    print("\n生成训练数据...")
    data = generate_sample_data(days=500, start_price=100.0)
    print(f"数据生成完成: {len(data)} 条记录")

    # 创建环境
    print("\n创建RL交易环境...")
    env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=60,
    )

    print(f"动作空间: {env.action_space}")
    print(f"观察空间: {env.observation_space}")
    print(f"最大步数: {env.max_steps}")

    # 测试环境
    print("\n测试环境...")
    obs, info = env.reset()
    print(f"初始观察形状: {obs.shape}")

    # 随机执行几步
    print("\n随机执行5步...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  步骤 {i+1}: 动作={action}, 奖励={reward:.4f}")

    env.close()

    return env


def example_train_ppo():
    """示例2：训练PPO策略"""
    print("\n" + "="*70)
    print("示例2：训练PPO策略")
    print("="*70)

    if not SB3_AVAILABLE:
        print("\n无法运行：需要安装 stable-baselines3")
        return None

    # 准备数据
    print("\n准备训练数据...")
    data = generate_sample_data(days=500, start_price=100.0)

    # 创建环境
    print("创建训练环境...")
    env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=60,
    )

    # 创建PPO模型
    print("\n初始化PPO模型...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    # 训练模型（使用较少的步数以加快演示）
    print("\n开始训练...")
    print("注意：实际应用中应该训练更多步数（如100000+）")
    model.learn(total_timesteps=5000)
    print("训练完成！")

    # 测试模型
    print("\n测试训练好的模型...")
    obs, info = env.reset()
    total_reward = 0

    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"测试总奖励: {total_reward:.2f}")
    print(f"最终资产: ¥{env.cash + env.position_size * env.data.iloc[env.current_step + env.window_size]['close']:,.2f}")

    env.close()

    return model


def example_train_dqn():
    """示例3：训练DQN策略"""
    print("\n" + "="*70)
    print("示例3：训练DQN策略")
    print("="*70)

    if not SB3_AVAILABLE:
        print("\n无法运行：需要安装 stable-baselines3")
        return None

    # 准备数据
    print("\n准备训练数据...")
    data = generate_sample_data(days=500, start_price=100.0)

    # 创建环境
    print("创建训练环境...")
    env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=60,
    )

    # 创建DQN模型
    print("\n初始化DQN模型...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
    )

    # 训练模型
    print("\n开始训练...")
    model.learn(total_timesteps=5000)
    print("训练完成！")

    # 测试模型
    print("\n测试训练好的模型...")
    obs, info = env.reset()
    total_reward = 0

    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"测试总奖励: {total_reward:.2f}")

    env.close()

    return model


def example_compare_algorithms():
    """示例4：对比不同RL算法"""
    print("\n" + "="*70)
    print("示例4：对比不同RL算法")
    print("="*70)

    if not SB3_AVAILABLE:
        print("\n无法运行：需要安装 stable-baselines3")
        return None

    # 准备数据
    print("\n准备数据...")
    data = generate_sample_data(days=300, start_price=100.0)

    algorithms = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
    }

    results = {}

    for name, Algorithm in algorithms.items():
        print(f"\n{'='*70}")
        print(f"训练 {name} 算法")
        print(f"{'='*70}")

        # 创建环境
        env = ASharesTradingEnv(
            data=data,
            initial_cash=1000000.0,
            commission_rate=0.0003,
            window_size=60,
        )

        # 创建模型
        model = Algorithm("MlpPolicy", env, verbose=0)

        # 训练
        print(f"训练 {name}...")
        model.learn(total_timesteps=3000)

        # 测试
        print(f"测试 {name}...")
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        for i in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        results[name] = {
            'total_reward': total_reward,
            'steps': steps,
            'avg_reward': total_reward / steps if steps > 0 else 0,
        }

        print(f"总奖励: {total_reward:.2f}")
        print(f"平均奖励: {results[name]['avg_reward']:.4f}")

        env.close()

    # 对比结果
    print(f"\n{'='*70}")
    print("算法对比结果")
    print(f"{'='*70}")

    results_df = pd.DataFrame(results).T
    print(results_df.to_string())

    # 找出最佳算法
    best = results_df['total_reward'].idxmax()
    print(f"\n最佳算法: {best} (奖励: {results_df.loc[best, 'total_reward']:.2f})")

    return results_df


def example_custom_reward_function():
    """示例5：自定义奖励函数"""
    print("\n" + "="*70)
    print("示例5：自定义奖励函数")
    print("="*70)

    if not SB3_AVAILABLE:
        print("\n无法运行：需要安装 stable-baselines3")
        return None

    # 这里展示如何自定义奖励函数
    # 实际实现需要修改 ASharesTradingEnv 类

    print("""
自定义奖励函数的方法：

1. 修改 ASharesTradingEnv 的 _calculate_reward 方法

    def _calculate_reward(self, action: int, current_price: float):
        # 基础收益奖励
        reward = self.current_value - self.previous_value

        # 添加风险惩罚
        if self.position_size > 0:
            volatility = self._calculate_volatility()
            reward -= 0.1 * volatility  # 惩罚高波动

        # 添加交易成本惩罚
        if action in [1, 2]:  # 买入或卖出
            reward -= 100  # 固定交易成本

        # 添加持仓时间奖励
        if action == 0 and self.position_size > 0:  # 持有
            reward += 10  # 鼓励长期持有

        return reward

2. 或者创建自定义环境继承ASharesTradingEnv

    class CustomTradingEnv(ASharesTradingEnv):
        def _calculate_reward(self, action, current_price):
            # 自定义奖励逻辑
            reward = super()._calculate_reward(action, current_price)

            # 添加夏普比率奖励
            if self.trade_history:
                returns = [t['return'] for t in self.trade_history]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                reward += sharpe * 10

            return reward

3. 使用自定义环境
    env = CustomTradingEnv(data=data)
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=10000)
    """)


def example_train_with_real_data():
    """示例6：使用真实数据训练"""
    print("\n" + "="*70)
    print("示例6：使用真实数据训练")
    print("="*70)

    print("""
使用真实数据训练的步骤：

1. 获取真实数据
    from data.market.sources.akshare_provider import AKShareProvider

    provider = AKShareProvider()
    provider.connect()

    df = provider.get_daily_bar(
        symbol="000001.SZ",
        start_date="20230101",
        end_date="20241231",
        adjust="qfq"
    )

    provider.disconnect()

2. 准备数据
    # 重命名列以匹配环境要求
    df = df.rename(columns={'date': 'datetime'})
    df['datetime'] = pd.to_datetime(df['datetime'])

3. 创建环境并训练
    env = ASharesTradingEnv(
        data=df,
        initial_cash=1000000.0,
        window_size=60,
    )

    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=50000)

4. 保存模型
    model.save("models/ppo_trading_agent")

5. 加载并使用模型
    model = PPO.load("models/ppo_trading_agent")
    obs, info = env.reset()
    action, _states = model.predict(obs)
    """)


def example_hyperparameter_tuning():
    """示例7：超参数调优"""
    print("\n" + "="*70)
    print("示例7：超参数调优")
    print("="*70)

    print("""
超参数调优的方法：

1. 网格搜索
    from itertools import product

    learning_rates = [1e-4, 3e-4, 1e-3]
    n_steps_list = [1024, 2048, 4096]

    best_score = -np.inf
    best_params = None

    for lr, n_steps in product(learning_rates, n_steps_list):
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            verbose=0,
        )

        model.learn(total_timesteps=5000)
        score = evaluate_model(model, env)

        if score > best_score:
            best_score = score
            best_params = {'lr': lr, 'n_steps': n_steps}

    print(f"最佳参数: {best_params}")
    print(f"最佳分数: {best_score}")

2. 使用Optuna进行贝叶斯优化

    import optuna

    def objective(trial):
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=n_steps,
            verbose=0,
        )

        model.learn(total_timesteps=5000)
        score = evaluate_model(model, env)

        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    print(f"最佳参数: {study.best_params}")
    print(f"最佳分数: {study.best_value}")

3. 评估函数
    def evaluate_model(model, env, n_episodes=10):
        total_rewards = []

        for _ in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0

            while True:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)
    """)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 强化学习策略示例")
    print("="*70)

    try:
        # 示例1：创建环境
        # example_create_environment()

        # 示例2：训练PPO
        # example_train_ppo()

        # 示例3：训练DQN
        # example_train_dqn()

        # 示例4：对比算法
        # example_compare_algorithms()

        # 示例5-7：概念说明
        example_custom_reward_function()
        example_train_with_real_data()
        example_hyperparameter_tuning()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
