"""
强化学习完整训练工作流示例

展示从数据准备到模型部署的完整流程：
1. 数据准备和预处理
2. 环境创建和配置
3. 模型训练
4. 模型评估
5. 模型保存和加载
6. 模型推理和部署
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("警告: stable_baselines3未安装，请运行: pip install stable-baselines3 gymnasium")

from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.evaluation.model_evaluator import ModelComparator, ModelEvaluator
from rl.rewards.reward_functions import create_reward_function
from rl.training.trainer import RLTrainer, create_trainer
from utils.logging import get_logger

logger = get_logger(__name__)


def prepare_real_data(symbol: str = "000001.SZ", days: int = 1000):
    """
    准备真实市场数据

    Args:
        symbol: 股票代码
        days: 数据天数

    Returns:
        DataFrame: 处理后的市场数据
    """
    print("\n" + "="*70)
    print("步骤 1: 数据准备")
    print("="*70)

    try:
        from data.market.sources.akshare_provider import AKShareProvider

        print(f"\n获取 {symbol} 的历史数据...")

        provider = AKShareProvider()
        provider.connect()

        # 获取数据
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days*2)).strftime("%Y%m%d")

        df = provider.get_daily_bar(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权
        )

        provider.disconnect()

        # 数据预处理
        df = df.rename(columns={'date': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])

        print(f"数据获取成功: {len(df)} 条记录")
        print(f"日期范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")

        return df

    except Exception as e:
        print(f"获取真实数据失败: {e}")
        print("\n使用模拟数据代替...")

        # 生成模拟数据
        return generate_mock_data(days=days)


def generate_mock_data(days: int = 1000, start_price: float = 100.0):
    """
    生成模拟数据（用于测试）

    Args:
        days: 天数
        start_price: 起始价格

    Returns:
        DataFrame: 模拟市场数据
    """
    print("\n生成模拟数据...")

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
    print(f"模拟数据生成完成: {len(df)} 条记录")

    return df


def create_trading_environment(data: pd.DataFrame, window_size: int = 60):
    """
    创建交易环境

    Args:
        data: 市场数据
        window_size: 观察窗口大小

    Returns:
        ASharesTradingEnv: 交易环境
    """
    print("\n" + "="*70)
    print("步骤 2: 创建交易环境")
    print("="*70)

    # 划分训练集和测试集
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size].reset_index(drop=True)
    test_data = data[train_size:].reset_index(drop=True)

    print(f"\n训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")

    # 创建环境
    env = ASharesTradingEnv(
        data=train_data,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=window_size,
    )

    print(f"\n环境配置:")
    print(f"  初始资金: ¥1,000,000")
    print(f"  佣金率: 0.03%")
    print(f"  观察窗口: {window_size} 天")
    print(f"  动作空间: {env.action_space} (0=持有, 1=买入, 2=卖出)")
    print(f"  观察空间: {env.observation_space.shape}")

    return env, test_data


def train_model(
    env,
    algorithm: str = "ppo",
    total_timesteps: int = 50000,
    model_save_path: str = "models/trading_agent"
):
    """
    训练RL模型

    Args:
        env: 训练环境
        algorithm: 算法名称 (ppo, dqn)
        total_timesteps: 总训练步数
        model_save_path: 模型保存路径

    Returns:
        训练好的模型
    """
    print("\n" + "="*70)
    print("步骤 3: 模型训练")
    print("="*70)

    # 创建训练器
    print(f"\n初始化 {algorithm.upper()} 训练器...")
    trainer = create_trainer(
        env=env,
        algorithm=algorithm,
        learning_rate=3e-4,
        tensorboard_log="./logs/rl_training/",
    )

    # 构建模型
    model = trainer.build_model(
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    print(f"\n开始训练 ({total_timesteps} 步)...")
    print("提示: 训练过程中可以使用 tensorboard 查看训练曲线")
    print("  tensorboard --logdir ./logs/rl_training/")

    start_time = datetime.now()

    # 训练模型
    model = trainer.train(
        total_timesteps=total_timesteps,
        save_freq=10000,
    )

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    print(f"\n训练完成!")
    print(f"训练时长: {training_duration/60:.2f} 分钟")

    # 保存模型
    print(f"\n保存模型到: {model_save_path}")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)

    # 保存训练配置
    config = {
        'algorithm': algorithm,
        'total_timesteps': total_timesteps,
        'training_duration_seconds': training_duration,
        'timestamp': datetime.now().isoformat(),
    }

    config_path = f"{model_save_path}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"训练配置已保存到: {config_path}")

    return model


def evaluate_model(model, env, test_data: pd.DataFrame, n_episodes: int = 10):
    """
    评估模型性能

    Args:
        model: 训练好的模型
        env: 环境
        test_data: 测试数据
        n_episodes: 评估回合数

    Returns:
        评估结果
    """
    print("\n" + "="*70)
    print("步骤 4: 模型评估")
    print("="*70)

    # 使用测试数据创建评估环境
    test_env = ASharesTradingEnv(
        data=test_data,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        window_size=60,
    )

    # 创建评估器
    evaluator = ModelEvaluator(env=test_env, n_episodes=n_episodes)

    # 评估模型
    print(f"\n评估模型 ({n_episodes} 个回合)...")
    results = evaluator.evaluate(
        model=model,
        model_name="Trading Agent",
        deterministic=True
    )

    print(f"\n评估结果:")
    print(f"  平均奖励: {results.mean_reward:.2f} +/- {results.std_reward:.2f}")
    print(f"  奖励范围: [{results.min_reward:.2f}, {results.max_reward:.2f}]")
    print(f"  中位数: {results.median_reward:.2f}")
    print(f"  平均回合长度: {results.mean_length:.2f}")

    # 额外指标
    if results.metrics:
        print(f"\n额外指标:")
        for key, value in results.metrics.items():
            print(f"  {key}: {value:.4f}")

    # 模拟交易并记录详细轨迹
    print(f"\n模拟交易轨迹 (第一个回合)...")
    obs, info = test_env.reset()
    trajectory = []

    for step in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        current_price = test_data.iloc[test_env.current_step + test_env.window_size]['close']
        total_value = info['total_value']

        trajectory.append({
            'step': step,
            'action': ['持有', '买入', '卖出'][action],
            'price': current_price,
            'total_value': total_value,
            'reward': reward,
        })

        if terminated or truncated:
            break

    # 显示交易统计
    trades = [t for t in trajectory if t['action'] in ['买入', '卖出']]
    print(f"\n交易统计:")
    print(f"  总步数: {len(trajectory)}")
    print(f"  交易次数: {len(trades)}")
    print(f"  最终资产: ¥{trajectory[-1]['total_value']:,.2f}")
    print(f"  收益率: {(trajectory[-1]['total_value'] - 1000000) / 1000000 * 100:.2f}%")

    return results


def load_and_predict(model_path: str, data: pd.DataFrame):
    """
    加载模型并进行预测

    Args:
        model_path: 模型路径
        data: 市场数据

    Returns:
        预测结果
    """
    print("\n" + "="*70)
    print("步骤 5: 模型推理")
    print("="*70)

    # 加载模型配置
    config_path = f"{model_path}_config.json"
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\n模型配置:")
        print(f"  算法: {config['algorithm']}")
        print(f"  训练时间: {config['timestamp']}")
    else:
        config = {'algorithm': 'ppo'}  # 默认

    # 加载模型
    print(f"\n加载模型: {model_path}")
    if config['algorithm'] == 'ppo':
        model = PPO.load(model_path)
    elif config['algorithm'] == 'dqn':
        model = DQN.load(model_path)
    else:
        raise ValueError(f"未知算法: {config['algorithm']}")

    print("模型加载成功!")

    # 创建环境
    env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        window_size=60,
    )

    # 运行预测
    print("\n运行模拟交易...")
    obs, info = env.reset()

    actions_taken = []

    for step in range(min(100, env.max_steps)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        actions_taken.append(int(action))

        if step % 10 == 0:
            current_price = data.iloc[env.current_step + env.window_size]['close']
            print(f"  步骤 {step}: {['持有', '买入', '卖出'][action]}, "
                  f"价格: ¥{current_price:.2f}, "
                  f"总资产: ¥{info['total_value']:,.2f}")

        if terminated or truncated:
            break

    # 统计
    from collections import Counter
    action_counts = Counter(actions_taken)
    print(f"\n动作分布:")
    print(f"  持有: {action_counts[0]} 次")
    print(f"  买入: {action_counts[1]} 次")
    print(f"  卖出: {action_counts[2]} 次")

    final_value = info['total_value']
    return_pct = (final_value - 1000000) / 1000000 * 100

    print(f"\n最终收益:")
    print(f"  最终资产: ¥{final_value:,.2f}")
    print(f"  收益率: {return_pct:+.2f}%")

    return model


def main():
    """主函数 - 完整训练工作流"""
    print("\n" + "="*70)
    print("quantA 强化学习完整训练工作流")
    print("="*70)

    if not SB3_AVAILABLE:
        print("\n错误: 需要安装 stable-baselines3 和 gymnasium")
        print("安装命令: pip install stable-baselines3 gymnasium")
        return

    # 配置
    SYMBOL = "000001.SZ"  # 平安银行
    DATA_DAYS = 1000
    WINDOW_SIZE = 60
    ALGORITHM = "ppo"  # 或 "dqn"
    TRAINING_STEPS = 10000  # 实际应用建议 50000+
    MODEL_PATH = "models/rl_trading_agent"
    N_EVAL_EPISODES = 10

    try:
        # 步骤 1: 数据准备
        data = prepare_real_data(symbol=SYMBOL, days=DATA_DAYS)

        # 步骤 2: 创建环境
        env, test_data = create_trading_environment(data, window_size=WINDOW_SIZE)

        # 步骤 3: 训练模型
        model = train_model(
            env=env,
            algorithm=ALGORITHM,
            total_timesteps=TRAINING_STEPS,
            model_save_path=MODEL_PATH
        )

        # 步骤 4: 评估模型
        evaluate_model(
            model=model,
            env=env,
            test_data=test_data,
            n_episodes=N_EVAL_EPISODES
        )

        # 步骤 5: 加载并使用模型
        loaded_model = load_and_predict(MODEL_PATH, test_data)

        print("\n" + "="*70)
        print("工作流完成!")
        print("="*70)
        print(f"\n模型已保存到: {MODEL_PATH}")
        print("可以使用加载的模型进行实盘交易模拟或进一步优化")

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
