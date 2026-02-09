"""
强化学习超参数调优示例
展示如何使用超参数优化框架进行RL模型调优
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import optuna
    from stable_baselines3 import PPO
    OPTUNA_AVAILABLE = True
    SB3_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    SB3_AVAILABLE = False
    print("警告: optuna或stable_baselines3未安装")

from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.optimization import RLHyperparameterTuner
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_sample_data(days: int = 500, start_price: float = 100.0):
    """
    生成示例数据

    Args:
        days: 天数
        start_price: 起始价格

    Returns:
        DataFrame: 包含OHLCV数据
    """
    np.random.seed(42)

    # 生成价格序列
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


def example_ppo_hyperparameter_tuning():
    """示例1：PPO超参数调优"""
    print("\n" + "="*70)
    print("示例1：PPO超参数调优")
    print("="*70)

    if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
        print("\n无法运行：需要安装 optuna 和 stable-baselines3")
        print("安装命令: pip install optuna stable-baselines3")
        return None

    # 准备数据
    print("\n生成训练数据...")
    data = generate_sample_data(days=500)

    # 创建环境和评估环境
    print("创建环境...")
    train_env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        window_size=60,
    )

    eval_env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        window_size=60,
    )

    # 创建调优器
    print("初始化超参数调优器...")
    tuner = RLHyperparameterTuner(
        env=train_env,
        eval_env=eval_env,
        algorithm='ppo',
        n_trials=10,  # 实际使用时建议50+
        n_startup_trials=3,
        n_eval_episodes=5,
        total_timesteps=5000,  # 实际使用时建议50000+
        eval_freq=2500,
    )

    # 执行调优
    print("\n开始超参数调优...")
    optimization_results = tuner.optimize(
        study_name='ppo_tuning_example',
        sampler='tpe',
        pruner='median',
    )

    print("\n" + "="*70)
    print("调优结果")
    print("="*70)
    print(f"最佳参数: {optimization_results['best_params']}")
    print(f"最佳平均奖励: {optimization_results['best_value']:.2f}")
    print(f"试验次数: {optimization_results['n_trials']}")

    # 使用最佳参数训练模型
    print("\n使用最佳参数训练模型...")
    best_model = tuner.get_best_model(save_path='models/best_ppo_model')

    return tuner, best_model


def example_dqn_hyperparameter_tuning():
    """示例2：DQN超参数调优"""
    print("\n" + "="*70)
    print("示例2：DQN超参数调优")
    print("="*70)

    if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
        print("\n无法运行：需要安装 optuna 和 stable-baselines3")
        return None

    # 准备数据
    print("\n生成训练数据...")
    data = generate_sample_data(days=500)

    # 创建环境
    print("创建环境...")
    train_env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        window_size=60,
    )

    eval_env = ASharesTradingEnv(
        data=data,
        initial_cash=1000000.0,
        window_size=60,
    )

    # 创建调优器
    print("初始化超参数调优器...")
    tuner = RLHyperparameterTuner(
        env=train_env,
        eval_env=eval_env,
        algorithm='dqn',
        n_trials=10,
        n_eval_episodes=5,
        total_timesteps=5000,
    )

    # 执行调优
    print("\n开始超参数调优...")
    optimization_results = tuner.optimize(
        study_name='dqn_tuning_example',
    )

    print("\n最佳参数:")
    for key, value in optimization_results['best_params'].items():
        print(f"  {key}: {value}")

    return tuner


def example_compare_samplers():
    """示例3：对比不同的采样器"""
    print("\n" + "="*70)
    print("示例3：对比不同的采样器")
    print("="*70)

    if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
        print("\n无法运行：需要安装 optuna 和 stable-baselines3")
        return None

    # 准备数据
    print("\n生成训练数据...")
    data = generate_sample_data(days=500)

    # 创建环境
    train_env = ASharesTradingEnv(data=data, initial_cash=1000000.0, window_size=60)
    eval_env = ASharesTradingEnv(data=data, initial_cash=1000000.0, window_size=60)

    samplers = ['tpe', 'random']
    results_comparison = {}

    for sampler in samplers:
        print(f"\n使用 {sampler.upper()} 采样器...")

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm='ppo',
            n_trials=5,
            total_timesteps=3000,
        )

        results = tuner.optimize(sampler=sampler)
        results_comparison[sampler] = results

        print(f"最佳奖励: {results['best_value']:.2f}")

    # 对比结果
    print("\n" + "="*70)
    print("采样器对比")
    print("="*70)

    for sampler, results in results_comparison.items():
        print(f"{sampler.upper()}: {results['best_value']:.2f}")

    return results_comparison


def example_save_and_load_study():
    """示例4：保存和加载study"""
    print("\n" + "="*70)
    print("示例4：保存和加载study")
    print("="*70)

    if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
        print("\n无法运行：需要安装 optuna 和 stable-baselines3")
        return None

    print("""
保存和加载Study的方法：

1. 使用数据库持久化
    # 创建study时指定storage
    tuner.optimize(
        study_name='my_study',
        storage='sqlite:///optuna_studies.db',
    )

    # 从数据库加载
    study = optuna.load_study(study_name='my_study', storage='sqlite:///optuna_studies.db')

2. 保存结果为JSON
    tuner.save_results('optimization_results.json')

3. 可视化优化历史
    tuner.plot_optimization_history(save_path='optimization_history.png')

    tuner.plot_param_importances(save_path='param_importances.png')

4. 使用Optuna Dashboard
    # 安装: pip install optuna-dashboard

    # 启动dashboard
    optuna-dashboard sqlite:///optuna_studies.db

    # 然后在浏览器中访问 http://localhost:8080
""")


def example_custom_hyperparameter_space():
    """示例5：自定义超参数空间"""
    print("\n" + "="*70)
    print("示例5：自定义超参数空间")
    print("="*70)

    print("""
自定义超参数空间的方法：

1. 修改 RLHyperparameterTuner._sample_hyperparameters 方法

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}

        if self.algorithm == "ppo":
            # 自定义参数空间
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            params['n_steps'] = trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192])

            # 添加自定义参数
            params['vf_coef'] = trial.suggest_float('vf_coef', 0.1, 0.9)
            params['max_grad_norm'] = trial.suggest_float('max_grad_norm', 0.3, 1.0)

        return params

2. 创建子类

    class CustomTuner(RLHyperparameterTuner):
        def _sample_hyperparameters(self, trial: optuna.Trial):
            params = super()._sample_hyperparameters(trial)

            # 添加额外的参数
            params['custom_param'] = trial.suggest_float('custom_param', 0.0, 1.0)

            return params

3. 使用自定义调优器
    tuner = CustomTuner(env, eval_env, algorithm='ppo')
    tuner.optimize()
""")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 强化学习超参数调优示例")
    print("="*70)

    try:
        # 示例1：PPO超参数调优
        # example_ppo_hyperparameter_tuning()

        # 示例2：DQN超参数调优
        # example_dqn_hyperparameter_tuning()

        # 示例3：对比不同采样器
        # example_compare_samplers()

        # 示例4-5：概念说明
        example_save_and_load_study()
        example_custom_hyperparameter_space()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
