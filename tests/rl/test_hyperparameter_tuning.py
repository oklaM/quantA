"""
强化学习超参数调优模块测试
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

try:
    import optuna
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    OPTUNA_AVAILABLE = True
    SB3_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    SB3_AVAILABLE = False

from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.optimization import RLHyperparameterTuner, TrialEvalCallback, TuningResult


@pytest.fixture
def sample_env_data():
    """生成示例环境数据"""
    np.random.seed(42)
    n = 200
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * (1 + returns).cumprod()

    dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq="D")

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.015)))
        low = close * (1 - abs(np.random.normal(0, 0.015)))
        open_price = close * (1 + np.random.normal(0, 0.008))

        data.append(
            {
                "datetime": date,
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": np.random.randint(1000000, 10000000),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_envs(sample_env_data):
    """创建训练和评估环境"""
    train_env = ASharesTradingEnv(
        data=sample_env_data,
        initial_cash=1000000.0,
        window_size=20,
    )

    eval_env = ASharesTradingEnv(
        data=sample_env_data,
        initial_cash=1000000.0,
        window_size=20,
    )

    return train_env, eval_env


@pytest.mark.rl
class TestTuningResult:
    """测试调优结果"""

    def test_tuning_result_creation(self):
        """测试创建调优结果"""
        result = TuningResult(
            trial_number=1,
            params={"learning_rate": 0.001, "n_steps": 2048},
            mean_reward=100.0,
            std_reward=20.0,
            best_mean_reward=150.0,
        )

        assert result.trial_number == 1
        assert result.params["learning_rate"] == 0.001
        assert result.mean_reward == 100.0
        assert isinstance(result.datetime, str)


@pytest.mark.rl
class TestTrialEvalCallback:
    """测试试验评估回调"""

    def test_callback_initialization(self, sample_envs):
        """测试回调初始化"""
        train_env, eval_env = sample_envs

        callback = TrialEvalCallback(
            eval_env=eval_env,
            n_eval_episodes=2,
            eval_freq=100,
        )

        assert callback.eval_env == eval_env
        assert callback.n_eval_episodes == 2
        assert callback.eval_freq == 100
        assert callback.best_mean_reward == -np.inf

    def test_callback_on_step(self, sample_envs):
        """测试回调步进"""
        if not SB3_AVAILABLE:
            pytest.skip("stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        # 创建简单模型
        model = PPO("MlpPolicy", train_env, verbose=0)

        # 创建回调
        callback = TrialEvalCallback(
            eval_env=eval_env,
            n_eval_episodes=2,
            eval_freq=50,
        )

        # 设置parent（模拟）
        callback.parent = type("obj", (object,), {"report": lambda self, value, step: None})()

        # 运行几步
        for _ in range(100):
            callback.on_step()

        # 检查最佳奖励已更新
        assert callback.best_mean_reward > -np.inf


@pytest.mark.rl
class TestRLHyperparameterTuner:
    """测试RL超参数调优器"""

    @pytest.mark.skip(reason="需要安装optuna: pip install optuna")
    def test_tuner_initialization(self, sample_envs):
        """测试调优器初始化"""
        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
            n_trials=5,
            total_timesteps=100,
        )

        assert tuner.env == train_env
        assert tuner.eval_env == eval_env
        assert tuner.algorithm == "ppo"
        assert tuner.n_trials == 5

    def test_tuner_requires_optuna(self, sample_envs):
        """测试调优器需要optuna"""
        if OPTUNA_AVAILABLE:
            pytest.skip("optuna is installed")

        train_env, eval_env = sample_envs

        with pytest.raises(ImportError):
            tuner = RLHyperparameterTuner(
                env=train_env,
                eval_env=eval_env,
                algorithm="ppo",
            )

    def test_get_algorithm_class(self, sample_envs):
        """测试获取算法类"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
        )

        algo_class = tuner._get_algorithm_class()
        assert algo_class == PPO

    def test_get_invalid_algorithm(self, sample_envs):
        """测试获取无效算法"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="invalid",
        )

        with pytest.raises(ValueError):
            tuner._get_algorithm_class()

    def test_sample_ppo_hyperparameters(self, sample_envs):
        """测试采样PPO超参数"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
        )

        # 创建模拟trial
        study = optuna.create_study()
        trial = study.ask()

        # 采样超参数
        params = tuner._sample_hyperparameters(trial)

        # 检查参数
        assert "learning_rate" in params
        assert "n_steps" in params
        assert "batch_size" in params
        assert "gamma" in params
        assert isinstance(params["learning_rate"], float)
        assert isinstance(params["n_steps"], int)

    def test_sample_dqn_hyperparameters(self, sample_envs):
        """测试采样DQN超参数"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="dqn",
        )

        # 创建模拟trial
        study = optuna.create_study()
        trial = study.ask()

        # 采样超参数
        params = tuner._sample_hyperparameters(trial)

        # 检查参数
        assert "learning_rate" in params
        assert "buffer_size" in params
        assert "batch_size" in params
        assert "exploration_fraction" in params

    def test_optimize_small(self, sample_envs):
        """测试小规模优化"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
            n_trials=2,
            total_timesteps=100,
            eval_freq=50,
            n_eval_episodes=2,
        )

        # 运行优化
        results = tuner.optimize()

        # 检查结果
        assert "best_params" in results
        assert "best_value" in results
        assert "n_trials" in results
        assert len(tuner.results) <= 2

    def test_optimize_with_study_name(self, sample_envs):
        """测试使用指定study名称优化"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
            n_trials=2,
            total_timesteps=100,
        )

        # 使用指定study名称
        results = tuner.optimize(study_name="test_study")

        assert results["study_name"] == "test_study"

    def test_optimize_different_samplers(self, sample_envs):
        """测试使用不同采样器"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuners = [
            RLHyperparameterTuner(
                env=train_env,
                eval_env=eval_env,
                algorithm="ppo",
                n_trials=2,
                total_timesteps=100,
            )
            for _ in ["tpe", "random"]
        ]

        results = []
        for i, sampler in enumerate(["tpe", "random"]):
            result = tuners[i].optimize(sampler=sampler)
            results.append(result)

        # 两个优化都应该成功
        for result in results:
            assert "best_params" in result
            assert "best_value" in result


@pytest.mark.rl
class TestTunerResults:
    """测试调优器结果处理"""

    def test_get_best_model_without_optimize(self, sample_envs):
        """测试在未优化时获取最佳模型"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
        )

        # 在未运行optimize时调用get_best_model
        with pytest.raises(ValueError):
            tuner.get_best_model()

    def test_save_results_without_optimize(self, sample_envs):
        """测试在未优化时保存结果"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
        )

        # 在未运行optimize时调用save_results
        with pytest.raises(ValueError):
            tuner.save_results("/tmp/test.json")

    def test_plot_optimization_history_without_optimize(self, sample_envs):
        """测试在未优化时绘制优化历史"""
        if not OPTUNA_AVAILABLE or not SB3_AVAILABLE:
            pytest.skip("optuna or stable_baselines3 not installed")

        train_env, eval_env = sample_envs

        tuner = RLHyperparameterTuner(
            env=train_env,
            eval_env=eval_env,
            algorithm="ppo",
        )

        # 在未运行optimize时调用plot
        with pytest.raises(ValueError):
            tuner.plot_optimization_history()
