"""
改进的端到端集成测试
测试完整的量化交易工作流
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.base.agent_base import Agent, AgentResponse, MessageType
from agents.collaboration import AgentOrchestrator
from backtest.engine import BacktestEngine
from backtest.engine.analysis import PerformanceAnalyzer
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy
from backtest.optimization import GridSearchOptimizer
from data.market.data_manager import DataManager
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer
from trading.risk import RiskController
from utils.logging import get_logger

logger = get_logger(__name__)


class MockAgent(Agent):
    """Mock Agent for testing"""

    def __init__(self, name, response_text):
        super().__init__(name=name, description="Mock agent")
        self.response_text = response_text

    def process(self, input_data, context=None):
        """Process input and return response"""
        return AgentResponse(
            agent_id=self.agent_id,
            content=self.response_text,
            confidence=0.9,
            metadata={"agent": self.name},
        )


class TestEndToEndIntegration:
    """端到端集成测试类"""

    def setup_method(self):
        """每个测试方法前执行"""
        # 创建模拟数据
        self.data = self._generate_mock_data()

        # 使用第一个股票创建策略
        first_symbol = list(self.data.keys())[0]
        self.strategy = BuyAndHoldStrategy(symbol=first_symbol)

        # 创建回测引擎
        self.engine = BacktestEngine(
            data=self.data,
            strategy=self.strategy,
            initial_cash=1000000,
            commission_rate=0.0003,
            slippage_rate=0.0001,
        )

    def _generate_mock_data(self):
        """生成模拟市场数据"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

        # 生成4只股票的模拟数据
        symbols = ["600000.SH", "000001.SZ", "600036.SH", "000002.SZ"]
        data = {}

        for symbol in symbols:
            # 生成随机价格序列
            base_price = np.random.uniform(10, 50)
            prices = []
            current_price = base_price

            for date in dates:
                # 随机游走
                change = np.random.normal(0, 0.02)
                current_price *= (1 + change)
                current_price = max(current_price, 1)  # 价格不能为负

                prices.append(current_price)

            # 创建DataFrame
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'volume': np.random.randint(10000, 100000, len(dates)),
                'amount': [p * v for p, v in zip(prices, np.random.randint(10000, 100000, len(dates)))],
            })

            # 设置索引
            df.index = df['date']
            df.index.name = 'date'

            data[symbol] = df

        return data

    def test_complete_backtest_workflow(self):
        """测试完整回测工作流"""
        print("\n=== 测试1: 完整回测工作流 ===")

        # 1. 使用已有的BuyAndHold策略
        results = self.engine.run()

        # 2. 验证结果
        assert "account" in results
        assert "performance" in results
        assert "equity_curve" in results
        assert "total_return_pct" in results["account"]
        assert "sharpe_ratio" in results["performance"]
        assert "max_drawdown" in results["performance"]

        print(f"总收益率: {results['account']['total_return_pct']:.2%}")
        print(f"夏普比率: {results['performance']['sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['performance']['max_drawdown']:.2%}")

        # 3. 性能分析
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.calculate_all_metrics(
            equity_curve=results['equity_curve']['total_value']
        )

        assert metrics["total_trades"] >= 0
        assert len(metrics["equity_curve"]) > 0

        # 4. 生成报告
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name
            analyzer.save_report(report_path)

        assert Path(report_path).exists()
        Path(report_path).unlink()

        print("✅ 回测工作流测试通过")

    def test_multi_strategy_comparison(self):
        """测试多策略对比"""
        print("\n=== 测试2: 多策略对比 ===")

        strategies = {
            "buy_and_hold": BuyAndHoldStrategy(symbol="600000.SH"),
            "ma_cross_5_20": MovingAverageCrossStrategy(symbol="600000.SH", fast_period=5, slow_period=20),
            "ma_cross_10_30": MovingAverageCrossStrategy(symbol="600000.SH", fast_period=10, slow_period=30),
        }

        results = {}
        for name, strategy in strategies.items():
            result = self.engine.run(strategy)
            results[name] = result

        # 验证所有策略都有结果
        assert len(results) == 3

        # 验证结果可以比较
        returns = {name: r["account"]["total_return_pct"] for name, r in results.items()}
        assert all(v is not None for v in returns.values())

        print("策略对比结果:")
        for name, ret in returns.items():
            print(f"  {name}: {ret:.2%}")

        print("✅ 多策略对比测试通过")

    def test_parameter_optimization(self):
        """测试参数优化"""
        print("\n=== 测试3: 参数优化 ===")

        # 创建优化器
        first_symbol = list(self.data.keys())[0]
        optimizer = GridSearchOptimizer(
            data=self.engine.data[first_symbol],
            strategy_class=MovingAverageCrossStrategy,
            param_space={
                "fast_period": [5, 10],
                "slow_period": [20, 30],
            },
        )

        # 运行优化
        best_result = optimizer.optimize(
            optimization_target="sharpe_ratio",
        )

        best_params = best_result.params

        # 验证结果
        assert "short_window" in best_params
        assert "long_window" in best_params
        assert len(all_results) == 4  # 2 * 2 组合

        # 使用最优参数回测
        best_strategy = MovingAverageCrossStrategy(symbol=first_symbol, **best_params)
        best_results = self.engine.run(best_strategy)

        assert best_results["performance"]["sharpe_ratio"] > 0

        print(f"最优参数: {best_params}")
        print(f"最优夏普比率: {best_results['performance']['sharpe_ratio']:.2f}")

        print("✅ 参数优化测试通过")

    def test_rl_environment(self):
        """测试RL环境"""
        print("\n=== 测试4: RL环境 ===")

        # 创建RL环境
        env = ASharesTradingEnv(
            df=self.engine.data["600000.SH"],
            initial_cash=100000,
            commission=0.0003,
        )

        # 测试环境重置
        obs, info = env.reset()
        assert obs is not None
        assert info is not None

        # 测试环境步进
        action = 0  # hold
        next_obs, reward, done, truncated, info = env.step(action)

        assert action in env.action_space
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)

        print(f"环境观察空间: {env.observation_space}")
        print(f"环境动作空间: {env.action_space}")
        print(f"奖励值: {reward:.2f}")

        print("✅ RL环境测试通过")

    def test_rl_training(self):
        """测试RL训练"""
        print("\n=== 测试5: RL训练 ===")

        # 创建环境
        env = ASharesTradingEnv(
            df=self.engine.data["600000.SH"],
            initial_cash=100000,
            commission=0.0003,
        )

        # 创建训练器
        trainer = RLTrainer(
            env=env,
            algorithm="ppo",
            learning_rate=3e-4,
            n_steps=2048,
        )

        # 快速训练
        model = trainer.train(total_timesteps=1000)
        assert model is not None

        # 评估模型
        eval_results = trainer.evaluate(model, n_episodes=3)
        assert "mean_reward" in eval_results
        assert "std_reward" in eval_results

        print(f"平均奖励: {eval_results['mean_reward']:.2f}")
        print(f"奖励标准差: {eval_results['std_reward']:.2f}")

        print("✅ RL训练测试通过")

    def test_agent_collaboration(self):
        """测试Agent协作"""
        print("\n=== 测试6: Agent协作 ===")

        # 创建测试Agent
        analyzer = MockAgent("analyzer", "市场分析：上涨趋势")
        strategist = MockAgent("strategist", "策略建议：买入")
        risk_manager = MockAgent("risk_manager", "风险评估：低风险")

        # 创建协作器
        orchestrator = AgentOrchestrator()
        orchestrator.add_agent(analyzer)
        orchestrator.add_agent(strategist)
        orchestrator.add_agent(risk_manager)

        # 运行协作流程
        input_data = {
            "symbol": "600000.SH",
            "date": "2023-01-01",
            "price": 10.50,
        }

        final_decision = orchestrator.collaborate(input_data)

        # 验证结果
        assert final_decision is not None
        assert len(orchestrator.conversation_history) > 0

        # 验证所有Agent都参与了
        participants = set(msg["agent_id"] for msg in orchestrator.conversation_history)
        assert len(participants) == 3

        print(f"最终决策: {final_decision}")
        print(f"对话历史长度: {len(orchestrator.conversation_history)}")

        print("✅ Agent协作测试通过")

    def test_risk_control(self):
        """测试风控系统"""
        print("\n=== 测试7: 风控系统 ===")

        # 创建风控控制器
        controller = RiskController({
            "min_available_cash": 100000,
            "max_single_order_amount": 1000000,
            "max_daily_loss_ratio": 0.05,
        })

        # 测试正常订单
        context = {
            "account": {
                "total_asset": 1000000,
                "available_cash": 500000,
            },
            "positions": [],
            "daily_stats": {
                "initial_asset": 1000000,
                "traded_volume": 0,
                "daily_pnl": 0,
            },
        }

        allowed, rejects = controller.validate_order(
            symbol="600036.SH",
            action="buy",
            quantity=1000,
            price=10.0,
            context=context,
        )

        assert allowed is True
        assert len(rejects) == 0

        # 测试超额订单
        allowed, rejects = controller.validate_order(
            symbol="600036.SH",
            action="buy",
            quantity=200000,  # 200万，超过限额
            price=10.0,
            context=context,
        )

        assert allowed is False
        assert len(rejects) > 0

        print("正常订单通过: ✅")
        print("超额订单被拒绝: ✅")

        # 获取统计信息
        stats = controller.get_statistics()
        assert stats["total_checks"] == 2
        assert stats["total_rejects"] == 1

        print("✅ 风控系统测试通过")

    def test_data_pipeline(self):
        """测试数据处理管道"""
        print("\n=== 测试8: 数据处理管道 ===")

        # 1. 测试数据管理器
        data_manager = DataManager()

        # 添加数据
        for symbol, df in self.engine.data.items():
            data_manager.add_market_data(symbol, df)

        # 验证数据存储
        assert data_manager.get_available_symbols() == list(self.engine.data.keys())

        # 2. 测试数据获取
        symbol_data = data_manager.get_market_data("600000.SH")
        assert symbol_data is not None
        assert len(symbol_data) > 0

        # 3. 测试数据预处理
        # 这里跳过数据预处理和特征工程测试，因为模块不存在
        processed_data = symbol_data.copy()

        # 生成一些简单特征
        processed_data["returns"] = processed_data["close"].pct_change()
        processed_data["sma_5"] = processed_data["close"].rolling(5).mean()
        processed_data["sma_20"] = processed_data["close"].rolling(20).mean()

        assert processed_data.isnull().sum().sum() == 0

        # 验证特征已添加
        assert "returns" in feature_data.columns

        print(f"可用符号: {data_manager.get_available_symbols()}")
        print(f"特征数量: {len(feature_data.columns)}")

        print("✅ 数据处理管道测试通过")

    def test_trading_workflow(self):
        """测试完整交易工作流"""
        print("\n=== 测试9: 完整交易工作流 ===")

        # 1. 数据获取
        assert len(self.engine.data) == 4
        print(f"数据已获取: {len(self.engine.data)} 只股票")

        # 2. 策略选择
        strategy = MovingAverageCrossStrategy(symbol="600000.SH", fast_period=5, slow_period=20)

        # 3. 回测验证
        backtest_results = self.engine.run()
        assert backtest_results["account"]["total_return_pct"] is not None

        print(f"初始策略收益: {backtest_results['account']['total_return_pct']:.2%}")

        # 4. 参数优化
        first_symbol = list(self.data.keys())[0]
        optimizer = GridSearchOptimizer(
            data=self.engine.data[first_symbol],
            strategy_class=MovingAverageCrossStrategy,
            param_space={
                "fast_period": [5, 10],
                "slow_period": [20, 30],
            },
        )

        best_result = optimizer.optimize(optimization_target="sharpe_ratio")
        best_params = best_result.params
        optimized_strategy = MovingAverageCrossStrategy(symbol=first_symbol, **best_params)

        # 5. 优化后回测
        optimized_results = self.engine.run(optimized_strategy)
        assert optimized_results["performance"]["sharpe_ratio"] >= backtest_results["performance"]["sharpe_ratio"] * 0.9

        print(f"优化后收益: {optimized_results['account']['total_return_pct']:.2%}")
        print(f"优化后夏普比率: {optimized_results['performance']['sharpe_ratio']:.2f}")

        # 6. 风控验证
        controller = RiskController({"max_single_order_amount": 1000000})

        context = {
            "account": {"total_asset": 1000000, "available_cash": 500000},
            "positions": [],
            "daily_stats": {
                "initial_asset": 1000000,
                "traded_volume": 0,
                "daily_pnl": 0,
            },
        }

        allowed, _ = controller.validate_order(
            symbol="600000.SH",
            action="buy",
            quantity=1000,
            price=10.0,
            context=context,
        )

        assert allowed is True

        # 7. 性能分析
        analyzer = PerformanceAnalyzer(optimized_results)
        metrics = analyzer.calculate_all_metrics()

        assert metrics["total_return"] is not None
        assert metrics["sharpe_ratio"] is not None

        print(f"最终总收益率: {metrics['total_return']:.2%}")
        print(f"最终夏普比率: {metrics['sharpe_ratio']:.2f}")

        logger.info(f"完整交易工作流测试成功！总收益率: {metrics['total_return']:.2%}")

        print("✅ 完整交易工作流测试通过")


if __name__ == "__main__":
    # 运行所有测试
    test = TestEndToEndIntegration()
    test.setup_method()

    # 执行所有测试
    test.test_complete_backtest_workflow()
    test.test_multi_strategy_comparison()
    test.test_parameter_optimization()
    test.test_rl_environment()
    test.test_rl_training()
    test.test_agent_collaboration()
    test.test_risk_control()
    test.test_data_pipeline()
    test.test_trading_workflow()

    print("\n" + "="*60)
    print("🎉 所有端到端测试完成！")
    print("="*60)