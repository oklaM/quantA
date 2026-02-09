"""
端到端集成测试
测试完整的用户工作流程
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.base.agent_base import AgentBase, AgentResponse
from agents.collaboration import AgentOrchestrator

# 需要测试的完整流程
from backtest.engine import BacktestEngine
from backtest.engine.analysis import PerformanceAnalyzer
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy
from backtest.optimization import GridSearchOptimizer
from monitoring import Alert, AlertLevel, AlertManager, AlertType
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer
from trading.risk import RiskController
from utils.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture
def sample_multi_stock_data():
    """生成多股票测试数据"""
    engine = BacktestEngine(initial_cash=1000000)
    return engine.generate_mock_data(
        symbols=["600000.SH", "000001.SZ", "600036.SH", "000002.SZ"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        freq="1d",
    )


@pytest.fixture
def sample_context():
    """示例交易上下文"""
    return {
        "account": {
            "total_asset": 1000000,
            "available_cash": 500000,
        },
        "positions": [
            {"symbol": "600000.SH", "quantity": 10000, "market_value": 100000},
            {"symbol": "000001.SZ", "quantity": 20000, "market_value": 200000},
        ],
        "daily_stats": {
            "initial_asset": 1000000,
            "traded_volume": 0,
            "daily_pnl": 0,
        },
    }


@pytest.mark.integration
class TestEndToEndBacktest:
    """测试端到端回测流程"""

    def test_complete_backtest_workflow(self, sample_multi_stock_data):
        """测试完整回测工作流：数据 → 策略 → 执行 → 分析"""
        # 1. 创建引擎
        engine = BacktestEngine(
            initial_cash=1000000,
            commission=0.0003,
            slippage=0.0001,
        )

        # 2. 运行策略
        strategy = BuyAndHoldStrategy()
        results = engine.run(strategy, sample_multi_stock_data)

        # 3. 验证结果
        assert "total_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "equity_curve" in results

        # 4. 性能分析
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.calculate_all_metrics()

        assert metrics["total_trades"] >= 0
        assert len(metrics["equity_curve"]) > 0

        # 5. 生成报告
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name
            analyzer.save_report(report_path)

        assert Path(report_path).exists()
        Path(report_path).unlink()  # 清理

    def test_multi_strategy_comparison(self, sample_multi_stock_data):
        """测试多策略对比"""
        engine = BacktestEngine(initial_cash=1000000)

        strategies = {
            "buy_and_hold": BuyAndHoldStrategy(),
            "ma_cross_5_20": MovingAverageCrossStrategy(5, 20),
            "ma_cross_10_30": MovingAverageCrossStrategy(10, 30),
        }

        results = {}
        for name, strategy in strategies.items():
            result = engine.run(strategy, sample_multi_stock_data)
            results[name] = result

        # 验证所有策略都有结果
        assert len(results) == 3

        # 验证结果可以比较
        returns = {name: r["total_return"] for name, r in results.items()}
        assert all(v is not None for v in returns.values())


@pytest.mark.integration
class TestEndToEndOptimization:
    """测试端到端优化流程"""

    def test_parameter_optimization_workflow(self, sample_multi_stock_data):
        """测试参数优化完整流程"""
        # 1. 创建引擎
        engine = BacktestEngine(initial_cash=1000000)

        # 2. 创建优化器
        optimizer = GridSearchOptimizer(
            engine=engine,
            strategy=MovingAverageCrossStrategy,
            param_grid={
                "short_window": [5, 10],
                "long_window": [20, 30],
            },
        )

        # 3. 运行优化
        best_params, all_results = optimizer.optimize(
            sample_multi_stock_data,
            metric="sharpe_ratio",
        )

        # 4. 验证结果
        assert "short_window" in best_params
        assert "long_window" in best_params
        assert len(all_results) == 4  # 2 * 2 组合

        # 5. 使用最优参数回测
        best_strategy = MovingAverageCrossStrategy(**best_params)
        best_results = engine.run(best_strategy, sample_multi_stock_data)

        assert best_results["sharpe_ratio"] > 0


@pytest.mark.integration
class TestEndToEndRL:
    """测试端到端强化学习流程"""

    def test_rl_training_workflow(self):
        """测试RL完整训练流程"""
        # 1. 生成训练数据
        engine = BacktestEngine(initial_cash=1000000)
        train_data = engine.generate_mock_data(
            symbols=["600000.SH", "000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-06-30",
        )

        # 2. 创建环境
        env = ASharesTradingEnv(
            df=train_data,
            initial_cash=100000,
            commission=0.0003,
        )

        # 3. 训练模型
        trainer = RLTrainer(
            env=env,
            algorithm="ppo",
            learning_rate=3e-4,
            n_steps=2048,
        )

        model = trainer.train(total_timesteps=1000)

        # 4. 评估模型
        eval_results = trainer.evaluate(model, n_episodes=5)

        assert "mean_reward" in eval_results
        assert "std_reward" in eval_results
        assert len(eval_results["rewards"]) == 5

    def test_rl_inference_workflow(self):
        """测试RL推理流程"""
        # 1. 准备环境
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )

        env = ASharesTradingEnv(df=data, initial_cash=100000)

        # 2. 快速训练
        trainer = RLTrainer(env=env, algorithm="ppo")
        model = trainer.train(total_timesteps=500)

        # 3. 使用模型预测
        obs, _ = env.reset()
        action, _ = model.predict(obs)

        # 4. 执行动作
        next_obs, reward, done, truncated, info = env.step(action)

        assert action in env.action_space
        assert isinstance(reward, (int, float))


@pytest.mark.integration
class TestEndToEndAgentCollaboration:
    """测试端到端智能体协作流程"""

    def test_agent_collaboration_workflow(self):
        """测试智能体协作完整流程"""

        # 1. 创建测试智能体
        class MockAgent(AgentBase):
            def __init__(self, name, response_text):
                super().__init__(name=name, role="test")
                self.response_text = response_text

            def process(self, input_data, context=None):
                return AgentResponse(
                    agent_id=self.agent_id,
                    content=self.response_text,
                    confidence=0.9,
                    metadata={"agent": self.name},
                )

        # 2. 创建智能体
        analyzer = MockAgent("analyzer", "市场分析：上涨趋势")
        strategist = MockAgent("strategist", "策略建议：买入")
        risk_manager = MockAgent("risk_manager", "风险评估：低风险")

        # 3. 创建协作器
        orchestrator = AgentOrchestrator()
        orchestrator.add_agent(analyzer)
        orchestrator.add_agent(strategist)
        orchestrator.add_agent(risk_manager)

        # 4. 运行协作流程
        input_data = {
            "symbol": "600000.SH",
            "date": "2023-01-01",
            "price": 10.50,
        }

        final_decision = orchestrator.collaborate(input_data)

        # 5. 验证结果
        assert final_decision is not None
        assert len(orchestrator.conversation_history) > 0

        # 验证所有智能体都参与了
        participants = set(msg["agent_id"] for msg in orchestrator.conversation_history)
        assert len(participants) == 3


@pytest.mark.integration
class TestEndToEndRiskControl:
    """测试端到端风控流程"""

    def test_risk_control_workflow(self, sample_context):
        """测试风控完整流程"""
        # 1. 创建风控控制器
        controller = RiskController(
            {
                "min_available_cash": 100000,
                "max_single_order_amount": 1000000,
                "max_daily_loss_ratio": 0.05,
            }
        )

        # 2. 测试正常订单
        allowed, rejects = controller.validate_order(
            symbol="600036.SH",
            action="buy",
            quantity=1000,
            price=10.0,
            context=sample_context,
        )

        assert allowed is True
        assert len(rejects) == 0

        # 3. 测试超额订单
        allowed, rejects = controller.validate_order(
            symbol="600036.SH",
            action="buy",
            quantity=200000,
            price=10.0,  # 200万，超过限额
            context=sample_context,
        )

        assert allowed is False
        assert len(rejects) > 0

        # 4. 获取统计信息
        stats = controller.get_statistics()
        assert stats["total_checks"] == 2
        assert stats["total_rejects"] == 1


@pytest.mark.integration
class TestEndToEndMonitoring:
    """测试端到端监控告警流程"""

    def test_monitoring_workflow(self):
        """测试监控告警完整流程"""
        # 1. 创建告警管理器
        alert_manager = AlertManager()
        alert_manager.start()

        # 2. 创建测试告警
        alert = Alert(
            alert_id="test_001",
            alert_type=AlertType.LOSS_LIMIT,
            level=AlertLevel.HIGH,
            title="测试告警",
            message="这是一条测试告警",
            metadata={"symbol": "600000.SH"},
        )

        # 3. 发送告警
        alert_manager.send_alert(alert)

        # 4. 检查告警历史
        alerts = alert_manager.get_alerts(limit=10)
        assert len(alerts) > 0

        # 5. 验证告警内容
        latest_alert = alerts[0]
        assert latest_alert.alert_id == "test_001"
        assert latest_alert.level == AlertLevel.HIGH

        # 6. 停止管理器
        alert_manager.stop()


@pytest.mark.integration
class TestEndToEndDataPipeline:
    """测试端到端数据流程"""

    def test_data_pipeline_workflow(self):
        """测试数据处理完整流程"""
        # 1. 生成原始数据
        engine = BacktestEngine(initial_cash=1000000)
        raw_data = engine.generate_mock_data(
            symbols=["600000.SH"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # 2. 数据预处理
        from utils.data import preprocess_data

        processed_data = preprocess_data(
            raw_data,
            fill_method="ffill",
            normalize=False,
        )

        assert processed_data.isnull().sum().sum() == 0

        # 3. 特征工程
        from utils.features import add_technical_features

        feature_data = add_technical_features(processed_data)

        # 验证特征已添加
        assert "returns" in feature_data.columns
        assert (
            "sma_20" in feature_data.columns or len(feature_data) < 20
        )  # 可能数据不足

        # 4. 数据分割
        train_size = int(len(feature_data) * 0.8)
        train_data = feature_data[:train_size]
        test_data = feature_data[train_size:]

        assert len(train_data) > 0
        assert len(test_data) > 0


@pytest.mark.integration
class TestEndToEndVisualization:
    """测试端到端可视化流程"""

    def test_visualization_workflow(self, sample_multi_stock_data):
        """测试可视化生成流程"""
        # 1. 运行回测
        engine = BacktestEngine(initial_cash=1000000)
        strategy = BuyAndHoldStrategy()
        results = engine.run(strategy, sample_multi_stock_data)

        # 2. 创建可视化器
        from backtest.engine.visualization import PerformanceVisualizer

        visualizer = PerformanceVisualizer(results)

        # 3. 生成图表
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            plot_path = f.name
            visualizer.plot_equity_curve(save_path=plot_path)

        assert Path(plot_path).exists()
        Path(plot_path).unlink()  # 清理

        # 4. 生成HTML报告
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            html_path = f.name
            visualizer.generate_html_report(html_path)

        assert Path(html_path).exists()
        Path(html_path).unlink()  # 清理


@pytest.mark.integration
class TestCompleteTradingWorkflow:
    """测试完整交易工作流"""

    def test_trading_system_workflow(self):
        """测试从数据到交易的完整流程"""
        # 1. 数据获取
        engine = BacktestEngine(initial_cash=1000000)
        data = engine.generate_mock_data(
            symbols=["600000.SH", "000001.SZ"],
            start_date="2023-01-01",
            end_date="2023-12-31",
        )

        # 2. 策略选择
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)

        # 3. 回测验证
        backtest_results = engine.run(strategy, data)
        assert backtest_results["total_return"] is not None

        # 4. 参数优化
        optimizer = GridSearchOptimizer(
            engine=engine,
            strategy=MovingAverageCrossStrategy,
            param_grid={
                "short_window": [5, 10],
                "long_window": [20, 30],
            },
        )

        best_params, _ = optimizer.optimize(data)
        optimized_strategy = MovingAverageCrossStrategy(**best_params)

        # 5. 优化后回测
        optimized_results = engine.run(optimized_strategy, data)
        assert (
            optimized_results["sharpe_ratio"] >= backtest_results["sharpe_ratio"] * 0.9
        )  # 允许一定误差

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

        logger.info(f"完整流程测试成功！总收益率: {metrics['total_return']:.2%}")
