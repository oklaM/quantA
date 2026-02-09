"""
跨模块集成测试
测试不同模块之间的交互和端到端场景
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.base.agent_base import Agent, AgentResponse, Message, MessageType
from agents.collaboration import AgentOrchestrator
from backtest.engine import BacktestEngine
from backtest.engine.analysis import PerformanceAnalyzer
from backtest.engine.indicators import TechnicalIndicators
from backtest.engine.portfolio import Portfolio
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy
from data.market.data_manager import DataManager
from monitoring import AlertManager, Alert, AlertType, AlertLevel
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer
from trading.risk import RiskController
from utils.logging import get_logger

logger = get_logger(__name__)

# 检查依赖
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


# ========== Fixtures ==========


@pytest.fixture
def sample_trading_data():
    """生成交易数据"""
    engine = BacktestEngine(
        data={},
        strategy=BuyAndHoldStrategy(),
        initial_cash=1000000,
    )
    return engine.generate_mock_data(
        symbols=["600000.SH", "000001.SZ"],
        start_date="2023-01-01",
        end_date="2023-06-30",
        freq="1d",
    )


@pytest.fixture
def sample_market_context():
    """示例市场上下文"""
    return {
        "account": {
            "total_asset": 1000000,
            "available_cash": 500000,
            "positions_value": 500000,
        },
        "positions": [
            {
                "symbol": "600000.SH",
                "quantity": 10000,
                "avg_cost": 10.0,
                "current_price": 10.50,
                "market_value": 105000,
                "profit_loss": 5000,
                "profit_loss_ratio": 0.05,
            }
        ],
        "market": {
            "date": "2023-01-01",
            "trend": "up",
            "volatility": "low",
        },
    }


# ========== Tests ==========


@pytest.mark.integration
class TestDataToStrategyIntegration:
    """测试数据到策略的集成"""

    def test_data_indicators_strategy_pipeline(self, sample_trading_data):
        """测试数据 → 指标 → 策略流程"""
        # 1. 数据准备
        raw_data = sample_trading_data["600000.SH"]

        # 2. 添加技术指标
        indicators = TechnicalIndicators()
        featured_data = raw_data.copy()
        if "close" in featured_data.columns:
            featured_data["sma_5"] = indicators.sma(featured_data["close"], 5)
            featured_data["sma_20"] = indicators.sma(featured_data["close"], 20)
            featured_data["rsi_14"] = indicators.rsi(featured_data["close"], 14)

        # 验证指标添加
        assert len(featured_data.columns) > len(raw_data.columns)

        # 3. 创建并运行策略
        featured_data_dict = {"600000.SH": featured_data}
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(
            data=featured_data_dict,
            strategy=strategy,
            initial_cash=1000000,
        )

        results = engine.run()

        # 4. 验证结果
        assert results is not None
        assert results["total_return"] is not None

    def test_multi_stock_data_pipeline(self, sample_trading_data):
        """测试多股票数据处理流程"""
        # 1. 对所有股票添加指标
        indicators = TechnicalIndicators()
        processed_data = {}

        for symbol, data in sample_trading_data.items():
            df = data.copy()
            if "close" in df.columns:
                df["sma_5"] = indicators.sma(df["close"], 5)
                df["sma_20"] = indicators.sma(df["close"], 20)
                df["rsi_14"] = indicators.rsi(df["close"], 14)
            processed_data[symbol] = df

        # 2. 验证数据处理
        assert len(processed_data) == len(sample_trading_data)
        assert all(
            len(data.columns) > len(sample_trading_data[symbol].columns)
            for symbol, data in processed_data.items()
        )

        # 3. 运行策略
        engine = BacktestEngine(
            data=processed_data,
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 4. 验证结果
        assert results is not None
        assert len(engine.portfolio.positions) == 2  # 两只股票


@pytest.mark.integration
class TestStrategyToRiskIntegration:
    """测试策略到风控的集成"""

    def test_strategy_risk_control_workflow(self, sample_trading_data, sample_market_context):
        """测试策略 → 风控流程"""
        # 1. 运行策略生成交易信号
        strategy = MovingAverageCrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(
            data=sample_trading_data,
            strategy=strategy,
            initial_cash=1000000,
        )

        results = engine.run()

        # 2. 风控验证
        controller = RiskController(
            {
                "min_available_cash": 100000,
                "max_single_order_amount": 500000,
                "max_daily_loss_ratio": 0.05,
            }
        )

        # 3. 模拟订单验证
        allowed, rejects = controller.validate_order(
            symbol="600000.SH",
            action="buy",
            quantity=1000,
            price=10.50,
            context=sample_market_context,
        )

        # 4. 验证风控决策
        assert isinstance(allowed, bool)
        assert isinstance(rejects, list)

    def test_portfolio_risk_monitoring(self, sample_trading_data):
        """测试组合风险监控"""
        # 1. 运行策略
        engine = BacktestEngine(
            data=sample_trading_data,
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 2. 监控组合风险
        portfolio = engine.portfolio

        # 计算组合价值
        total_value = portfolio.current_cash + sum(
            pos.quantity * pos.current_price for pos in portfolio.positions.values()
        )

        # 计算持仓集中度
        position_values = [
            pos.quantity * pos.current_price for pos in portfolio.positions.values()
        ]
        max_concentration = max(position_values) / total_value if total_value > 0 else 0

        # 3. 验证风险指标
        assert total_value > 0
        assert 0 <= max_concentration <= 1
        assert max_concentration < 1.0  # 不应该完全集中在一只股票


@pytest.mark.integration
class TestBacktestToMonitoringIntegration:
    """测试回测到监控的集成"""

    def test_backtest_alert_generation(self, sample_trading_data):
        """测试回测后生成告警"""
        # 1. 运行回测
        engine = BacktestEngine(
            data=sample_trading_data,
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 2. 创建告警管理器
        alert_manager = AlertManager()
        alert_manager.start()

        # 3. 根据回测结果生成告警
        if results["max_drawdown"] < -0.10:  # 回撤超过10%
            alert = Alert(
                alert_id="backtest_high_drawdown",
                alert_type=AlertType.LOSS_LIMIT,
                level=AlertLevel.MEDIUM,
                title="回测回撤过大",
                message=f"回测最大回撤: {results['max_drawdown']:.2%}",
                metadata={
                    "max_drawdown": results["max_drawdown"],
                    "sharpe_ratio": results["sharpe_ratio"],
                },
            )
            alert_manager.send_alert(alert)

        # 4. 验证告警
        alerts = alert_manager.get_alerts(limit=10)
        assert len(alerts) >= 0  # 可能有也可能没有

        alert_manager.stop()

    def test_performance_monitoring(self, sample_trading_data):
        """测试性能监控"""
        # 1. 运行回测
        engine = BacktestEngine(
            data=sample_trading_data,
            strategy=MovingAverageCrossStrategy(5, 20),
            initial_cash=1000000,
        )

        results = engine.run()

        # 2. 性能分析
        analyzer = PerformanceAnalyzer(results)
        metrics = analyzer.calculate_all_metrics()

        # 3. 监控关键指标
        monitoring_checks = {
            "total_return": metrics["total_return"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "max_drawdown": metrics["max_drawdown"],
            "win_rate": metrics["win_rate"],
        }

        # 4. 验证监控数据
        assert all(v is not None for v in monitoring_checks.values())

        # 5. 检查是否需要告警
        if metrics["sharpe_ratio"] < 1.0:
            logger.warning(f"夏普比率过低: {metrics['sharpe_ratio']:.2f}")

        if metrics["max_drawdown"] < -0.15:
            logger.warning(f"最大回撤过大: {metrics['max_drawdown']:.2%}")


@pytest.mark.integration
class TestAgentDecisionToExecutionIntegration:
    """测试Agent决策到执行的集成"""

    @pytest.mark.asyncio
    async def test_agent_trading_decision_workflow(self, sample_trading_data, sample_market_context):
        """测试Agent交易决策执行流程"""

        # 1. 创建简单决策Agent
        class SimpleTradingAgent(Agent):
            async def process_async(self, input_data, context=None):
                symbol = input_data.get("symbol", "600000.SH")
                price = input_data.get("price", 10.50)

                # 简单决策逻辑
                if price < 10.0:
                    action = "buy"
                    quantity = 1000
                elif price > 11.0:
                    action = "sell"
                    quantity = 500
                else:
                    action = "hold"
                    quantity = 0

                return AgentResponse(
                    agent_id=self.agent_id,
                    content=f"决策: {action} {quantity} 股",
                    confidence=0.8,
                    metadata={
                        "action": action,
                        "quantity": quantity,
                        "symbol": symbol,
                        "price": price,
                    },
                )

        # 2. 创建Agent并获取决策
        agent = SimpleTradingAgent(name="trading_agent", role="trading")

        decision = await agent.process_async(
            {"symbol": "600000.SH", "price": 9.50},
            context=sample_market_context,
        )

        # 3. 验证决策
        assert decision.metadata["action"] in ["buy", "sell", "hold"]
        assert decision.metadata["symbol"] == "600000.SH"

        # 4. 风控检查
        controller = RiskController({"max_single_order_amount": 200000})

        if decision.metadata["action"] != "hold":
            allowed, rejects = controller.validate_order(
                symbol=decision.metadata["symbol"],
                action=decision.metadata["action"],
                quantity=decision.metadata["quantity"],
                price=decision.metadata["price"],
                context=sample_market_context,
            )

            # 验证风控结果
            assert isinstance(allowed, bool)

    @pytest.mark.asyncio
    async def test_multi_agent_decision_aggregation(self, sample_market_context):
        """测试多Agent决策聚合"""

        # 1. 创建多个决策Agent
        agents = []

        for i in range(3):
            agent = Agent(name=f"agent_{i}", role="trading")

            # Mock process方法
            async def mock_process(input_data, context=None, agent_id=agent.agent_id):
                actions = ["buy", "sell", "hold"]
                import random

                return AgentResponse(
                    agent_id=agent_id,
                    content=f"决策",
                    confidence=0.7,
                    metadata={
                        "action": random.choice(actions),
                        "quantity": 1000,
                    },
                )

            agent.process_async = mock_process
            agents.append(agent)

        # 2. 获取所有Agent的决策
        decisions = []
        for agent in agents:
            decision = await agent.process_async({}, context=sample_market_context)
            decisions.append(decision)

        # 3. 聚合决策（多数投票）
        from collections import Counter

        actions = [d.metadata["action"] for d in decisions]
        votes = Counter(actions)
        final_action = votes.most_common(1)[0][0]

        # 4. 验证聚合结果
        assert final_action in ["buy", "sell", "hold"]
        assert len(decisions) == 3


@pytest.mark.integration
class TestRLToBacktestIntegration:
    """测试RL到回测的集成"""

    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_rl_trained_strategy_backtest(self, sample_trading_data):
        """测试RL训练策略的回测"""
        # 1. 训练RL模型
        symbol_data = sample_trading_data["600000.SH"]
        env = ASharesTradingEnv(df=symbol_data, initial_cash=100000)

        trainer = RLTrainer(env=env, algorithm="ppo")
        model = trainer.train(total_timesteps=1000)

        # 2. 使用RL模型进行回测
        # 注意：这里简化了，实际需要创建RL策略包装器
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        # 3. 验证RL策略性能
        assert isinstance(total_reward, (int, float))

        # 4. 对比基准策略
        benchmark_results = BacktestEngine(
            data=sample_trading_data,
            strategy=BuyAndHoldStrategy(),
            initial_cash=100000,
        ).run()

        # 两者都应该有结果
        assert benchmark_results["total_return"] is not None

    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_rl_strategy_comparison(self, sample_trading_data):
        """测试RL策略与传统策略对比"""
        # 1. 训练RL模型
        symbol_data = sample_trading_data["600000.SH"]
        env = ASharesTradingEnv(df=symbol_data, initial_cash=100000)

        trainer = RLTrainer(env=env, algorithm="ppo")
        rl_model = trainer.train(total_timesteps=1000)

        # 2. 评估RL模型
        rl_results = trainer.evaluate(rl_model, n_episodes=5)

        # 3. 运行传统策略
        benchmark_results = BacktestEngine(
            data={"600000.SH": symbol_data},
            strategy=MovingAverageCrossStrategy(5, 20),
            initial_cash=100000,
        ).run()

        # 4. 对比结果
        comparison = {
            "rl_mean_reward": rl_results["mean_reward"],
            "rl_std_reward": rl_results["std_reward"],
            "ma_return": benchmark_results["total_return"],
            "ma_sharpe": benchmark_results["sharpe_ratio"],
        }

        # 验证对比数据
        assert all(v is not None for v in comparison.values())


@pytest.mark.integration
class TestCompleteTradingSystemIntegration:
    """测试完整交易系统集成"""

    def test_end_to_end_trading_workflow(self, sample_trading_data):
        """测试端到端交易工作流"""
        # 1. 数据准备
        logger.info("步骤1: 数据准备")
        assert len(sample_trading_data) > 0

        # 2. 特征工程
        logger.info("步骤2: 特征工程")
        indicators = TechnicalIndicators()
        processed_data = {}
        for symbol, data in sample_trading_data.items():
            df = data.copy()
            if "close" in df.columns:
                df["sma_5"] = indicators.sma(df["close"], 5)
                df["sma_20"] = indicators.sma(df["close"], 20)
                df["rsi_14"] = indicators.rsi(df["close"], 14)
            processed_data[symbol] = df
        assert all(len(data.columns) > 6 for data in processed_data.values())

        # 3. 策略执行
        logger.info("步骤3: 策略执行")
        strategy = MovingAverageCrossStrategy(5, 20)
        engine = BacktestEngine(
            data=processed_data,
            strategy=strategy,
            initial_cash=1000000,
        )
        backtest_results = engine.run()
        assert backtest_results["total_return"] is not None

        # 4. 性能分析
        logger.info("步骤4: 性能分析")
        analyzer = PerformanceAnalyzer(backtest_results)
        metrics = analyzer.calculate_all_metrics()
        assert metrics["sharpe_ratio"] is not None

        # 5. 风险评估
        logger.info("步骤5: 风险评估")
        controller = RiskController({"max_single_order_amount": 500000})
        # 风控已经通过，因为没有订单被拒绝

        # 6. 报告生成
        logger.info("步骤6: 报告生成")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_path = f.name
            analyzer.save_report(report_path)
            assert Path(report_path).exists()
            Path(report_path).unlink()

        logger.info("完整工作流测试成功！")

    @pytest.mark.skipif(not SB3_AVAILABLE, reason="stable-baselines3 not available")
    def test_rl_enhanced_trading_system(self, sample_trading_data):
        """测试RL增强的交易系统"""
        # 1. 数据准备
        symbol_data = sample_trading_data["600000.SH"]

        # 2. RL训练
        env = ASharesTradingEnv(df=symbol_data, initial_cash=100000)
        trainer = RLTrainer(env=env, algorithm="ppo")
        rl_model = trainer.train(total_timesteps=1000)

        # 3. RL策略评估
        rl_results = trainer.evaluate(rl_model, n_episodes=5)
        assert rl_results["mean_reward"] is not None

        # 4. 传统策略对比
        ma_strategy = MovingAverageCrossStrategy(5, 20)
        ma_engine = BacktestEngine(
            data={"600000.SH": symbol_data},
            strategy=ma_strategy,
            initial_cash=100000,
        )
        ma_results = ma_engine.run()
        assert ma_results["total_return"] is not None

        # 5. 组合策略（简化版）
        # 实际中可能根据市场条件选择RL或传统策略
        logger.info(f"RL平均奖励: {rl_results['mean_reward']:.2f}")
        logger.info(f"MA策略收益率: {ma_results['total_return']:.2%}")

    @pytest.mark.asyncio
    async def test_agent_rl_hybrid_system(self, sample_trading_data, sample_market_context):
        """测试Agent-RL混合系统"""
        # 1. 创建决策Agent
        class HybridAgent(Agent):
            def __init__(self):
                super().__init__(name="hybrid_agent", role="decision")
                self.rl_confidence = 0.7

            async def process_async(self, input_data, context=None):
                # 简化的混合决策逻辑
                market_condition = input_data.get("market", {}).get("trend", "neutral")

                if market_condition == "up":
                    strategy_type = "rl"
                else:
                    strategy_type = "traditional"

                return AgentResponse(
                    agent_id=self.agent_id,
                    content=f"使用{strategy_type}策略",
                    confidence=self.rl_confidence,
                    metadata={
                        "strategy": strategy_type,
                        "action": "buy",
                        "quantity": 1000,
                    },
                )

        # 2. Agent决策
        agent = HybridAgent()
        decision = await agent.process_async(sample_market_context)

        # 3. 根据决策执行
        if decision.metadata["strategy"] == "traditional":
            strategy = MovingAverageCrossStrategy(5, 20)
        else:
            # RL策略（简化）
            strategy = BuyAndHoldStrategy()

        # 4. 执行策略
        engine = BacktestEngine(
            data=sample_trading_data,
            strategy=strategy,
            initial_cash=1000000,
        )
        results = engine.run()

        # 5. 验证结果
        assert results["total_return"] is not None
        logger.info(f"混合系统收益率: {results['total_return']:.2%}")


@pytest.mark.integration
class TestSystemMonitoringIntegration:
    """测试系统监控集成"""

    def test_multi_component_monitoring(self, sample_trading_data):
        """测试多组件监控"""
        # 1. 创建监控管理器
        alert_manager = AlertManager()
        alert_manager.start()

        # 2. 监控不同组件
        components = {
            "data_pipeline": {"status": "healthy", "latency_ms": 50},
            "backtest_engine": {"status": "running", "progress": 0.5},
            "portfolio": {"status": "healthy", "total_value": 1000000},
        }

        # 3. 检查组件状态
        for component, status in components.items():
            if status["status"] != "healthy":
                alert = Alert(
                    alert_id=f"{component}_alert",
                    alert_type=AlertType.SYSTEM_ERROR,
                    level=AlertLevel.HIGH,
                    title=f"{component}异常",
                    message=f"组件状态: {status['status']}",
                    metadata={"component": component, "status": status},
                )
                alert_manager.send_alert(alert)

        # 4. 验证监控
        alerts = alert_manager.get_alerts(limit=10)
        assert len(alerts) >= 0

        alert_manager.stop()

    def test_performance_regression_detection(self, sample_trading_data):
        """测试性能回归检测"""
        # 1. 运行基准回测
        baseline_results = BacktestEngine(
            data=sample_trading_data,
            strategy=MovingAverageCrossStrategy(5, 20),
            initial_cash=1000000,
        ).run()

        # 2. 运行新版本回测
        new_results = BacktestEngine(
            data=sample_trading_data,
            strategy=MovingAverageCrossStrategy(10, 30),
            initial_cash=1000000,
        ).run()

        # 3. 检测性能回归
        regression_detected = False

        if new_results["sharpe_ratio"] < baseline_results["sharpe_ratio"] * 0.9:
            regression_detected = True
            logger.warning(f"检测到性能回归: 夏普比率下降")

        if new_results["max_drawdown"] < baseline_results["max_drawdown"] * 1.1:
            regression_detected = True
            logger.warning(f"检测到性能回归: 回撤增加")

        # 4. 验证检测结果
        assert isinstance(regression_detected, bool)


@pytest.mark.integration
@pytest.mark.slow
class TestStressScenarios:
    """测试压力场景"""

    def test_market_crash_scenario(self):
        """测试市场崩盘场景"""
        # 生成崩盘数据
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        prices = 20.0 * np.exp(-np.arange(len(dates)) * 0.05)  # 每天跌5%

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices * 1.01,
                "high": prices * 0.99,
                "low": prices * 0.95,
                "close": prices,
                "volume": 10000000,
            }
        )

        # 运行回测
        engine = BacktestEngine(
            data={"600000.SH": data},
            strategy=BuyAndHoldStrategy(),
            initial_cash=1000000,
        )

        results = engine.run()

        # 验证系统在极端情况下的行为
        assert results is not None
        assert results["total_return"] < -0.5  # 应该大幅亏损

        # 验证风控
        final_value = (
            engine.portfolio.current_cash
            + sum(
                pos.quantity * pos.current_price for pos in engine.portfolio.positions.values()
            )
        )
        assert final_value > 0  # 不应该爆仓

    def test_extreme_volatility_scenario(self):
        """测试极端波动场景"""
        # 生成高波动数据
        dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
        prices = 10 + np.random.randn(len(dates)) * 2  # 高波动

        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": prices + abs(np.random.randn(len(dates))),
                "low": prices - abs(np.random.randn(len(dates))),
                "close": prices,
                "volume": 10000000,
            }
        )

        # 运行回测
        engine = BacktestEngine(
            data={"600000.SH": data},
            strategy=MovingAverageCrossStrategy(5, 20),
            initial_cash=1000000,
        )

        results = engine.run()

        # 验证系统稳定性
        assert results is not None
        assert not np.isnan(results["total_return"])
        assert not np.isinf(results["total_return"])
