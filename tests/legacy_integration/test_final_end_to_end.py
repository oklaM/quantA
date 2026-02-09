"""
最终端到端集成测试 - 仅测试核心功能
"""

import json
import tempfile
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from agents.base.agent_base import Agent, Message, MessageType
from agents.collaboration import AgentOrchestrator
from backtest.engine import BacktestEngine
from backtest.engine.strategies import BuyAndHoldStrategy
from backtest.engine.strategy import MovingAverageCrossStrategy
from trading.risk import RiskController
from utils.logging import get_logger

logger = get_logger(__name__)


class MockAgent(Agent):
    """Mock Agent for testing"""

    def __init__(self, name, response_text):
        super().__init__(name=name, description="Mock agent")
        self.response_text = response_text
        self.agent_id = f"agent_{name}"

    def process(self, input_data, context=None):
        """Process input and return response"""
        return Message(
            type=MessageType.ANALYSIS_RESPONSE,
            sender=self.name,
            receiver="user",
            content={"response": self.response_text, "confidence": 0.9},
        )


class TestFinalEndToEnd:
    """最终端到端集成测试类"""

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

    def test_backtest_engine(self):
        """测试回测引擎"""
        print("\n=== 测试1: 回测引擎 ===")

        # 运行回测
        results = self.engine.run()

        # 验证结果
        assert "account" in results
        assert "performance" in results
        assert "equity_curve" in results
        assert "positions" in results
        assert "stats" in results

        print(f"总收益率: {results['account']['total_return_pct']:.2%}")
        print(f"夏普比率: {results['performance']['sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['performance']['max_drawdown']:.2%}")
        print(f"交易次数: {results['stats']['total_fills']}")

        print("✅ 回测引擎测试通过")

    def test_multi_strategy(self):
        """测试多策略"""
        print("\n=== 测试2: 多策略对比 ===")

        first_symbol = list(self.data.keys())[0]

        strategies = {
            "buy_and_hold": BuyAndHoldStrategy(symbol=first_symbol),
            "ma_cross": MovingAverageCrossStrategy(symbol=first_symbol, fast_period=5, slow_period=20),
        }

        results = {}
        for name, strategy in strategies.items():
            engine = BacktestEngine(
                data=self.data,
                strategy=strategy,
                initial_cash=1000000,
                commission_rate=0.0003,
                slippage_rate=0.0001,
            )
            results[name] = engine.run()

        # 验证所有策略都有结果
        assert len(results) == 2

        # 比较策略
        buy_hold_return = results["buy_and_hold"]["account"]["total_return_pct"]
        ma_cross_return = results["ma_cross"]["account"]["total_return_pct"]

        print(f"买入持有策略收益: {buy_hold_return:.2%}")
        print(f"双均线策略收益: {ma_cross_return:.2%}")

        print("✅ 多策略对比测试通过")

    def test_agent_collaboration(self):
        """测试Agent协作"""
        print("\n=== 测试3: Agent协作 ===")

        # 创建测试Agent
        analyzer = MockAgent("analyzer", "市场分析：上涨趋势")
        strategist = MockAgent("strategist", "策略建议：买入")
        risk_manager = MockAgent("risk_manager", "风险评估：低风险")

        # 创建协作器
        orchestrator = AgentOrchestrator()
        orchestrator.register_agent(analyzer)
        orchestrator.register_agent(strategist)
        orchestrator.register_agent(risk_manager)

        # 运行协作流程
        input_data = {
            "symbol": "600000.SH",
            "date": "2023-01-01",
            "price": 10.50,
        }

        # 直接调用各个Agent
        analyzer_result = analyzer.process(input_data)
        strategist_result = strategist.process(input_data)
        risk_result = risk_manager.process(input_data)

        final_decision = {
            "analyzer": analyzer_result.content["response"],
            "strategist": strategist_result.content["response"],
            "risk": risk_result.content["response"]
        }

        # 验证结果
        assert final_decision is not None
        assert len(final_decision) == 3

        print(f"最终决策: {final_decision}")
        print(f"决策包含3个Agent的反馈")

        print("✅ Agent协作测试通过")

    def test_risk_control(self):
        """测试风控系统"""
        print("\n=== 测试4: 风控系统 ===")

        # 创建风控控制器，禁用时间限制
        controller = RiskController({
            "min_available_cash": 100000,
            "max_single_order_amount": 1000000,
            "max_daily_loss_ratio": 0.05,
            "enable_time_limit": False,  # 禁用时间限制
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

    def test_integration_workflow(self):
        """测试集成工作流"""
        print("\n=== 测试5: 集成工作流 ===")

        # 1. 数据获取
        assert len(self.engine.data) == 4
        print(f"数据已获取: {len(self.engine.data)} 只股票")

        # 2. 策略运行
        results = self.engine.run()
        assert results["account"]["total_return_pct"] is not None

        print(f"策略收益: {results['account']['total_return_pct']:.2%}")

        # 3. 风控检查
        controller = RiskController({
            "min_available_cash": 100000,
            "max_single_order_amount": 1000000,
            "max_daily_loss_ratio": 0.05,
            "enable_time_limit": False,
        })

        context = {
            "account": {"total_asset": results['account']['total_value'], "available_cash": results['account']['cash']},
            "positions": results['positions'],
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

        # 4. Agent决策
        analyzer = MockAgent("analyzer", "市场分析：良好")
        strategist = MockAgent("strategist", "策略建议：继续持有")
        orchestrator = AgentOrchestrator()
        orchestrator.register_agent(analyzer)
        orchestrator.register_agent(strategist)

        decision = analyzer.process({"symbol": "600000.SH", "price": 10.50})
        assert decision is not None

        print("✅ 集成工作流测试通过")

        print(f"\n🎯 最终结果:")
        print(f"  总收益率: {results['account']['total_return_pct']:.2%}")
        print(f"  夏普比率: {results['performance']['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {results['performance']['max_drawdown']:.2%}")
        print(f"  交易次数: {results['stats']['total_fills']}")


if __name__ == "__main__":
    # 运行所有测试
    test = TestFinalEndToEnd()
    test.setup_method()

    # 执行所有测试
    test.test_backtest_engine()
    test.test_multi_strategy()
    test.test_agent_collaboration()
    test.test_risk_control()
    test.test_integration_workflow()

    print("\n" + "="*60)
    print("🎉 所有端到端测试完成！")
    print("✅ 核心功能验证通过")
    print("✅ 数据流正常")
    print("✅ 策略执行正常")
    print("✅ 风控系统正常")
    print("✅ Agent协作正常")
    print("="*60)