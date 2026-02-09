"""
Agent协作集成测试
测试多Agent协同工作、消息传递和决策生成
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import pytest

from agents.base.agent_base import Agent, AgentResponse, Message, MessageType
from agents.base.coordinator import AgentCoordinator, Workflow
from agents.collaboration import AgentOrchestrator
from utils.logging import get_logger

logger = get_logger(__name__)


# ========== Mock Agents for Testing ==========


class MockMarketDataAgent(Agent):
    """模拟市场数据Agent"""

    def __init__(self):
        super().__init__(name="market_data_agent", agent_type="market_analysis")
        self.processed_count = 0

    async def process_async(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        self.processed_count += 1

        symbol = input_data.get("symbol", "unknown")
        date = input_data.get("date", "2023-01-01")

        return AgentResponse(
            agent_id=self.agent_id,
            content=f"市场数据：{symbol} 在 {date} 的价格趋势上涨",
            confidence=0.95,
            metadata={
                "symbol": symbol,
                "date": date,
                "price": 10.50,
                "volume": 1000000,
                "trend": "up",
            },
        )

    def process(self, input_data, context=None):
        """同步处理方法（必需的抽象方法）"""
        return AgentResponse(
            agent_id=self.agent_id,
            content="Mock response",
            confidence=0.9,
        )


class MockTechnicalAgent(Agent):
    """模拟技术分析Agent"""

    def __init__(self):
        super().__init__(name="technical_agent", agent_type="technical_analysis")
        self.signals_generated = 0

    async def process_async(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        self.signals_generated += 1

        # 根据输入生成技术信号
        price = input_data.get("price", 10.0)
        trend = input_data.get("metadata", {}).get("trend", "neutral")

        if trend == "up" and price < 11.0:
            signal = "买入信号"
            confidence = 0.85
        elif trend == "down" and price > 10.0:
            signal = "卖出信号"
            confidence = 0.80
        else:
            signal = "持有信号"
            confidence = 0.70

        return AgentResponse(
            agent_id=self.agent_id,
            content=f"技术分析：{signal}，MA金叉形成",
            confidence=confidence,
            metadata={
                "signal": signal,
                "indicators": {
                    "ma_5": 10.2,
                    "ma_20": 10.0,
                    "rsi": 55.0,
                },
            },
        )

    def process(self, input_data, context=None):
        """同步处理方法"""
        return AgentResponse(
            agent_id=self.agent_id,
            content="Mock technical response",
            confidence=0.9,
        )


class MockStrategyAgent(Agent):
    """模拟策略生成Agent"""

    def __init__(self):
        super().__init__(name="strategy_agent", agent_type="strategy_generation")
        self.strategies_created = 0

    async def process_async(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        self.strategies_created += 1

        signal = input_data.get("metadata", {}).get("signal", "持有")
        confidence = input_data.get("confidence", 0.7)

        # 根据信号生成策略
        if signal == "买入信号":
            action = "买入"
            position_size = 1000
        elif signal == "卖出信号":
            action = "卖出"
            position_size = 500
        else:
            action = "持有"
            position_size = 0

        return AgentResponse(
            agent_id=self.agent_id,
            content=f"策略建议：{action} {position_size} 股",
            confidence=confidence,
            metadata={
                "action": action,
                "position_size": position_size,
                "entry_price": 10.50,
                "stop_loss": 10.00,
                "take_profit": 11.50,
            },
        )

    def process(self, input_data, context=None):
        """同步处理方法"""
        return AgentResponse(
            agent_id=self.agent_id,
            content="Mock strategy response",
            confidence=0.9,
        )


class MockRiskAgent(Agent):
    """模拟风控Agent"""

    def __init__(self):
        super().__init__(name="risk_agent", agent_type="risk_management")
        self.risk_checks = 0

    async def process_async(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        self.risk_checks += 1

        action = input_data.get("metadata", {}).get("action", "持有")
        position_size = input_data.get("metadata", {}).get("position_size", 0)
        confidence = input_data.get("confidence", 0.7)

        # 风控检查
        risk_level = "低"
        approved = True

        if position_size > 2000:
            risk_level = "高"
            approved = False
        elif position_size > 1000:
            risk_level = "中"
            approved = confidence > 0.8

        return AgentResponse(
            agent_id=self.agent_id,
            content=f"风控评估：{risk_level}风险，{'批准' if approved else '拒绝'}交易",
            confidence=0.90 if approved else 0.60,
            metadata={
                "risk_level": risk_level,
                "approved": approved,
                "max_position": 2000 if approved else position_size * 0.5,
            },
        )

    def process(self, input_data, context=None):
        """同步处理方法"""
        return AgentResponse(
            agent_id=self.agent_id,
            content="Mock risk response",
            confidence=0.9,
        )


# ========== Fixtures ==========


@pytest.fixture
def sample_agents():
    """创建测试Agent集合"""
    return {
        "market": MockMarketDataAgent(),
        "technical": MockTechnicalAgent(),
        "strategy": MockStrategyAgent(),
        "risk": MockRiskAgent(),
    }


@pytest.fixture
def sample_input():
    """示例输入数据"""
    return {
        "symbol": "600000.SH",
        "date": "2023-01-01",
        "price": 10.50,
        "volume": 1000000,
    }


@pytest.fixture
def coordinator(sample_agents):
    """创建Agent协调器"""
    coord = AgentCoordinator()
    for agent in sample_agents.values():
        coord.register_agent(agent)
    return coord


@pytest.fixture
def orchestrator(sample_agents):
    """创建Agent编排器"""
    orch = AgentOrchestrator()
    for agent in sample_agents.values():
        orch.register_agent(agent)
    return orch


# ========== Tests ==========


@pytest.mark.integration
class TestAgentRegistration:
    """测试Agent注册和管理"""

    def test_register_single_agent(self, coordinator):
        """测试注册单个Agent"""
        agent = MockMarketDataAgent()
        coordinator.register_agent(agent)

        assert agent.agent_id in coordinator._agents
        assert len(coordinator.list_agents()) > 0

    def test_register_multiple_agents(self, coordinator, sample_agents):
        """测试注册多个Agent"""
        agent_ids = set(agent.agent_id for agent in sample_agents.values())

        assert len(agent_ids) == 4
        assert len(coordinator.list_agents()) == 4

    def test_agent_retrieval(self, coordinator, sample_agents):
        """测试Agent检索"""
        market_agent = sample_agents["market"]

        retrieved = coordinator.get_agent(market_agent.agent_id)
        assert retrieved is not None
        assert retrieved.name == "market_data_agent"

    def test_list_agents_by_role(self, coordinator, sample_agents):
        """测试按角色列出Agent"""
        all_agents = coordinator.list_agents()

        # 验证所有Agent都注册了
        assert len(all_agents) == 4

        # 验证角色
        roles = set(coordinator.get_agent(agent_id).role for agent_id in all_agents)
        assert "market_analysis" in roles
        assert "technical_analysis" in roles
        assert "strategy_generation" in roles
        assert "risk_management" in roles


@pytest.mark.integration
class TestMessageRouting:
    """测试消息路由机制"""

    @pytest.mark.asyncio
    async def test_send_message_to_agent(self, coordinator, sample_agents, sample_input):
        """测试发送消息到特定Agent"""
        market_agent = sample_agents["market"]

        message = Message(
            sender_id="system",
            receiver_id=market_agent.agent_id,
            message_type=MessageType.REQUEST,
            content=sample_input,
        )

        response = await coordinator.route_message(message)

        assert response is not None
        assert response.agent_id == market_agent.agent_id
        assert "市场数据" in response.content

    @pytest.mark.asyncio
    async def test_broadcast_message(self, coordinator, sample_agents, sample_input):
        """测试广播消息到所有Agent"""
        message = Message(
            sender_id="system",
            receiver_id="broadcast",
            message_type=MessageType.NOTIFICATION,
            content=sample_input,
        )

        responses = await coordinator.broadcast_message(
            sender="system",
            message_type=MessageType.NOTIFICATION,
            content=sample_input,
        )

        # 应该收到所有Agent的响应
        assert len(responses) == len(sample_agents)

    @pytest.mark.asyncio
    async def test_message_filtering(self, coordinator, sample_agents, sample_input):
        """测试消息过滤"""
        # 排除market agent
        exclude_ids = [sample_agents["market"].agent_id]

        responses = await coordinator.broadcast_message(
            sender="system",
            message_type=MessageType.REQUEST,
            content=sample_input,
            exclude=exclude_ids,
        )

        # 应该只收到3个响应（排除market）
        assert len(responses) == 3


@pytest.mark.integration
class TestAgentWorkflow:
    """测试Agent工作流程"""

    @pytest.mark.asyncio
    async def test_sequential_workflow(self, coordinator, sample_agents, sample_input):
        """测试顺序工作流"""
        # 创建工作流
        workflow = Workflow(name="trading_workflow", description="交易决策流程")

        # 添加步骤
        workflow.add_step(sample_agents["market"])
        workflow.add_step(sample_agents["technical"])
        workflow.add_step(sample_agents["strategy"])
        workflow.add_step(sample_agents["risk"])

        # 运行工作流
        result = await coordinator.run_workflow(workflow, initial_input=sample_input)

        # 验证结果
        assert result is not None
        assert "final_decision" in result or "metadata" in result

        # 验证所有Agent都被调用
        assert sample_agents["market"].processed_count == 1
        assert sample_agents["technical"].signals_generated == 1
        assert sample_agents["strategy"].strategies_created == 1
        assert sample_agents["risk"].risk_checks == 1

    @pytest.mark.asyncio
    async def test_conditional_workflow(self, coordinator, sample_agents, sample_input):
        """测试条件工作流"""
        workflow = Workflow(name="conditional_workflow")

        # 根据market数据决定是否执行technical分析
        market_response = await sample_agents["market"].process_async(sample_input)

        if market_response.metadata.get("trend") == "up":
            # 执行technical分析
            tech_response = await sample_agents["technical"].process_async(market_response.__dict__)
            assert tech_response is not None

    @pytest.mark.asyncio
    async def test_parallel_workflow(self, coordinator, sample_agents, sample_input):
        """测试并行工作流"""
        import time

        # 并行执行多个Agent
        start_time = time.time()

        tasks = [
            agent.process_async(sample_input)
            for agent in [sample_agents["market"], sample_agents["technical"]]
        ]

        results = await asyncio.gather(*tasks)

        elapsed_time = time.time() - start_time

        # 验证结果
        assert len(results) == 2
        assert all(r is not None for r in results)

        # 并行执行应该更快（虽然mock很快，但仍然验证逻辑）
        assert elapsed_time < 1.0


@pytest.mark.integration
class TestAgentOrchestrator:
    """测试Agent编排器"""

    @pytest.mark.asyncio
    async def test_create_and_run_workflow(self, orchestrator, sample_agents):
        """测试创建和运行工作流"""
        # 创建工作流
        workflow = orchestrator.create_workflow(
            name="analysis_workflow",
            description="市场分析流程",
            steps=[
                sample_agents["market"],
                sample_agents["technical"],
            ],
        )

        # 验证工作流创建
        assert "analysis_workflow" in orchestrator.list_workflows()

        # 运行工作流
        result = await orchestrator.run_workflow("analysis_workflow")

        # 验证结果
        assert result is not None

    @pytest.mark.asyncio
    async def test_orchestrator_message_handling(self, orchestrator, sample_input):
        """测试编排器消息处理"""
        # 创建消息
        message = Message(
            sender_id="system",
            receiver_id=sample_agents["market"].agent_id,
            message_type=MessageType.REQUEST,
            content=sample_input,
        )

        # 发送消息
        await orchestrator.send_message(message)

        # 验证Agent被调用
        assert sample_agents["market"].processed_count > 0

    @pytest.mark.asyncio
    async def test_orchestrator_broadcast(self, orchestrator, sample_input):
        """测试编排器广播"""
        # 广播消息
        await orchestrator.broadcast(
            sender="system",
            message_type=MessageType.NOTIFICATION,
            content=sample_input,
        )

        # 验证所有Agent都收到消息
        # 注意：这需要Agent实际实现了消息接收


@pytest.mark.integration
class TestAgentCollaboration:
    """测试Agent协作"""

    @pytest.mark.asyncio
    async def test_simple_collaboration(self, sample_agents, sample_input):
        """测试简单协作场景"""
        # Market Agent -> Technical Agent
        market_response = await sample_agents["market"].process_async(sample_input)

        # Technical Agent使用market的数据
        tech_response = await sample_agents["technical"].process_async(market_response.__dict__)

        # 验证数据传递
        assert tech_response.metadata["signal"] in ["买入信号", "卖出信号", "持有信号"]
        assert tech_response.confidence > 0

    @pytest.mark.asyncio
    async def test_complex_collaboration(self, sample_agents, sample_input):
        """测试复杂协作场景（完整流程）"""
        # 1. Market Analysis
        market_response = await sample_agents["market"].process_async(sample_input)

        # 2. Technical Analysis (based on market data)
        tech_response = await sample_agents["technical"].process_async(market_response.__dict__)

        # 3. Strategy Generation (based on technical signal)
        strategy_response = await sample_agents["strategy"].process_async(tech_response.__dict__)

        # 4. Risk Check (based on strategy)
        risk_response = await sample_agents["risk"].process_async(strategy_response.__dict__)

        # 验证完整流程
        assert market_response.confidence > 0.8
        assert tech_response.metadata.get("signal") is not None
        assert strategy_response.metadata.get("action") is not None
        assert risk_response.metadata.get("approved") in [True, False]

        # 验证决策链条
        final_decision = {
            "symbol": sample_input["symbol"],
            "action": strategy_response.metadata["action"],
            "position_size": strategy_response.metadata["position_size"],
            "approved": risk_response.metadata["approved"],
            "risk_level": risk_response.metadata["risk_level"],
        }

        assert final_decision["symbol"] == "600000.SH"

    @pytest.mark.asyncio
    async def test_error_handling_in_collaboration(self, sample_agents, sample_input):
        """测试协作中的错误处理"""
        # 发送无效数据
        invalid_input = {"invalid": "data"}

        # Agent应该能够处理无效输入
        try:
            response = await sample_agents["market"].process_async(invalid_input)
            # 如果返回了响应，验证它
            assert response is not None
        except Exception as e:
            # 或者抛出异常
            assert isinstance(e, (ValueError, KeyError))


@pytest.mark.integration
class TestAgentContextManagement:
    """测试Agent上下文管理"""

    @pytest.mark.asyncio
    async def test_context_sharing(self, sample_agents):
        """测试上下文共享"""
        context = {
            "account": {"cash": 1000000, "positions": {}},
            "market": {"trend": "bull", "volatility": "low"},
        }

        input_data = {"symbol": "600000.SH", "price": 10.50}

        # Agent使用context
        response = await sample_agents["strategy"].process_async(input_data, context=context)

        assert response is not None

    @pytest.mark.asyncio
    async def test_context_accumulation(self, sample_agents, sample_input):
        """测试上下文累积"""
        context = {}

        # 多个Agent依次更新context
        market_response = await sample_agents["market"].process_async(sample_input, context=context)
        context["market_analysis"] = market_response.__dict__

        tech_response = await sample_agents["technical"].process_async(market_response.__dict__, context=context)
        context["technical_analysis"] = tech_response.__dict__

        # 验证context累积
        assert "market_analysis" in context
        assert "technical_analysis" in context


@pytest.mark.integration
class TestAgentPerformance:
    """测试Agent性能"""

    @pytest.mark.asyncio
    async def test_concurrent_agent_execution(self, sample_agents):
        """测试并发Agent执行"""
        import time

        input_data = {"symbol": "600000.SH", "price": 10.50}

        # 顺序执行
        start_time = time.time()
        for agent in sample_agents.values():
            await agent.process_async(input_data)
        sequential_time = time.time() - start_time

        # 并发执行
        start_time = time.time()
        await asyncio.gather(*[agent.process_async(input_data) for agent in sample_agents.values()])
        concurrent_time = time.time() - start_time

        # 并发应该更快或至少差不多
        logger.info(f"顺序执行: {sequential_time:.3f}s, 并发执行: {concurrent_time:.3f}s")

    @pytest.mark.asyncio
    async def test_message_throughput(self, coordinator, sample_agents):
        """测试消息吞吐量"""
        import time

        messages = [
            Message(
                sender_id="system",
                receiver_id=sample_agents["market"].agent_id,
                message_type=MessageType.REQUEST,
                content={"symbol": f"60000{i}.SH", "price": 10.0 + i},
            )
            for i in range(10)
        ]

        start_time = time.time()

        # 批量发送消息
        responses = await asyncio.gather(*[coordinator.route_message(msg) for msg in messages])

        elapsed_time = time.time() - start_time

        # 验证所有消息都得到响应
        assert len(responses) == 10
        assert all(r is not None for r in responses)

        # 计算吞吐量
        throughput = len(messages) / elapsed_time
        logger.info(f"消息吞吐量: {throughput:.2f} msg/s")

        assert throughput > 1.0  # 至少1 msg/s


@pytest.mark.integration
class TestAgentErrorRecovery:
    """测试Agent错误恢复"""

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, coordinator, sample_agents, sample_input):
        """测试Agent失败处理"""
        # 模拟一个会失败的Agent
        class FailingAgent(Agent):
            async def process_async(self, input_data, context=None):
                raise RuntimeError("Agent failed")

        failing_agent = FailingAgent(name="failing_agent", role="test")
        coordinator.register_agent(failing_agent)

        # 发送消息
        message = Message(
            sender_id="system",
            receiver_id=failing_agent.agent_id,
            message_type=MessageType.REQUEST,
            content=sample_input,
        )

        # 应该捕获异常
        try:
            response = await coordinator.route_message(message)
            # 如果没有抛出异常，验证响应
            assert response is None or response.metadata.get("error") is not None
        except RuntimeError:
            # 或者抛出异常
            pass

    @pytest.mark.asyncio
    async def test_timeout_handling(self, coordinator, sample_agents, sample_input):
        """测试超时处理"""
        # 模拟一个会超时的Agent
        class SlowAgent(Agent):
            async def process_async(self, input_data, context=None):
                await asyncio.sleep(5)  # 超时
                return AgentResponse(agent_id=self.agent_id, content="slow", confidence=0.5)

        slow_agent = SlowAgent(name="slow_agent", role="test")
        coordinator.register_agent(slow_agent)

        # 使用timeout
        try:
            response = await asyncio.wait_for(
                coordinator.route_message(
                    Message(
                        sender_id="system",
                        receiver_id=slow_agent.agent_id,
                        message_type=MessageType.REQUEST,
                        content=sample_input,
                    )
                ),
                timeout=1.0,
            )
            assert False, "应该超时"
        except asyncio.TimeoutError:
            # 预期的超时
            pass
