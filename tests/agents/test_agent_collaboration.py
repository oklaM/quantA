"""
Agent协作流程的集成测试
测试多个Agent之间的消息传递和协作
"""

import asyncio
from datetime import datetime

import pytest

from agents.base.agent_base import (
    LLMAgent,
    Message,
    MessageQueue,
    MessageType,
)


class MockMarketDataAgent(LLMAgent):
    """模拟市场数据Agent"""

    def __init__(self):
        super().__init__(name="MarketDataAgent", description="提供市场数据")

    async def process(self, message: Message) -> Message:
        """处理消息"""
        if message.type == MessageType.MARKET_DATA:
            # 返回模拟市场数据
            return Message(
                type=MessageType.MARKET_DATA,
                sender=self.name,
                receiver=message.sender,
                content={
                    "symbol": message.content.get("symbol", "000001.SZ"),
                    "price": 10.50,
                    "volume": 1000000,
                    "ma5": 10.45,
                    "ma10": 10.40,
                    "ma20": 10.35,
                    "rsi": 55.0,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        return None


class MockTechnicalAgent(LLMAgent):
    """模拟技术分析Agent"""

    def __init__(self):
        super().__init__(name="TechnicalAgent", description="进行技术分析")

    async def process(self, message: Message) -> Message:
        """处理消息"""
        if message.type == MessageType.ANALYSIS_REQUEST:
            market_data = message.content.get("market_data", {})

            # 返回分析结果
            return Message(
                type=MessageType.ANALYSIS_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content={
                    "trend": (
                        "上涨" if market_data.get("ma5", 0) > market_data.get("ma20", 0) else "下跌"
                    ),
                    "strength": 0.75,
                    "recommendation": "买入",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        return None


class MockStrategyAgent(LLMAgent):
    """模拟策略生成Agent"""

    def __init__(self):
        super().__init__(name="StrategyAgent", description="生成交易策略")

    async def process(self, message: Message) -> Message:
        """处理消息"""
        if message.type == MessageType.SIGNAL_REQUEST:
            analysis = message.content.get("analysis", {})

            # 返回交易信号
            return Message(
                type=MessageType.SIGNAL_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content={
                    "action": "买入" if analysis.get("recommendation") == "买入" else "观望",
                    "quantity": 1000,
                    "price": "市价",
                    "confidence": analysis.get("strength", 0.5),
                    "timestamp": datetime.now().isoformat(),
                },
            )
        return None


class MockRiskAgent(LLMAgent):
    """模拟风控Agent"""

    def __init__(self):
        super().__init__(name="RiskAgent", description="风险控制")

    async def process(self, message: Message) -> Message:
        """处理消息"""
        if message.type == MessageType.RISK_CHECK_REQUEST:
            signal = message.content.get("signal", {})

            # 简单风控检查
            approved = signal.get("confidence", 0) > 0.6

            return Message(
                type=MessageType.RISK_CHECK_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content={
                    "approved": approved,
                    "reason": "信心度不足" if not approved else "通过",
                    "max_position": 0.3 if approved else 0,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        return None


@pytest.mark.agents
@pytest.mark.asyncio
class TestAgentCommunication:
    """测试Agent通信"""

    async def test_message_creation(self):
        """测试消息创建"""
        message = Message(
            type=MessageType.MARKET_DATA,
            sender="Agent1",
            receiver="Agent2",
            content={"data": "test"},
        )

        assert message.type == MessageType.MARKET_DATA
        assert message.sender == "Agent1"
        assert message.receiver == "Agent2"
        assert message.content == {"data": "test"}
        assert message.message_id is not None

    async def test_message_queue(self):
        """测试消息队列"""
        queue = MessageQueue()

        # 发送消息
        msg1 = Message(
            type=MessageType.MARKET_DATA,
            sender="Agent1",
            receiver="Agent2",
            content={"data": "test1"},
        )
        msg2 = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="Agent1",
            receiver="Agent2",
            content={"data": "test2"},
        )

        queue.send(msg1)
        queue.send(msg2)

        assert queue.size() == 2

        # 接收消息
        received = queue.receive("Agent2")
        assert received is not None
        assert received.type == MessageType.MARKET_DATA

        assert queue.size() == 1

    async def test_agent_send_receive(self):
        """测试Agent发送和接收消息"""
        agent1 = MockMarketDataAgent()
        agent2 = MockTechnicalAgent()

        await agent1.start()
        await agent2.start()

        # Agent2发送请求给Agent1
        request = Message(
            type=MessageType.MARKET_DATA,
            sender="TechnicalAgent",
            receiver="MarketDataAgent",
            content={"symbol": "000001.SZ"},
        )

        # 使用共享的消息队列
        shared_queue = MessageQueue()
        agent1.message_queue = shared_queue
        agent2.message_queue = shared_queue

        # Agent1发送消息
        agent1.send_message(
            receiver="TechnicalAgent",
            message_type=MessageType.MARKET_DATA,
            content={"symbol": "000001.SZ", "price": 10.50},
        )

        # Agent2接收消息
        received = agent2.receive_message()
        assert received is not None
        assert received.sender == "MarketDataAgent"
        assert received.content["symbol"] == "000001.SZ"

        await agent1.stop()
        await agent2.stop()


@pytest.mark.agents
@pytest.mark.asyncio
class TestAgentCollaboration:
    """测试Agent协作"""

    async def test_simple_two_agent_collaboration(self):
        """测试两个Agent的简单协作"""
        market_agent = MockMarketDataAgent()
        tech_agent = MockTechnicalAgent()

        await market_agent.start()
        await tech_agent.start()

        # 共享消息队列
        shared_queue = MessageQueue()
        market_agent.message_queue = shared_queue
        tech_agent.message_queue = shared_queue

        # 1. 技术Agent请求数据
        data_request = Message(
            type=MessageType.MARKET_DATA,
            sender="TechnicalAgent",
            receiver="MarketDataAgent",
            content={"symbol": "000001.SZ"},
        )

        # 处理请求
        market_response = await market_agent.process(data_request)

        assert market_response is not None
        assert market_response.type == MessageType.MARKET_DATA
        assert market_response.content["symbol"] == "000001.SZ"

        # 2. 技术Agent分析数据
        analysis_request = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="User",
            receiver="TechnicalAgent",
            content={"market_data": market_response.content},
        )

        analysis_response = await tech_agent.process(analysis_request)

        assert analysis_response is not None
        assert analysis_response.type == MessageType.ANALYSIS_RESPONSE
        assert "trend" in analysis_response.content

        await market_agent.stop()
        await tech_agent.stop()

    async def test_three_agent_pipeline(self):
        """测试三个Agent的流水线协作"""
        market_agent = MockMarketDataAgent()
        tech_agent = MockTechnicalAgent()
        strategy_agent = MockStrategyAgent()

        await market_agent.start()
        await tech_agent.start()
        await strategy_agent.start()

        # 步骤1: 获取市场数据
        data_request = Message(
            type=MessageType.MARKET_DATA,
            sender="User",
            receiver="MarketDataAgent",
            content={"symbol": "000001.SZ"},
        )

        market_response = await market_agent.process(data_request)
        assert market_response is not None

        # 步骤2: 技术分析
        analysis_request = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="User",
            receiver="TechnicalAgent",
            content={"market_data": market_response.content},
        )

        analysis_response = await tech_agent.process(analysis_request)
        assert analysis_response is not None
        assert analysis_response.content["trend"] in ["上涨", "下跌"]

        # 步骤3: 生成策略
        signal_request = Message(
            type=MessageType.SIGNAL_REQUEST,
            sender="User",
            receiver="StrategyAgent",
            content={"analysis": analysis_response.content},
        )

        signal_response = await strategy_agent.process(signal_request)
        assert signal_response is not None
        assert signal_response.content["action"] in ["买入", "观望"]

        await market_agent.stop()
        await tech_agent.stop()
        await strategy_agent.stop()

    async def test_four_agent_full_workflow(self):
        """测试四个Agent的完整工作流"""
        market_agent = MockMarketDataAgent()
        tech_agent = MockTechnicalAgent()
        strategy_agent = MockStrategyAgent()
        risk_agent = MockRiskAgent()

        # 启动所有Agent
        await market_agent.start()
        await tech_agent.start()
        await strategy_agent.start()
        await risk_agent.start()

        # 完整工作流
        # 1. 获取市场数据
        market_response = await market_agent.process(
            Message(
                type=MessageType.MARKET_DATA,
                sender="User",
                receiver="MarketDataAgent",
                content={"symbol": "000001.SZ"},
            )
        )

        # 2. 技术分析
        analysis_response = await tech_agent.process(
            Message(
                type=MessageType.ANALYSIS_REQUEST,
                sender="User",
                receiver="TechnicalAgent",
                content={"market_data": market_response.content},
            )
        )

        # 3. 生成交易信号
        signal_response = await strategy_agent.process(
            Message(
                type=MessageType.SIGNAL_REQUEST,
                sender="User",
                receiver="StrategyAgent",
                content={"analysis": analysis_response.content},
            )
        )

        # 4. 风控检查
        risk_response = await risk_agent.process(
            Message(
                type=MessageType.RISK_CHECK_REQUEST,
                sender="User",
                receiver="RiskAgent",
                content={"signal": signal_response.content},
            )
        )

        # 验证完整流程
        assert market_response.content["symbol"] == "000001.SZ"
        assert "trend" in analysis_response.content
        assert "action" in signal_response.content
        assert "approved" in risk_response.content

        # 如果信心度足够，交易应该被批准
        if analysis_response.content.get("strength", 0) > 0.6:
            assert risk_response.content["approved"] is True

        await market_agent.stop()
        await tech_agent.stop()
        await strategy_agent.stop()
        await risk_agent.stop()


@pytest.mark.agents
@pytest.mark.asyncio
class TestAgentErrorHandling:
    """测试Agent错误处理"""

    async def test_unsupported_message_type(self):
        """测试不支持的消息类型"""
        agent = MockMarketDataAgent()
        await agent.start()

        # 发送不支持的消息类型
        message = Message(
            type=MessageType.ERROR,
            sender="User",
            receiver="MarketDataAgent",
            content={"test": "data"},
        )

        response = await agent.process(message)

        # 应该返回None或错误消息
        assert response is None

        await agent.stop()

    async def test_missing_required_fields(self):
        """测试缺少必需字段"""
        agent = MockTechnicalAgent()
        await agent.start()

        # 发送缺少market_data的消息
        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="User",
            receiver="TechnicalAgent",
            content={},  # 缺少market_data
        )

        response = await agent.process(message)

        # 应该返回响应但使用默认值
        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE

        await agent.stop()


@pytest.mark.agents
@pytest.mark.asyncio
class TestAgentMessageHistory:
    """测试Agent消息历史"""

    async def test_message_id_generation(self):
        """测试消息ID生成"""
        queue = MessageQueue()

        msg1 = Message(
            type=MessageType.MARKET_DATA, sender="Agent1", receiver="Agent2", content={"test": 1}
        )

        msg2 = Message(
            type=MessageType.MARKET_DATA, sender="Agent1", receiver="Agent2", content={"test": 2}
        )

        queue.send(msg1)
        queue.send(msg2)

        # 验证ID唯一
        assert msg1.message_id != msg2.message_id

    async def test_message_retrieval_by_id(self):
        """测试通过ID检索消息"""
        queue = MessageQueue()

        msg = Message(
            type=MessageType.MARKET_DATA,
            sender="Agent1",
            receiver="Agent2",
            content={"test": "data"},
        )

        queue.send(msg)

        # 通过ID检索
        retrieved = queue.get_by_id(msg.message_id)
        assert retrieved is not None
        assert retrieved.content == {"test": "data"}

    async def test_message_reply_chain(self):
        """测试消息回复链"""
        original_msg = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="User",
            receiver="Agent",
            content={"data": "test"},
        )

        reply_msg = Message(
            type=MessageType.ANALYSIS_RESPONSE,
            sender="Agent",
            receiver="User",
            content={"result": "ok"},
            reply_to=original_msg.message_id,
        )

        assert reply_msg.reply_to == original_msg.message_id
