"""
测试LLM Agents
测试agents模块中的各种Agent
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agents.base.agent_base import LLMAgent, Message, MessageType


# Create a concrete test agent since LLMAgent is abstract
# Note: Not using "Test" prefix to avoid pytest collection warning
class ConcreteTestAgent(LLMAgent):
    """Concrete implementation of LLMAgent for testing"""

    async def process(self, message: Message) -> Message:
        """Process a message and return a response"""
        # Simple echo implementation for testing
        response = Message(
            type=MessageType.ANALYSIS_RESPONSE,
            sender=self.name,
            receiver=message.sender,
            content={"response": f"Processed: {message.content}"},
        )
        return response


@pytest.mark.unit
class TestLLMAgent:
    """测试LLM Agent基类"""

    def test_initialization(self):
        """测试初始化"""
        agent = ConcreteTestAgent(name="test_agent", description="这是一个测试")
        assert agent.name == "test_agent"
        assert agent.description == "这是一个测试"

    def test_send_message(self):
        """测试发送消息"""
        agent = ConcreteTestAgent(name="test")

        message_id = agent.send_message(
            receiver="receiver", message_type=MessageType.MARKET_DATA, content={"price": 100.0}
        )

        assert message_id is not None
        assert message_id.startswith("msg_")

    def test_receive_message(self):
        """测试接收消息"""
        agent = ConcreteTestAgent(name="test")

        # Send a message to self
        agent.send_message(
            receiver="test", message_type=MessageType.MARKET_DATA, content={"price": 100.0}
        )

        # Receive the message
        received = agent.receive_message(MessageType.MARKET_DATA)

        assert received is not None
        assert received.content == {"price": 100.0}

    @pytest.mark.asyncio
    async def test_process_message(self):
        """测试处理消息"""
        agent = ConcreteTestAgent(name="test")

        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="user",
            receiver="test",
            content={"data": "test"},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE
        assert response.sender == "test"


@pytest.mark.unit
class TestMessage:
    """测试消息类"""

    def test_message_creation(self):
        """测试消息创建"""
        message = Message(
            type=MessageType.MARKET_DATA,
            sender="sender",
            receiver="receiver",
            content={"price": 100.0},
            timestamp=datetime.now(),
        )

        assert message.content == {"price": 100.0}
        assert message.type == MessageType.MARKET_DATA
        assert message.sender == "sender"
        assert message.receiver == "receiver"

    def test_message_to_dict(self):
        """测试消息转换为字典"""
        timestamp = datetime.now()
        message = Message(
            type=MessageType.TECHNICAL_DATA,
            sender="sender",
            receiver="receiver",
            content={"indicator": "RSI"},
            timestamp=timestamp,
        )

        message_dict = message.to_dict()

        assert message_dict["content"] == {"indicator": "RSI"}
        assert message_dict["type"] == "technical_data"
        assert message_dict["sender"] == "sender"
        assert message_dict["receiver"] == "receiver"


@pytest.mark.unit
class TestMessageType:
    """测试消息类型枚举"""

    def test_message_types(self):
        """测试消息类型"""
        assert MessageType.MARKET_DATA.value == "market_data"
        assert MessageType.TECHNICAL_DATA.value == "technical_data"
        assert MessageType.FUNDAMENTAL_DATA.value == "fundamental_data"
        assert MessageType.SENTIMENT_DATA.value == "sentiment_data"
        assert MessageType.ANALYSIS_REQUEST.value == "analysis_request"
        assert MessageType.ANALYSIS_RESPONSE.value == "analysis_response"
        assert MessageType.SIGNAL_REQUEST.value == "signal_request"
        assert MessageType.SIGNAL_RESPONSE.value == "signal_response"
        assert MessageType.RISK_CHECK_REQUEST.value == "risk_check_request"
        assert MessageType.RISK_CHECK_RESPONSE.value == "risk_check_response"
        assert MessageType.ERROR.value == "error"
        assert MessageType.ACK.value == "ack"


@pytest.mark.unit
class TestAgentCoordinator:
    """测试Agent协调器"""

    def test_coordinator_initialization(self):
        """测试协调器初始化"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()
        assert len(coordinator.list_agents()) == 0  # Fixed: use list_agents() method

    def test_register_agent(self):
        """测试注册Agent"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()
        agent = ConcreteTestAgent(name="test_agent")

        coordinator.register_agent(agent)

        assert "test_agent" in coordinator.list_agents()  # Fixed: use list_agents()
        assert coordinator.get_agent("test_agent") == agent

    def test_get_agent(self):
        """测试获取Agent"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()
        agent = ConcreteTestAgent(name="test_agent")

        coordinator.register_agent(agent)

        retrieved = coordinator.get_agent("test_agent")

        assert retrieved == agent

    def test_unregister_agent(self):
        """测试注销Agent"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()
        agent = ConcreteTestAgent(name="test_agent")

        coordinator.register_agent(agent)
        assert "test_agent" in coordinator.list_agents()

        coordinator.unregister_agent("test_agent")  # Fixed: method name
        assert "test_agent" not in coordinator.list_agents()

    @pytest.mark.asyncio
    async def test_start_stop_agents(self):
        """测试启动和停止Agent"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()

        agent1 = ConcreteTestAgent(name="agent1")
        agent2 = ConcreteTestAgent(name="agent2")

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        # Start all agents
        await coordinator.start_all()
        assert agent1._is_running is True
        assert agent2._is_running is True

        # Stop all agents
        await coordinator.stop_all()
        assert agent1._is_running is False
        assert agent2._is_running is False

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """测试广播消息"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()

        agent1 = ConcreteTestAgent(name="agent1")
        agent2 = ConcreteTestAgent(name="agent2")

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        # Broadcast message
        await coordinator.broadcast_message(
            sender="system", message_type=MessageType.MARKET_DATA, content={"test": "data"}
        )

        # Both agents should receive the message in their global queue
        assert coordinator.global_message_queue.size() >= 2

    def test_list_agents(self):
        """测试列出所有Agent"""
        from agents.base.coordinator import AgentCoordinator

        coordinator = AgentCoordinator()

        agent1 = ConcreteTestAgent(name="agent1")
        agent2 = ConcreteTestAgent(name="agent2")

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        agents_list = coordinator.list_agents()

        assert len(agents_list) == 2
        assert "agent1" in agents_list
        assert "agent2" in agents_list
