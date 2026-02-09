"""
Agent协调器模块单元测试
测试 agents/base/coordinator.py 中的所有类和方法
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from datetime import datetime

from agents.base.coordinator import (
    AgentCoordinator,
    Workflow,
)
from agents.base.agent_base import (
    Agent,
    Message,
    MessageQueue,
    MessageType,
)


# 创建一个测试用的Agent
class TestAgent(Agent):
    """测试用的Agent实现"""

    def __init__(self, name: str):
        super().__init__(name)
        self.started = False
        self.stopped = False
        self.processed_messages = []

    async def start(self):
        """启动Agent"""
        self.started = True
        await asyncio.sleep(0.01)  # 模拟异步操作

    async def stop(self):
        """停止Agent"""
        self.stopped = True
        await asyncio.sleep(0.01)  # 模拟异步操作

    async def process(self, message: Message) -> Message:
        """处理消息"""
        self.processed_messages.append(message)
        # 返回一个响应消息
        return Message(
            type=MessageType.RESPONSE,
            sender=self.name,
            receiver=message.sender,
            content={"status": "processed"},
            reply_to=message.message_id
        )


@pytest.mark.unit
class TestAgentCoordinator:
    """Agent协调器测试"""

    @pytest.fixture
    def coordinator(self):
        """创建协调器实例"""
        return AgentCoordinator()

    @pytest.fixture
    def test_agents(self):
        """创建测试Agent"""
        agents = {
            "agent1": TestAgent("agent1"),
            "agent2": TestAgent("agent2"),
            "agent3": TestAgent("agent3"),
        }
        return agents

    def test_coordinator_creation(self, coordinator):
        """测试协调器创建"""
        assert coordinator._is_running is False
        assert len(coordinator._agents) == 0
        assert isinstance(coordinator.global_message_queue, MessageQueue)

    def test_register_agent(self, coordinator, test_agents):
        """测试注册Agent"""
        agent1 = test_agents["agent1"]
        coordinator.register_agent(agent1)

        assert "agent1" in coordinator._agents
        assert coordinator._agents["agent1"] == agent1

    def test_register_duplicate_agent(self, coordinator, test_agents):
        """测试注册重复Agent"""
        agent1 = test_agents["agent1"]
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent1)  # 再次注册

        # 应该只有一个
        assert len(coordinator._agents) == 1

    def test_unregister_agent(self, coordinator, test_agents):
        """测试注销Agent"""
        agent1 = test_agents["agent1"]
        coordinator.register_agent(agent1)
        coordinator.unregister_agent("agent1")

        assert "agent1" not in coordinator._agents

    def test_unregister_nonexistent_agent(self, coordinator):
        """测试注销不存在的Agent"""
        # 不应该抛出异常
        coordinator.unregister_agent("nonexistent")

    def test_get_agent(self, coordinator, test_agents):
        """测试获取Agent"""
        agent1 = test_agents["agent1"]
        coordinator.register_agent(agent1)

        retrieved_agent = coordinator.get_agent("agent1")
        assert retrieved_agent == agent1

    def test_get_nonexistent_agent(self, coordinator):
        """测试获取不存在的Agent"""
        retrieved_agent = coordinator.get_agent("nonexistent")
        assert retrieved_agent is None

    def test_list_agents(self, coordinator, test_agents):
        """测试列出所有Agent"""
        for agent in test_agents.values():
            coordinator.register_agent(agent)

        agent_list = coordinator.list_agents()
        assert set(agent_list) == {"agent1", "agent2", "agent3"}

    def test_list_agents_empty(self, coordinator):
        """测试列出空Agent列表"""
        agent_list = coordinator.list_agents()
        assert agent_list == []

    @pytest.mark.asyncio
    async def test_start_all_agents(self, coordinator, test_agents):
        """测试启动所有Agent"""
        for agent in test_agents.values():
            coordinator.register_agent(agent)

        await coordinator.start_all()

        # 所有Agent应该已启动
        for agent in test_agents.values():
            assert agent.started is True

        assert coordinator._is_running is True

    @pytest.mark.asyncio
    async def test_stop_all_agents(self, coordinator, test_agents):
        """测试停止所有Agent"""
        for agent in test_agents.values():
            coordinator.register_agent(agent)

        await coordinator.start_all()
        await coordinator.stop_all()

        # 所有Agent应该已停止
        for agent in test_agents.values():
            assert agent.stopped is True

        assert coordinator._is_running is False

    @pytest.mark.asyncio
    async def test_broadcast_message(self, coordinator, test_agents):
        """测试广播消息"""
        for agent in test_agents.values():
            coordinator.register_agent(agent)

        sender = "agent1"
        message_type = MessageType.MARKET_DATA
        content = {"test": "data"}

        await coordinator.broadcast_message(
            sender=sender,
            message_type=message_type,
            content=content
        )

        # 检查全局消息队列
        # 广播应该发送给其他两个agent（agent2和agent3）
        queue = coordinator.global_message_queue
        # 应该有2条消息（agent2和agent3各一条）
        messages = []
        while not queue.empty():
            messages.append(queue.receive())

        assert len(messages) == 2

    @pytest.mark.asyncio
    async def test_broadcast_message_with_exclude(self, coordinator, test_agents):
        """测试带排除的广播消息"""
        for agent in test_agents.values():
            coordinator.register_agent(agent)

        await coordinator.broadcast_message(
            sender="agent1",
            message_type=MessageType.DATA,
            content={"test": "data"},
            exclude=["agent2"]
        )

        # 只应该发送给agent3（排除了agent2）
        queue = coordinator.global_message_queue
        messages = []
        while not queue.empty():
            messages.append(queue.receive())

        assert len(messages) == 1
        assert messages[0].receiver == "agent3"

    @pytest.mark.asyncio
    async def test_route_message(self, coordinator, test_agents):
        """测试路由消息"""
        agent1 = test_agents["agent1"]
        agent2 = test_agents["agent2"]
        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        message = Message(
            type=MessageType.MARKET_DATA,
            sender="agent1",
            receiver="agent2",
            content={"test": "data"}
        )

        await coordinator.route_message(message)

        # agent2应该收到并处理消息
        assert len(agent2.processed_messages) == 1
        assert agent2.processed_messages[0] == message

    @pytest.mark.asyncio
    async def test_route_message_to_nonexistent_agent(self, coordinator, test_agents):
        """测试路由消息到不存在的Agent"""
        agent1 = test_agents["agent1"]
        coordinator.register_agent(agent1)

        message = Message(
            type=MessageType.MARKET_DATA,
            sender="agent1",
            receiver="nonexistent",
            content={"test": "data"}
        )

        # 不应该抛出异常
        await coordinator.route_message(message)

    @pytest.mark.asyncio
    async def test_run_workflow(self, coordinator):
        """测试运行工作流"""
        workflow = Workflow("test_workflow", "Test workflow description")

        # 添加一个简单的步骤
        async def test_step(coordinator):
            return {"status": "success"}

        workflow.add_step(test_step)

        result = await coordinator.run_workflow(workflow)

        assert "step_1" in result
        assert result["step_1"]["status"] == "success"


@pytest.mark.unit
class TestWorkflow:
    """工作流测试"""

    def test_workflow_creation(self):
        """测试工作流创建"""
        workflow = Workflow("test_workflow", "Test description")

        assert workflow.name == "test_workflow"
        assert workflow.description == "Test description"
        assert len(workflow._steps) == 0

    def test_workflow_creation_without_description(self):
        """测试不带描述的工作流创建"""
        workflow = Workflow("test_workflow")

        assert workflow.name == "test_workflow"
        assert workflow.description == ""

    def test_workflow_add_step(self):
        """测试添加步骤"""
        workflow = Workflow("test_workflow")

        async def test_step(coordinator):
            return {"result": "step1"}

        workflow.add_step(test_step)

        assert len(workflow._steps) == 1

    def test_workflow_add_multiple_steps(self):
        """测试添加多个步骤"""
        workflow = Workflow("test_workflow")

        async def step1(coordinator):
            return {"result": "step1"}

        async def step2(coordinator):
            return {"result": "step2"}

        workflow.add_step(step1)
        workflow.add_step(step2)

        assert len(workflow._steps) == 2

    @pytest.mark.asyncio
    async def test_workflow_execute_single_step(self):
        """测试执行单步骤工作流"""
        workflow = Workflow("test_workflow")

        async def test_step(coordinator):
            return {"status": "completed"}

        workflow.add_step(test_step)

        coordinator = AgentCoordinator()
        result = await workflow.execute(coordinator)

        assert "step_1" in result
        assert result["step_1"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_workflow_execute_multiple_steps(self):
        """测试执行多步骤工作流"""
        workflow = Workflow("test_workflow")

        async def step1(coordinator):
            return {"value": 1}

        async def step2(coordinator):
            return {"value": 2}

        async def step3(coordinator):
            return {"value": 3}

        workflow.add_step(step1)
        workflow.add_step(step2)
        workflow.add_step(step3)

        coordinator = AgentCoordinator()
        result = await workflow.execute(coordinator)

        assert len(result) == 3
        assert result["step_1"]["value"] == 1
        assert result["step_2"]["value"] == 2
        assert result["step_3"]["value"] == 3

    @pytest.mark.asyncio
    async def test_workflow_execute_with_error(self):
        """测试工作流执行错误"""
        workflow = Workflow("test_workflow")

        async def success_step(coordinator):
            return {"status": "ok"}

        async def error_step(coordinator):
            raise ValueError("Step error")

        async def skipped_step(coordinator):
            return {"status": "should_not_run"}

        workflow.add_step(success_step)
        workflow.add_step(error_step)
        workflow.add_step(skipped_step)

        coordinator = AgentCoordinator()
        result = await workflow.execute(coordinator)

        # 第一步成功
        assert result["step_1"]["status"] == "ok"
        # 第二步失败
        assert "error" in result["step_2"]
        # 第三步应该被跳过
        assert "step_3" not in result


@pytest.mark.unit
class TestCoordinatorIntegration:
    """协调器集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_agents(self):
        """测试完整的Agent工作流"""
        coordinator = AgentCoordinator()

        # 创建并注册Agent
        agent1 = TestAgent("market_agent")
        agent2 = TestAgent("strategy_agent")
        agent3 = TestAgent("risk_agent")

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)
        coordinator.register_agent(agent3)

        # 启动所有Agent
        await coordinator.start_all()
        assert coordinator._is_running

        # 发送消息
        message = Message(
            type=MessageType.MARKET_DATA,
            sender="market_agent",
            receiver="strategy_agent",
            content={"price": 100.0}
        )

        await coordinator.route_message(message)

        # 验证消息被处理
        assert len(agent2.processed_messages) == 1

        # 停止所有Agent
        await coordinator.stop_all()
        assert not coordinator._is_running

    @pytest.mark.asyncio
    async def test_message_response_flow(self):
        """测试消息响应流程"""
        coordinator = AgentCoordinator()

        agent1 = TestAgent("agent1")
        agent2 = TestAgent("agent2")

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        # 发送从agent1到agent2的消息
        message = Message(
            type=MessageType.MARKET_DATA,
            sender="agent1",
            receiver="agent2",
            content={"query": "test"}
        )

        await coordinator.route_message(message)

        # agent2应该处理消息
        assert len(agent2.processed_messages) == 1

        # agent1也应该收到响应（如果实现了响应路由）
        # 这取决于实际实现

    @pytest.mark.asyncio
    async def test_sequential_message_routing(self):
        """测试顺序消息路由"""
        coordinator = AgentCoordinator()

        agents = [TestAgent(f"agent{i}") for i in range(5)]
        for agent in agents:
            coordinator.register_agent(agent)

        # 发送一系列消息
        for i in range(4):
            message = Message(
                type=MessageType.MARKET_DATA,
                sender=f"agent{i}",
                receiver=f"agent{i+1}",
                content={"sequence": i}
            )
            await coordinator.route_message(message)

        # 验证每个agent都收到了消息
        for i in range(1, 5):
            assert len(agents[i].processed_messages) == 1


@pytest.mark.unit
class TestCoordinatorEdgeCases:
    """协调器边界情况测试"""

    @pytest.mark.asyncio
    async def test_start_empty_coordinator(self):
        """测试启动空的协调器"""
        coordinator = AgentCoordinator()
        await coordinator.start_all()

        assert coordinator._is_running
        await coordinator.stop_all()

    @pytest.mark.asyncio
    async def test_broadcast_from_nonexistent_sender(self):
        """测试从不存在的发送者广播"""
        coordinator = AgentCoordinator()

        agent1 = TestAgent("agent1")
        coordinator.register_agent(agent1)

        # 从不存在的agent广播
        await coordinator.broadcast_message(
            sender="nonexistent",
            message_type=MessageType.DATA,
            content={"test": "data"}
        )

        # 应该仍然发送给agent1
        queue = coordinator.global_message_queue
        assert not queue.empty()

    @pytest.mark.asyncio
    async def test_workflow_with_no_steps(self):
        """测试没有步骤的工作流"""
        workflow = Workflow("empty_workflow")
        coordinator = AgentCoordinator()

        result = await workflow.execute(coordinator)
        assert result == {}

    @pytest.mark.asyncio
    async def test_agent_start_failure(self):
        """测试Agent启动失败"""
        class FailingAgent(Agent):
            def __init__(self, name):
                super().__init__(name)

            async def start(self):
                raise RuntimeError("Start failed")

            async def stop(self):
                pass

            async def process(self, message):
                pass

        coordinator = AgentCoordinator()
        agent = FailingAgent("failing_agent")
        coordinator.register_agent(agent)

        # 启动应该抛出异常
        with pytest.raises(RuntimeError):
            await coordinator.start_all()

    @pytest.mark.asyncio
    async def test_message_processing_error(self):
        """测试消息处理错误"""
        class ErrorAgent(Agent):
            def __init__(self, name):
                super().__init__(name)

            async def start(self):
                pass

            async def stop(self):
                pass

            async def process(self, message):
                raise ValueError("Processing failed")

        coordinator = AgentCoordinator()
        agent = ErrorAgent("error_agent")
        coordinator.register_agent(agent)

        test_message = Message(
            type=MessageType.MARKET_DATA,
            sender="test",
            receiver="error_agent",
            content={}
        )

        # 不应该抛出异常，应该记录错误
        await coordinator.route_message(test_message)

    def test_coordinator_multiple_registrations(self):
        """测试多次注册"""
        coordinator = AgentCoordinator()
        agent = TestAgent("agent1")

        coordinator.register_agent(agent)
        coordinator.register_agent(agent)
        coordinator.register_agent(agent)

        # 应该只有一个
        assert len(coordinator._agents) == 1

    @pytest.mark.asyncio
    async def test_workflow_step_receiving_coordinator(self):
        """测试工作流步骤接收协调器参数"""
        workflow = Workflow("test_workflow")
        coordinator = AgentCoordinator()

        received_coordinator = None

        async def test_step(coord):
            nonlocal received_coordinator
            received_coordinator = coord
            return {}

        workflow.add_step(test_step)
        await workflow.execute(coordinator)

        assert received_coordinator is coordinator


@pytest.mark.unit
class TestMessageFlow:
    """消息流测试"""

    @pytest.mark.asyncio
    async def test_message_queue_integration(self):
        """测试消息队列集成"""
        coordinator = AgentCoordinator()

        agent1 = TestAgent("agent1")
        agent2 = TestAgent("agent2")

        coordinator.register_agent(agent1)
        coordinator.register_agent(agent2)

        # 广播消息
        await coordinator.broadcast_message(
            sender="agent1",
            message_type=MessageType.DATA,
            content={"data": "test"}
        )

        # 检查全局队列
        queue = coordinator.global_message_queue
        assert not queue.empty()

    @pytest.mark.asyncio
    async def test_excluding_sender_in_broadcast(self):
        """测试广播时排除发送者"""
        coordinator = AgentCoordinator()

        agents = [TestAgent(f"agent{i}") for i in range(3)]
        for agent in agents:
            coordinator.register_agent(agent)

        # agent1广播，排除agent2
        await coordinator.broadcast_message(
            sender="agent1",
            message_type=MessageType.DATA,
            content={},
            exclude=["agent2"]
        )

        # 应该只有agent0和agent2收到
        queue = coordinator.global_message_queue
        messages = []
        while not queue.empty():
            messages.append(queue.receive())

        receivers = {msg.receiver for msg in messages}
        assert "agent1" not in receivers  # 发送者被排除
        assert "agent2" not in receivers  # 明确排除
        assert "agent0" in receivers


@pytest.mark.unit
class TestWorkflowAdvanced:
    """工作流高级测试"""

    @pytest.mark.asyncio
    async def test_workflow_data_passing(self):
        """测试工作流数据传递"""
        workflow = Workflow("data_passing_workflow")
        coordinator = AgentCoordinator()

        async def step1(coord):
            return {"value": 10}

        async def step2(coord):
            # 在实际应用中，可以访问之前步骤的结果
            return {"doubled": 20}

        workflow.add_step(step1)
        workflow.add_step(step2)

        result = await workflow.execute(coordinator)
        assert "step_1" in result
        assert "step_2" in result

    @pytest.mark.asyncio
    async def test_workflow_exception_recovery(self):
        """测试工作流异常恢复"""
        workflow = Workflow("recovery_workflow")
        coordinator = AgentCoordinator()

        attempt_count = 0

        async def retry_step(coord):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not yet")
            return {"success": True}

        workflow.add_step(retry_step)

        # 步骤会失败
        result = await workflow.execute(coordinator)
        assert "error" in result["step_1"]
        assert attempt_count == 1  # 不会自动重试

    def test_workflow_description(self):
        """测试工作流描述"""
        workflow = Workflow(
            "test_workflow",
            "This is a detailed description of the workflow"
        )

        assert workflow.description == "This is a detailed description of the workflow"

    @pytest.mark.asyncio
    async def test_workflow_with_coordinator_operations(self):
        """测试使用协调器操作的工作流"""
        workflow = Workflow("coordinator_ops_workflow")
        coordinator = AgentCoordinator()

        agent = TestAgent("test_agent")
        coordinator.register_agent(agent)

        async def register_agent_step(coord):
            # 在工作流中注册新agent
            new_agent = TestAgent("new_agent")
            coord.register_agent(new_agent)
            return {"registered": "new_agent"}

        async def list_agents_step(coord):
            agents = coord.list_agents()
            return {"agent_count": len(agents)}

        workflow.add_step(register_agent_step)
        workflow.add_step(list_agents_step)

        result = await workflow.execute(coordinator)
        assert result["step_1"]["registered"] == "new_agent"
        assert result["step_2"]["agent_count"] >= 2
