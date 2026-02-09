"""
测试Agent模板基类
"""

import pytest

from agents.base.agent_base import Message, MessageType
from agents.base.agent_template import TemplateAgent


class DummyTemplateAgent(TemplateAgent):
    """测试用简单Agent"""

    agent_name = "dummy_agent"
    agent_description = "测试Agent"

    def _get_system_prompt(self) -> str:
        return "你是一个测试Agent。"


class TestTemplateAgent:
    """测试TemplateAgent基类"""

    @pytest.mark.unit
    def test_initialization(self):
        """测试初始化"""
        agent = DummyTemplateAgent()

        assert agent.name == "dummy_agent"
        assert agent.description == "测试Agent"
        assert agent._system_prompt == "你是一个测试Agent。"

    @pytest.mark.unit
    def test_class_attributes(self):
        """测试类属性"""
        assert DummyTemplateAgent.agent_name == "dummy_agent"
        assert DummyTemplateAgent.agent_description == "测试Agent"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_unknown_message(self):
        """测试处理未知消息"""
        agent = DummyTemplateAgent()

        message = Message(
            type=MessageType.MARKET_DATA,
            sender="coordinator",
            receiver="dummy_agent",
            content={"test": "data"},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE
        assert "error" in response.content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_process_analysis_request(self):
        """测试处理分析请求"""
        agent = DummyTemplateAgent()

        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="coordinator",
            receiver="dummy_agent",
            content={"task": "test"},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE
        assert response.sender == "dummy_agent"
        assert response.receiver == "coordinator"
        assert "error" in response.content

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """测试错误处理"""
        class ErrorAgent(TemplateAgent):
            agent_name = "error_agent"
            agent_description = "错误测试Agent"

            def _get_system_prompt(self) -> str:
                return "测试错误处理"

            async def _handle_analysis_request(self, content):
                raise ValueError("测试错误")

        agent = ErrorAgent()

        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="coordinator",
            receiver="error_agent",
            content={"task": "test"},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.ERROR
        assert "error" in response.content

    @pytest.mark.unit
    def test_helper_methods(self):
        """测试辅助方法"""
        agent = DummyTemplateAgent()

        # 测试提取任务
        content = {"task": "analyze", "data": "test_data"}
        assert agent._extract_task(content) == "analyze"
        assert agent._extract_data(content) == "test_data"

        # 测试构建成功响应
        success = agent._build_success_response(
            data={"result": "ok"},
            message="成功"
        )
        assert success["status"] == "success"
        assert success["message"] == "成功"
        assert success["data"] == {"result": "ok"}

        # 测试构建错误响应
        error = agent._build_error_response_dict(
            error_message="失败",
            error_code="ERR_001"
        )
        assert error["error"] == "失败"
        assert error["error_code"] == "ERR_001"
        assert error["status"] == "error"


class CustomAnalysisAgent(TemplateAgent):
    """自定义分析Agent"""

    agent_name = "custom_analysis_agent"
    agent_description = "自定义分析Agent"

    def _get_system_prompt(self) -> str:
        return "你是自定义分析Agent。"

    async def _handle_analysis_request(self, content):
        """处理分析请求"""
        task = self._extract_task(content)

        if task == "analyze":
            data = self._extract_data(content)
            return self._build_success_response(
                data={"analysis": f"分析了: {data}"},
                message="分析完成"
            )
        else:
            return self._build_error_response_dict(f"未知任务: {task}")


class TestCustomAgent:
    """测试自定义Agent"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_analysis_handler(self):
        """测试自定义分析处理"""
        agent = CustomAnalysisAgent()

        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="coordinator",
            receiver="custom_analysis_agent",
            content={"task": "analyze", "data": "test_data"},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE
        assert response.content["status"] == "success"
        assert response.content["data"]["analysis"] == "分析了: test_data"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_custom_analysis_error(self):
        """测试自定义分析错误"""
        agent = CustomAnalysisAgent()

        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="coordinator",
            receiver="custom_analysis_agent",
            content={"task": "unknown", "data": "test_data"},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.ANALYSIS_RESPONSE
        assert "error" in response.content


class MultiTypeAgent(TemplateAgent):
    """多消息类型Agent"""

    agent_name = "multi_type_agent"
    agent_description = "多消息类型Agent"

    def _get_system_prompt(self) -> str:
        return "你是多类型Agent。"

    async def _handle_analysis_request(self, content):
        """处理分析请求"""
        return self._build_success_response(
            data={"type": "analysis"},
            message="分析完成"
        )

    async def _handle_signal_request(self, content):
        """处理信号请求"""
        return self._build_success_response(
            data={"type": "signal", "signal": "buy"},
            message="信号生成完成"
        )


class TestMultiTypeAgent:
    """测试多消息类型Agent"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_analysis_request(self):
        """测试分析请求"""
        agent = MultiTypeAgent()

        message = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="coordinator",
            receiver="multi_type_agent",
            content={},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.content["data"]["type"] == "analysis"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_signal_request(self):
        """测试信号请求"""
        agent = MultiTypeAgent()

        message = Message(
            type=MessageType.SIGNAL_REQUEST,
            sender="coordinator",
            receiver="multi_type_agent",
            content={},
        )

        response = await agent.process(message)

        assert response is not None
        assert response.type == MessageType.SIGNAL_RESPONSE
        assert response.content["data"]["type"] == "signal"
