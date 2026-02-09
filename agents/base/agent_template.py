"""
Agent模板基类
使用模板方法模式消除子类重复代码
"""

from abc import abstractmethod
from typing import Any, Callable, Dict, Optional

from agents.base.agent_base import LLMAgent, Message, MessageType
from utils.logging import get_logger

logger = get_logger(__name__)


class TemplateAgent(LLMAgent):
    """
    Agent模板基类

    使用模板方法模式，子类只需：
    1. 定义类属性：agent_name, agent_description
    2. 实现_get_system_prompt()方法
    3. 实现_register_tools()方法（可选）
    4. 实现_handle_*_request()方法（可选）

    模板自动处理：
    - 统一的初始化流程
    - 统一的消息路由和处理
    - 统一的错误处理
    - 统一的响应构建
    """

    # 类属性：子类必须覆盖
    agent_name: str = "base_template_agent"
    agent_description: str = "基础模板Agent"

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        """
        初始化Agent

        Args:
            llm_config: LLM配置
        """
        # 使用类属性初始化父类
        super().__init__(
            name=self.agent_name,
            description=self.agent_description,
            llm_config=llm_config,
        )

        # 注册工具（子类可重写）
        self._register_tools()

        logger.info(f"{self.agent_name}初始化完成")

    def _register_tools(self):
        """
        注册Agent工具

        子类可重写此方法来注册自定义工具
        默认实现：不注册任何工具
        """
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        获取系统提示词

        子类必须实现此方法

        Returns:
            系统提示词字符串
        """
        raise NotImplementedError(f"{self.__class__.__name__}必须实现_get_system_prompt()方法")

    async def process(self, message: Message) -> Optional[Message]:
        """
        处理消息（模板方法）

        这是模板方法，定义了消息处理的标准流程：
        1. 路由消息到对应的处理方法
        2. 执行处理逻辑
        3. 构建响应
        4. 错误处理

        Args:
            message: 接收到的消息

        Returns:
            响应消息
        """
        logger.info(f"{self.agent_name}处理消息: {message.type.value}")

        try:
            # 步骤1：路由消息
            handler = self._route_message(message)

            # 步骤2：执行处理
            if handler is not None:
                result = await handler(message.content)
            else:
                result = await self._handle_unknown_message(message.content)

            # 步骤3：构建响应
            response = self._build_response(message, result)

            return response

        except Exception as e:
            # 步骤4：错误处理
            logger.error(f"{self.agent_name}处理消息失败: {e}", exc_info=True)
            return self._build_error_response(message, str(e))

    def _route_message(self, message: Message) -> Optional[Callable]:
        """
        路由消息到对应的处理方法

        根据消息类型，路由到对应的_handle_*_request方法
        子类可以通过重写此方法来自定义路由逻辑

        Args:
            message: 消息对象

        Returns:
            处理方法或None
        """
        # 默认路由规则
        if message.type == MessageType.ANALYSIS_REQUEST:
            return self._handle_analysis_request
        elif message.type == MessageType.SIGNAL_REQUEST:
            return self._handle_signal_request
        elif message.type == MessageType.RISK_CHECK_REQUEST:
            return self._handle_risk_check_request
        else:
            return None

    def _build_response(
        self,
        original_message: Message,
        result: Dict[str, Any],
    ) -> Message:
        """
        构建响应消息

        Args:
            original_message: 原始消息
            result: 处理结果

        Returns:
            响应消息
        """
        # 根据原始消息类型确定响应类型
        response_type = self._get_response_type(original_message.type)

        return Message(
            type=response_type,
            sender=self.agent_name,
            receiver=original_message.sender,
            content=result,
            reply_to=original_message.message_id,
        )

    def _build_error_response(
        self,
        original_message: Message,
        error_message: str,
    ) -> Message:
        """
        构建错误响应消息

        Args:
            original_message: 原始消息
            error_message: 错误信息

        Returns:
            错误响应消息
        """
        return Message(
            type=MessageType.ERROR,
            sender=self.agent_name,
            receiver=original_message.sender,
            content={"error": error_message},
            reply_to=original_message.message_id,
        )

    def _get_response_type(self, request_type: MessageType) -> MessageType:
        """
        根据请求类型获取响应类型

        Args:
            request_type: 请求消息类型

        Returns:
            响应消息类型
        """
        # 默认映射规则
        response_mapping = {
            MessageType.ANALYSIS_REQUEST: MessageType.ANALYSIS_RESPONSE,
            MessageType.SIGNAL_REQUEST: MessageType.SIGNAL_RESPONSE,
            MessageType.RISK_CHECK_REQUEST: MessageType.RISK_CHECK_RESPONSE,
        }

        return response_mapping.get(request_type, MessageType.ANALYSIS_RESPONSE)

    # ==================== 消息处理方法（子类可重写） ====================

    async def _handle_analysis_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理分析请求

        默认实现：返回不支持的错误
        子类应重写此方法以实现具体的分析逻辑

        Args:
            content: 消息内容

        Returns:
            处理结果
        """
        return await self._handle_unknown_message(content)

    async def _handle_signal_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理信号请求

        默认实现：返回不支持的错误
        子类应重写此方法以实现具体的信号生成逻辑

        Args:
            content: 消息内容

        Returns:
            处理结果
        """
        return await self._handle_unknown_message(content)

    async def _handle_risk_check_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理风控检查请求

        默认实现：返回不支持的错误
        子类应重写此方法以实现具体的风控逻辑

        Args:
            content: 消息内容

        Returns:
            处理结果
        """
        return await self._handle_unknown_message(content)

    async def _handle_unknown_message(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理未知类型的消息

        Args:
            content: 消息内容

        Returns:
            错误信息
        """
        return {
            "error": f"不支持的消息类型或任务",
            "received_content": content,
            "agent": self.agent_name,
        }

    # ==================== 辅助方法 ====================

    def _extract_task(self, content: Dict[str, Any]) -> str:
        """
        从消息内容中提取任务类型

        Args:
            content: 消息内容

        Returns:
            任务类型字符串
        """
        return content.get("task", "")

    def _extract_data(self, content: Dict[str, Any]) -> Any:
        """
        从消息内容中提取数据

        Args:
            content: 消息内容

        Returns:
            数据对象
        """
        return content.get("data")

    def _build_success_response(
        self,
        data: Any = None,
        message: str = "success",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        构建成功响应

        Args:
            data: 响应数据
            message: 响应消息
            **kwargs: 其他字段

        Returns:
            响应字典
        """
        response = {"status": "success", "message": message}
        if data is not None:
            response["data"] = data
        response.update(kwargs)
        return response

    def _build_error_response_dict(
        self,
        error_message: str,
        error_code: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        构建错误响应字典

        Args:
            error_message: 错误信息
            error_code: 错误代码
            **kwargs: 其他字段

        Returns:
            错误响应字典
        """
        response = {"error": error_message, "status": "error"}
        if error_code is not None:
            response["error_code"] = error_code
        response.update(kwargs)
        return response


__all__ = ["TemplateAgent"]
