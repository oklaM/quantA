"""
LLM Agent基类和消息协议
定义Agent的基础接口和通信机制
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """消息类型"""

    # 数据类
    MARKET_DATA = "market_data"  # 市场数据
    TECHNICAL_DATA = "technical_data"  # 技术指标数据
    FUNDAMENTAL_DATA = "fundamental_data"  # 基本面数据
    SENTIMENT_DATA = "sentiment_data"  # 情绪数据

    # 分析类
    ANALYSIS_REQUEST = "analysis_request"  # 分析请求
    ANALYSIS_RESPONSE = "analysis_response"  # 分析响应

    # 决策类
    SIGNAL_REQUEST = "signal_request"  # 信号请求
    SIGNAL_RESPONSE = "signal_response"  # 信号响应

    # 风控类
    RISK_CHECK_REQUEST = "risk_check_request"  # 风控检查请求
    RISK_CHECK_RESPONSE = "risk_check_response"  # 风控检查响应

    # 通用
    ERROR = "error"  # 错误
    ACK = "ack"  # 确认


@dataclass
class Message:
    """Agent消息"""

    type: MessageType
    sender: str  # 发送者
    receiver: str  # 接收者
    content: Dict[str, Any]  # 消息内容
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None  # 消息ID
    reply_to: Optional[str] = None  # 回复的消息ID

    def __post_init__(self):
        """初始化后生成消息ID"""
        if self.message_id is None:
            self.message_id = f"msg_{datetime.now().timestamp()}_{id(self)}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典创建"""
        return cls(
            type=MessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data.get("message_id"),
            reply_to=data.get("reply_to"),
        )

    def __repr__(self):
        return f"Message({self.type.value}, {self.sender} -> {self.receiver})"


class MessageQueue:
    """消息队列"""

    def __init__(self):
        self._messages: List[Message] = []
        self._message_history: Dict[str, Message] = {}

    def send(self, message: Message):
        """发送消息"""
        if message.message_id is None:
            message.message_id = f"msg_{datetime.now().timestamp()}"

        self._messages.append(message)
        self._message_history[message.message_id] = message

        logger.debug(
            f"消息发送: {message.sender} -> {message.receiver}, " f"类型={message.type.value}"
        )

    def receive(
        self,
        agent_name: str,
        message_type: Optional[MessageType] = None,
    ) -> Optional[Message]:
        """接收消息"""
        for i, msg in enumerate(self._messages):
            if msg.receiver == agent_name:
                if message_type is None or msg.type == message_type:
                    self._messages.pop(i)
                    return msg

        return None

    def get_by_id(self, message_id: str) -> Optional[Message]:
        """根据ID获取消息"""
        return self._message_history.get(message_id)

    def size(self) -> int:
        """获取队列大小"""
        return len(self._messages)

    def clear(self):
        """清空队列"""
        self._messages.clear()


class Agent(ABC):
    """
    Agent基类
    所有LLM Agent都应继承此类
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            name: Agent名称
            description: Agent描述
            llm_config: LLM配置
        """
        self.name = name
        self.description = description
        self.llm_config = llm_config or {}

        # 消息队列
        self.message_queue = MessageQueue()

        # 状态
        self._is_running = False

        # 工具
        self._tools: Dict[str, Callable] = {}

        logger.info(f"Agent初始化: {self.name}")

    @abstractmethod
    async def process(self, message: Message) -> Optional[Message]:
        """
        处理消息

        Args:
            message: 接收到的消息

        Returns:
            响应消息（可选）
        """
        pass

    async def start(self):
        """启动Agent"""
        self._is_running = True
        logger.info(f"Agent启动: {self.name}")

    async def stop(self):
        """停止Agent"""
        self._is_running = False
        logger.info(f"Agent停止: {self.name}")

    def register_tool(self, name: str, func: Callable):
        """注册工具"""
        self._tools[name] = func
        logger.debug(f"注册工具: {self.name}.{name}")

    def get_tools(self) -> Dict[str, Callable]:
        """获取所有工具"""
        return self._tools.copy()

    def send_message(
        self,
        receiver: str,
        message_type: MessageType,
        content: Dict[str, Any],
        reply_to: Optional[str] = None,
    ) -> str:
        """
        发送消息

        Args:
            receiver: 接收者
            message_type: 消息类型
            content: 消息内容
            reply_to: 回复的消息ID

        Returns:
            消息ID
        """
        message = Message(
            type=message_type,
            sender=self.name,
            receiver=receiver,
            content=content,
            reply_to=reply_to,
        )

        self.message_queue.send(message)
        return message.message_id

    def receive_message(
        self,
        message_type: Optional[MessageType] = None,
    ) -> Optional[Message]:
        """
        接收消息

        Args:
            message_type: 消息类型（None表示任意类型）

        Returns:
            消息或None
        """
        return self.message_queue.receive(self.name, message_type)


class LLMAgent(Agent):
    """
    LLM Agent基类
    提供LLM调用能力
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, description, llm_config)

        # LLM客户端（延迟初始化）
        self._llm_client = None

        # 提示词模板
        self._system_prompt = self._get_system_prompt()

        # GLM-4混合类支持
        self._use_glm4 = llm_config.get("use_glm4", True) if llm_config else True

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return f"你是一个专业的量化交易AI助手，名为{self.name}。{self.description}"

    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        调用LLM

        Args:
            prompt: 提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他参数

        Returns:
            LLM响应
        """
        if self._use_glm4:
            return await self._call_glm4(prompt, system_prompt, **kwargs)
        else:
            logger.warning(f"LLM调用未实现: {self.name}")
            return "模拟LLM响应"

    async def _call_glm4(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        调用GLM-4

        Args:
            prompt: 提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            GLM-4响应
        """
        try:
            from agents.base.glm4_integration import GLM4Client

            if self._llm_client is None:
                self._llm_client = GLM4Client()

            messages = []
            if system_prompt or self._system_prompt:
                messages.append({"role": "system", "content": system_prompt or self._system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = await self._llm_client.chat(messages, **kwargs)
            return response

        except ImportError:
            logger.error("zhipuai未安装，请运行: pip install zhipuai")
            return "错误：zhipuai未安装"
        except Exception as e:
            logger.error(f"GLM-4调用失败: {e}")
            return f"错误：{str(e)}"

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self._system_prompt = prompt


class Tool:
    """
    Agent工具装饰器
    用于将函数注册为Agent工具
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    def __call__(self, func: Callable) -> Callable:
        """装饰器"""
        func._tool_name = self.name
        func._tool_description = self.description
        func._is_tool = True
        return func


def tool(name: str, description: str = ""):
    """
    工具装饰器函数

    Args:
        name: 工具名称
        description: 工具描述

    Returns:
        装饰器
    """
    return Tool(name, description)


__all__ = [
    "MessageType",
    "Message",
    "MessageQueue",
    "Agent",
    "AgentBase",  # Alias for Agent
    "LLMAgent",
    "AgentResponse",  # Alias for Message (for backward compatibility)
    "Tool",
    "tool",
]


# Backward compatibility aliases
AgentBase = Agent  # Alias for tests that expect AgentBase
AgentResponse = Message  # Alias for tests that expect AgentResponse
