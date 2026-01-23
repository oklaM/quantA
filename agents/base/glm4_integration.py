"""
GLM-4模型集成
集成智谱AI的GLM-4模型到Agent系统
"""

import json
import os
from typing import Any, Dict, List, Optional

try:
    from zhipuai import ZhipuAI

    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZHIPUAI_AVAILABLE = False
    ZhipuAI = None

from config.settings import llm as llm_config
from utils.logging import get_logger

logger = get_logger(__name__)


class GLM4Client:
    """
    GLM-4客户端
    封装智谱AI API调用
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "glm-4-plus",
    ):
        """
        Args:
            api_key: API密钥
            model: 模型名称
        """
        if not ZHIPUAI_AVAILABLE:
            raise ImportError("zhipuai未安装，请运行: pip install zhipuai")

        self.api_key = api_key or llm_config.API_KEY
        if not self.api_key:
            raise ValueError("GLM-4 API密钥未设置")

        self.model = model
        self.client = ZhipuAI(api_key=self.api_key)

        logger.info(f"GLM-4客户端初始化: {model}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        聊天对话

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            响应文本
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"GLM-4调用失败: {e}")
            raise

    async def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        带工具的聊天

        Args:
            messages: 消息列表
            tools: 工具列表

        Returns:
            响应结果
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )

            return response.choices[0].message

        except Exception as e:
            logger.error(f"GLM-4工具调用失败: {e}")
            raise

    def format_tools(self, tools: Dict[str, callable]) -> List[Dict[str, Any]]:
        """
        格式化工具为GLM-4格式

        Args:
            tools: {name: func} 格式的工具字典

        Returns:
            GLM-4格式的工具列表
        """
        formatted_tools = []

        for name, func in tools.items():
            if hasattr(func, "_is_tool") and func._is_tool:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": getattr(func, "_tool_description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }

                formatted_tools.append(tool_def)

        return formatted_tools


class GLMAgentMixin:
    """
    GLM Agent混合类
    为Agent提供GLM-4调用能力
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._glm_client: Optional[GLM4Client] = None

    def init_glm4(self, model: str = "glm-4-plus"):
        """初始化GLM-4客户端"""
        try:
            self._glm_client = GLM4Client(model=model)
            logger.info(f"{self.name}集成GLM-4成功")
        except Exception as e:
            logger.error(f"{self.name}集成GLM-4失败: {e}")

    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        调用LLM

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            LLM响应
        """
        if self._glm_client is None:
            self.init_glm4()

        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await self._glm_client.chat(messages, **kwargs)

        return response

    async def call_llm_with_tools(
        self,
        prompt: str,
        tools: Dict[str, callable],
        **kwargs,
    ) -> Any:
        """
        调用带工具的LLM

        Args:
            prompt: 提示词
            tools: 工具字典

        Returns:
            LLM响应（可能包含工具调用）
        """
        if self._glm_client is None:
            self.init_glm4()

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted_tools = self._glm_client.format_tools(tools)

        response = await self._glm_client.chat_with_tools(
            messages=messages,
            tools=formatted_tools,
            **kwargs,
        )

        # 处理工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name in tools:
                    try:
                        result = tools[function_name](**function_args)
                        logger.debug(f"工具调用成功: {function_name}")
                        return result
                    except Exception as e:
                        logger.error(f"工具调用失败: {function_name}: {e}")

        # 返回文本响应
        return response.content


# 测试函数
async def test_glm4_integration():
    """测试GLM-4集成"""
    logger.info("测试GLM-4集成...")

    try:
        client = GLM4Client()

        # 简单对话测试
        messages = [
            {"role": "user", "content": "你好，请介绍一下你自己。"},
        ]

        response = await client.chat(messages)

        logger.info(f"GLM-4响应: {response}")
        return True

    except Exception as e:
        logger.error(f"GLM-4测试失败: {e}")
        return False


def create_glm_agent(
    name: str,
    system_prompt: str,
    tools: Optional[Dict[str, callable]] = None,
) -> GLMAgentMixin:
    """
    创建带GLM-4的Agent

    Args:
        name: Agent名称
        system_prompt: 系统提示词
        tools: 工具字典

    Returns:
        Agent实例
    """
    from agents.base.agent_base import LLMAgent

    class GLMAgent(LLMAgent, GLMAgentMixin):
        def __init__(self):
            LLMAgent.__init__(
                self,
                name=name,
                description="GLM-4驱动的AI Agent",
            )
            GLMAgentMixin.__init__(self)
            self._system_prompt = system_prompt
            self._tools = tools or {}

    agent = GLMAgent()
    agent.init_glm4()

    return agent


__all__ = [
    "GLM4Client",
    "GLMAgentMixin",
    "test_glm4_integration",
    "create_glm_agent",
]
