"""
Agent协调器
管理多个Agent之间的通信和协作
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base.agent_base import (
    Agent,
    Message,
    MessageQueue,
    MessageType,
)
from utils.logging import get_logger

logger = get_logger(__name__)


class AgentCoordinator:
    """
    Agent协调器
    负责管理多个Agent并协调它们之间的通信
    """

    def __init__(self):
        """初始化协调器"""
        self._agents: Dict[str, Agent] = {}
        self.global_message_queue = MessageQueue()
        self._is_running = False

        logger.info("Agent协调器初始化")

    def register_agent(self, agent: Agent):
        """
        注册Agent

        Args:
            agent: Agent实例
        """
        if agent.name in self._agents:
            logger.warning(f"Agent已存在: {agent.name}")
            return

        self._agents[agent.name] = agent
        logger.info(f"注册Agent: {agent.name}")

    def unregister_agent(self, agent_name: str):
        """
        注销Agent

        Args:
            agent_name: Agent名称
        """
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info(f"注销Agent: {agent_name}")

    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """获取Agent"""
        return self._agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """列出所有Agent"""
        return list(self._agents.keys())

    async def start_all(self):
        """启动所有Agent"""
        logger.info("启动所有Agent...")

        tasks = []
        for agent in self._agents.values():
            tasks.append(agent.start())

        await asyncio.gather(*tasks)

        self._is_running = True
        logger.info(f"已启动{len(self._agents)}个Agent")

    async def stop_all(self):
        """停止所有Agent"""
        logger.info("停止所有Agent...")

        tasks = []
        for agent in self._agents.values():
            tasks.append(agent.stop())

        await asyncio.gather(*tasks)

        self._is_running = False
        logger.info("所有Agent已停止")

    async def broadcast_message(
        self,
        sender: str,
        message_type: MessageType,
        content: Dict[str, Any],
        exclude: Optional[List[str]] = None,
    ):
        """
        广播消息给所有Agent

        Args:
            sender: 发送者
            message_type: 消息类型
            content: 消息内容
            exclude: 排除的Agent列表
        """
        exclude = exclude or []

        for agent_name in self._agents:
            if agent_name != sender and agent_name not in exclude:
                message = Message(
                    type=message_type,
                    sender=sender,
                    receiver=agent_name,
                    content=content,
                )

                self.global_message_queue.send(message)

        logger.debug(f"广播消息: {sender} -> {len(self._agents) - len(exclude)}个Agent")

    async def route_message(self, message: Message):
        """
        路由消息到目标Agent

        Args:
            message: 消息
        """
        target_agent = self.get_agent(message.receiver)

        if target_agent is None:
            logger.error(f"目标Agent不存在: {message.receiver}")
            return

        # 将消息放入目标Agent的队列
        target_agent.message_queue.send(message)

        # 触发Agent处理
        await self._process_agent_message(target_agent, message)

    async def _process_agent_message(self, agent: Agent, message: Message):
        """
        处理Agent消息

        Args:
            agent: 目标Agent
            message: 消息
        """
        try:
            response = await agent.process(message)

            if response:
                # 路由响应消息
                await self.route_message(response)

        except Exception as e:
            logger.error(f"处理消息失败 ({agent.name}): {e}", exc_info=True)

            # 发送错误消息
            error_message = Message(
                type=MessageType.ERROR,
                sender=agent.name,
                receiver=message.sender,
                content={
                    "error": str(e),
                    "original_message_id": message.message_id,
                },
                reply_to=message.message_id,
            )

            await self.route_message(error_message)

    async def run_workflow(
        self,
        workflow: "Workflow",
    ) -> Dict[str, Any]:
        """
        运行工作流

        Args:
            workflow: 工作流实例

        Returns:
            工作流结果
        """
        logger.info(f"运行工作流: {workflow.name}")

        results = {}

        try:
            # 执行工作流
            results = await workflow.execute(self)

        except Exception as e:
            logger.error(f"工作流执行失败: {e}", exc_info=True)
            results["error"] = str(e)

        return results


class Workflow:
    """
    工作流类
    定义Agent之间的协作流程
    """

    def __init__(
        self,
        name: str,
        description: str = "",
    ):
        """
        Args:
            name: 工作流名称
            description: 描述
        """
        self.name = name
        self.description = description
        self._steps: List[callable] = []

    def add_step(self, step: callable):
        """
        添加步骤

        Args:
            step: 步骤函数（接收coordinator作为参数）
        """
        self._steps.append(step)
        logger.debug(f"工作流添加步骤: {self.name} - {len(self._steps)}")

    async def execute(self, coordinator: AgentCoordinator) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            coordinator: 协调器

        Returns:
            执行结果
        """
        logger.info(f"开始执行工作流: {self.name}")

        results = {}

        for i, step in enumerate(self._steps):
            logger.debug(f"执行步骤 {i + 1}/{len(self._steps)}")

            try:
                step_result = await step(coordinator)
                results[f"step_{i + 1}"] = step_result

            except Exception as e:
                logger.error(f"步骤执行失败: {e}", exc_info=True)
                results[f"step_{i + 1}"] = {"error": str(e)}
                break

        logger.info(f"工作流执行完成: {self.name}")
        return results


__all__ = [
    "AgentCoordinator",
    "Workflow",
]
