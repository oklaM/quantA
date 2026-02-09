"""
Agent协作模块
提供多Agent编排和协同功能
"""

from typing import Any, Dict, List, Optional

from agents.base.agent_base import Agent, Message, MessageType
from agents.base.coordinator import AgentCoordinator, Workflow
from utils.logging import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Agent编排器

    负责管理多个Agent的协作流程
    """

    def __init__(self, coordinator: Optional[AgentCoordinator] = None):
        """
        Args:
            coordinator: Agent协调器（可选，默认创建新的）
        """
        self.coordinator = coordinator or AgentCoordinator()
        self._workflows: Dict[str, Workflow] = {}

        logger.info("AgentOrchestrator初始化")

    def register_agent(self, agent: Agent):
        """
        注册Agent

        Args:
            agent: Agent实例
        """
        self.coordinator.register_agent(agent)

    def create_workflow(
        self,
        name: str,
        description: str = "",
        steps: Optional[List] = None,
    ) -> Workflow:
        """
        创建工作流

        Args:
            name: 工作流名称
            description: 描述
            steps: 步骤列表

        Returns:
            Workflow实例
        """
        workflow = Workflow(name, description)

        if steps:
            for step in steps:
                workflow.add_step(step)

        self._workflows[name] = workflow
        logger.info(f"创建工作流: {name}")

        return workflow

    async def run_workflow(
        self,
        workflow_name: str,
    ) -> Dict[str, Any]:
        """
        运行工作流

        Args:
            workflow_name: 工作流名称

        Returns:
            执行结果
        """
        workflow = self._workflows.get(workflow_name)

        if not workflow:
            logger.error(f"工作流不存在: {workflow_name}")
            return {"error": f"工作流不存在: {workflow_name}"}

        logger.info(f"运行工作流: {workflow_name}")

        return await self.coordinator.run_workflow(workflow)

    async def broadcast(
        self,
        sender: str,
        message_type: MessageType,
        content: Dict[str, Any],
        exclude: Optional[List[str]] = None,
    ):
        """
        广播消息

        Args:
            sender: 发送者
            message_type: 消息类型
            content: 消息内容
            exclude: 排除的Agent列表
        """
        await self.coordinator.broadcast_message(sender, message_type, content, exclude)

    async def send_message(
        self,
        message: Message,
    ):
        """
        发送消息

        Args:
            message: 消息
        """
        await self.coordinator.route_message(message)

    def list_agents(self) -> List[str]:
        """列出所有Agent"""
        return self.coordinator.list_agents()

    def list_workflows(self) -> List[str]:
        """列出所有工作流"""
        return list(self._workflows.keys())


__all__ = ["AgentOrchestrator"]
