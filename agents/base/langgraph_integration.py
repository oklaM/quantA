"""
LangGraph Agent编排
使用LangGraph框架实现Agent工作流编排
"""

import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

# LangGraph相关导入
try:
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from agents.base.agent_base import Agent, Message, MessageType
from utils.logging import get_logger

logger = get_logger(__name__)


if LANGGRAPH_AVAILABLE:

    class AgentState(TypedDict):
        """Agent状态"""

        messages: Annotated[List[Message], operator.add]
        current_data: Dict[str, Any]
        analysis_results: Dict[str, Any]
        trading_signals: List[Dict[str, Any]]
        risk_assessment: Dict[str, Any]
        next_step: str

    class LangGraphAgentOrchestrator:
        """
        LangGraph Agent编排器
        使用状态图管理Agent工作流
        """

        def __init__(self):
            """初始化编排器"""
            if not LANGGRAPH_AVAILABLE:
                raise ImportError("langgraph未安装，请运行: pip install langgraph")

            self.graph = None
            self.agents: Dict[str, Agent] = {}
            self._workflow_name = "TradingWorkflow"

            logger.info("LangGraph编排器初始化")

        def register_agent(self, agent: Agent):
            """注册Agent"""
            self.agents[agent.name] = agent
            logger.info(f"注册Agent: {agent.name}")

        def build_workflow(self, workflow_config: Dict[str, List[str]]):
            """
            构建工作流

            Args:
                workflow_config: 工作流配置
                    {
                        'start': ['agent1', 'agent2'],
                        'agent1': ['agent3'],
                        'agent2': ['agent3'],
                        'agent3': ['end']
                    }
            """
            logger.info("构建LangGraph工作流...")

            # 创建状态图
            workflow = StateGraph(AgentState)

            # 添加节点（每个Agent作为一个节点）
            for agent_name, agent in self.agents.items():
                workflow.add_node(agent_name, self._create_agent_node(agent))

            # 设置入口点
            start_agents = workflow_config.get("start", [])
            if start_agents:
                # 如果有多个起始Agent，使用一个统一的start节点
                workflow.set_entry_point("start")
                workflow.add_node("start", self._create_start_node(start_agents))
                for agent in start_agents:
                    workflow.add_edge("start", agent)
            elif self.agents:
                # 否则使用第一个Agent作为入口
                first_agent = list(self.agents.keys())[0]
                workflow.set_entry_point(first_agent)

            # 添加边（定义Agent之间的转换）
            for from_node, to_nodes in workflow_config.items():
                if from_node == "start":
                    continue

                for to_node in to_nodes:
                    if to_node == "end":
                        workflow.add_edge(from_node, END)
                    elif to_node in self.agents:
                        workflow.add_edge(from_node, to_node)

            # 编译图
            self.graph = workflow.compile()

            logger.info(f"工作流构建完成: {len(self.agents)}个Agent节点")

        def _create_start_node(self, start_agents: List[str]):
            """创建起始节点"""

            async def start_node(state: AgentState) -> AgentState:
                """起始节点函数"""
                logger.info(f"工作流开始，启动Agent: {start_agents}")

                # 初始化状态
                if "messages" not in state:
                    state["messages"] = []
                if "current_data" not in state:
                    state["current_data"] = {}
                if "analysis_results" not in state:
                    state["analysis_results"] = {}
                if "trading_signals" not in state:
                    state["trading_signals"] = []
                if "risk_assessment" not in state:
                    state["risk_assessment"] = {}

                return state

            return start_node

        def _create_agent_node(self, agent: Agent):
            """创建Agent节点"""

            async def agent_node(state: AgentState) -> AgentState:
                """Agent节点函数"""
                logger.debug(f"执行Agent节点: {agent.name}")

                # 从状态中提取消息
                if state["messages"]:
                    last_message = state["messages"][-1]

                    # 创建Agent能处理的消息格式
                    agent_message = Message(
                        type=MessageType(
                            last_message.get("type", MessageType.ANALYSIS_REQUEST.value)
                        ),
                        sender=last_message.get("sender", "system"),
                        receiver=agent.name,
                        content=last_message.get("content", {}),
                    )

                    # 处理消息
                    response = await agent.process(agent_message)

                    if response:
                        state["messages"].append(
                            {
                                "type": response.type.value,
                                "sender": response.sender,
                                "receiver": response.receiver,
                                "content": response.content,
                                "timestamp": response.timestamp.isoformat(),
                            }
                        )

                        # 更新状态
                        if response.type == MessageType.ANALYSIS_RESPONSE:
                            state["analysis_results"][agent.name] = response.content
                        elif response.type == MessageType.SIGNAL_RESPONSE:
                            state["trading_signals"].append(response.content)
                        elif response.type == MessageType.RISK_CHECK_RESPONSE:
                            state["risk_assessment"][agent.name] = response.content

                return state

            return agent_node

        async def run(
            self,
            initial_state: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """
            运行工作流

            Args:
                initial_state: 初始状态

            Returns:
                最终状态
            """
            if self.graph is None:
                raise ValueError("工作流未构建，请先调用build_workflow()")

            logger.info("运行LangGraph工作流...")

            # 准备初始状态
            if initial_state is None:
                initial_state = {}

            state = AgentState(
                messages=initial_state.get("messages", []),
                current_data=initial_state.get("current_data", {}),
                analysis_results=initial_state.get("analysis_results", {}),
                trading_signals=initial_state.get("trading_signals", []),
                risk_assessment=initial_state.get("risk_assessment", {}),
                next_step=initial_state.get("next_step", ""),
            )

            # 运行图
            result = await self.graph.ainvoke(state)

            logger.info("工作流执行完成")
            return result

        def visualize(self, output_path: str):
            """可视化工作流"""
            try:
                from IPython.display import Image, display

                if self.graph:
                    img_data = self.graph.get_graph().draw_mermaid_png()

                    with open(output_path, "wb") as f:
                        f.write(img_data)

                    logger.info(f"工作流图已保存到: {output_path}")

            except Exception as e:
                logger.error(f"可视化失败: {e}")


class SimpleAgentWorkflow:
    """
    简化版Agent工作流
    不依赖LangGraph的基本实现
    """

    def __init__(self):
        """初始化工作流"""
        self.agents: Dict[str, Agent] = {}
        self.workflow_steps: List[Dict[str, Any]] = []
        logger.info("简单Agent工作流初始化")

    def add_agent(self, agent: Agent):
        """添加Agent"""
        self.agents[agent.name] = agent
        logger.debug(f"添加Agent: {agent.name}")

    def define_workflow(
        self,
        steps: List[Dict[str, Any]],
    ):
        """
        定义工作流步骤

        Args:
            steps: 步骤列表
                [
                    {'agent': 'agent_name', 'action': 'analyze', 'input': {...}},
                    {'agent': 'agent_name', 'action': 'generate_signal', 'input_from': 'previous'},
                    ...
                ]
        """
        self.workflow_steps = steps
        logger.info(f"定义工作流: {len(steps)}个步骤")

    async def execute(
        self,
        initial_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行工作流

        Args:
            initial_input: 初始输入

        Returns:
            执行结果
        """
        logger.info("执行Agent工作流...")

        results = {
            "input": initial_input,
            "steps": [],
            "final_output": {},
        }

        current_data = initial_input.copy()

        for i, step in enumerate(self.workflow_steps):
            agent_name = step.get("agent")
            action = step.get("action")
            step_input = step.get("input", current_data)

            logger.info(f"执行步骤 {i + 1}: {agent_name}.{action}")

            agent = self.agents.get(agent_name)
            if not agent:
                logger.error(f"Agent不存在: {agent_name}")
                continue

            # 创建消息
            message = Message(
                type=MessageType.ANALYSIS_REQUEST,
                sender="workflow",
                receiver=agent_name,
                content=step_input,
            )

            # 处理消息
            try:
                response = await agent.process(message)

                if response:
                    current_data = response.content
                    results["steps"].append(
                        {
                            "step": i + 1,
                            "agent": agent_name,
                            "action": action,
                            "result": response.content,
                        }
                    )

            except Exception as e:
                logger.error(f"步骤执行失败: {e}", exc_info=True)
                results["steps"].append(
                    {
                        "step": i + 1,
                        "agent": agent_name,
                        "action": action,
                        "error": str(e),
                    }
                )
                break

        results["final_output"] = current_data
        logger.info("工作流执行完成")

        return results


def create_trading_workflow() -> SimpleAgentWorkflow:
    """
    创建标准交易工作流

    Returns:
        工作流实例
    """
    workflow = SimpleAgentWorkflow()

    # 定义标准交易流程
    workflow.define_workflow(
        [
            {"agent": "market_data_agent", "action": "collect_data"},
            {"agent": "technical_agent", "action": "analyze", "input_from": "market_data_agent"},
            {"agent": "sentiment_agent", "action": "analyze", "input_from": "market_data_agent"},
            {
                "agent": "strategy_agent",
                "action": "generate_signal",
                "input_from": ["technical_agent", "sentiment_agent"],
            },
            {"agent": "risk_agent", "action": "assess_risk", "input_from": "strategy_agent"},
        ]
    )

    return workflow


__all__ = [
    "AgentState",
    "LangGraphAgentOrchestrator",
    "SimpleAgentWorkflow",
    "create_trading_workflow",
]
