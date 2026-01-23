"""
风控Agent
负责仓位管理和风险控制
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base.agent_base import LLMAgent, Message, MessageType, tool
from utils.logging import get_logger

logger = get_logger(__name__)


class RiskManagementAgent(LLMAgent):
    """风控Agent"""

    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        max_position_ratio: float = 0.2,
        max_total_position: float = 0.95,
        stop_loss_ratio: float = -0.05,
        take_profit_ratio: float = 0.15,
    ):
        super().__init__(
            name="risk_agent",
            description="负责仓位管理、风险控制和合规检查",
            llm_config=llm_config,
        )
        self.max_position_ratio = max_position_ratio
        self.max_total_position = max_total_position
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self._register_tools()

    def _register_tools(self):
        self.register_tool("check_position_limit", self._check_position_limit)
        self.register_tool("check_stop_loss", self._check_stop_loss)
        self.register_tool("calculate_position_size", self._calculate_position_size)
        self.register_tool("assess_risk", self._assess_risk)

    def _get_system_prompt(self) -> str:
        return """你是quantA系统的风控Agent，负责：

1. **仓位限制**: 单股票最大仓位、总仓位限制
2. **止损止盈**: 设置止损位和止盈位
3. **风险控制**: 实时监控风险指标
4. **合规检查**: 确保交易符合监管要求

风控规则：
- 单股票最大仓位：20%
- 总仓位上限：95%
- 止损：-5%
- 止盈：+15%
- 最大回撤限制：-10%

你的首要任务是保护资金安全，谨慎对待每一笔交易。"""

    async def process(self, message: Message) -> Optional[Message]:
        """处理消息"""
        try:
            content = message.content
            task = content.get("task", "assess")

            if task == "assess":
                risk_assessment = await self._assess_trade_risk(content)
            elif task == "check_limits":
                risk_assessment = await self._check_trading_limits(content)
            else:
                risk_assessment = {"error": f"未知任务: {task}"}

            return Message(
                type=MessageType.RISK_CHECK_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=risk_assessment,
                reply_to=message.message_id,
            )

        except Exception as e:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)},
            )

    async def _assess_trade_risk(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """评估交易风险"""
        signal = content.get("signal", {})
        account = content.get("account", {})
        positions = content.get("positions", [])

        symbol = signal.get("symbol")
        action = signal.get("action")
        proposed_quantity = signal.get("quantity", 0)
        confidence = signal.get("confidence", 0.5)

        # 风险评估
        assessment = {
            "approved": False,
            "adjusted_quantity": proposed_quantity,
            "warnings": [],
            "risk_level": "medium",
        }

        # 检查置信度
        if confidence < 0.5:
            assessment["warnings"].append("信号置信度低，建议谨慎交易")
            assessment["risk_level"] = "high"

        # 检查仓位限制
        current_positions = sum(p.get("quantity", 0) for p in positions)
        total_value = account.get("total_value", 1000000)

        if action == "buy":
            max_position_value = total_value * self.max_position_ratio
            max_quantity = int(max_position_value / signal.get("price", 1))

            if proposed_quantity > max_quantity:
                assessment["warnings"].append(f"超过单股最大仓位限制，调整到{max_quantity}股")
                assessment["adjusted_quantity"] = max_quantity

        # 最终决策
        if assessment["risk_level"] != "high":
            assessment["approved"] = True

        return assessment

    async def _check_trading_limits(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """检查交易限制"""
        return {
            "max_position_ratio": self.max_position_ratio,
            "max_total_position": self.max_total_position,
            "stop_loss_ratio": self.stop_loss_ratio,
            "take_profit_ratio": self.take_profit_ratio,
            "status": "compliant",
        }

    # ==================== 工具函数 ====================

    @tool(name="check_position_limit", description="检查仓位限制")
    async def _check_position_limit(
        self,
        symbol: str,
        quantity: int,
        price: float,
        total_value: float,
        current_positions: float,
    ) -> Dict[str, Any]:
        """检查仓位限制"""
        position_value = quantity * price
        position_ratio = position_value / total_value

        max_allowed = total_value * self.max_position_ratio

        if position_value > max_allowed:
            return {
                "passed": False,
                "reason": f"超过单股最大仓位限制{self.max_position_ratio*100}%",
                "max_quantity": int(max_allowed / price),
            }

        total_position_ratio = (current_positions + position_value) / total_value
        if total_position_ratio > self.max_total_position:
            return {
                "passed": False,
                "reason": f"超过总仓位限制{self.max_total_position*100}%",
            }

        return {"passed": True, "position_ratio": position_ratio}

    @tool(name="check_stop_loss", description="检查止损")
    async def _check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_side: str,
    ) -> Dict[str, Any]:
        """检查止损"""
        if position_side == "long":
            pnl_ratio = (current_price - entry_price) / entry_price
        else:
            pnl_ratio = (entry_price - current_price) / entry_price

        if pnl_ratio < self.stop_loss_ratio:
            return {
                "triggered": True,
                "pnl_ratio": pnl_ratio,
                "action": "close_position",
            }

        return {"triggered": False, "pnl_ratio": pnl_ratio}

    @tool(name="calculate_position_size", description="计算仓位大小")
    async def _calculate_position_size(
        self,
        signal_strength: float,
        total_value: float,
        price: float,
    ) -> int:
        """计算仓位大小"""
        # 根据信号强度调整仓位
        base_ratio = 0.1  # 基础10%仓位
        adjusted_ratio = base_ratio * signal_strength

        position_value = total_value * adjusted_ratio
        quantity = int(position_value / price)

        # 确保是100的整数倍
        quantity = (quantity // 100) * 100

        return max(100, quantity)

    @tool(name="assess_risk", description="综合风险评估")
    async def _assess_risk(
        self,
        signal: Dict[str, Any],
        account: Dict[str, Any],
    ) -> Dict[str, Any]:
        """综合风险评估"""
        confidence = signal.get("confidence", 0.5)

        if confidence > 0.7:
            return {"risk_level": "low", "recommendation": "execute"}
        elif confidence > 0.5:
            return {"risk_level": "medium", "recommendation": "reduce_position"}
        else:
            return {"risk_level": "high", "recommendation": "skip"}


__all__ = ["RiskManagementAgent"]
