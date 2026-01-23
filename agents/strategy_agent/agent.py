"""
策略生成Agent
综合信息生成交易信号
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base.agent_base import LLMAgent, Message, MessageType, tool
from utils.logging import get_logger

logger = get_logger(__name__)


class StrategyGenerationAgent(LLMAgent):
    """策略生成Agent"""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="strategy_agent",
            description="综合市场数据、技术分析和情绪分析生成交易信号",
            llm_config=llm_config,
        )
        self._register_tools()

    def _register_tools(self):
        self.register_tool("generate_signal", self._generate_signal)
        self.register_tool("evaluate_signal", self._evaluate_signal)
        self.register_tool("combine_signals", self._combine_signals)

    def _get_system_prompt(self) -> str:
        return """你是quantA系统的策略生成Agent，负责：

1. **信息整合**: 综合市场数据、技术分析、情绪分析的结果
2. **信号生成**: 基于多维度分析生成交易信号（买入/卖出/持有）
3. **信号评估**: 评估信号强度和置信度
4. **策略输出**: 输出结构化的交易建议

交易信号格式：
{
    'symbol': '股票代码',
    'action': 'buy/sell/hold',
    'quantity': 数量,
    'confidence': 置信度(0-1),
    'reasoning': '决策理由',
    'stop_loss': 止损价,
    'take_profit': 目标价
}

请综合考虑各种因素，给出理性、谨慎的交易建议。"""

    async def process(self, message: Message) -> Optional[Message]:
        """处理消息"""
        try:
            content = message.content

            # 整合分析结果
            technical_data = content.get("technical_analysis", {})
            sentiment_data = content.get("sentiment_analysis", {})
            market_data = content.get("market_data", {})

            # 生成信号
            signal = await self._generate_trading_signal(
                {
                    "technical": technical_data,
                    "sentiment": sentiment_data,
                    "market": market_data,
                }
            )

            return Message(
                type=MessageType.SIGNAL_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=signal,
                reply_to=message.message_id,
            )

        except Exception as e:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)},
            )

    async def _generate_trading_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成交易信号"""
        technical = data.get("technical", {})
        sentiment = data.get("sentiment", {})

        # 提取关键信息
        trend = technical.get("trend", "neutral")
        rsi = technical.get("indicators", {}).get("RSI", 50)
        sentiment_score = sentiment.get("overall_score", 0)

        # 决策逻辑
        action = "hold"
        confidence = 0.5
        reasoning = []

        # 趋势判断
        if trend == "uptrend":
            reasoning.append("上升趋势")
            if sentiment_score > 0.3:
                action = "buy"
                confidence = min(0.8, 0.5 + sentiment_score * 0.3)
                reasoning.append("情绪正面")
        elif trend == "downtrend":
            reasoning.append("下降趋势")
            if sentiment_score < -0.3:
                action = "sell"
                confidence = min(0.8, 0.5 + abs(sentiment_score) * 0.3)
                reasoning.append("情绪负面")

        # RSI超买超卖
        if rsi < 30:
            action = "buy"
            confidence = 0.7
            reasoning.append("RSI超卖")
        elif rsi > 70:
            action = "sell"
            confidence = 0.7
            reasoning.append("RSI超买")

        signal = {
            "action": action,
            "confidence": confidence,
            "reasoning": "; ".join(reasoning),
            "timestamp": datetime.now().isoformat(),
        }

        return signal

    @tool(name="generate_signal", description="生成交易信号")
    async def _generate_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成信号（工具）"""
        return await self._generate_trading_signal(data)

    @tool(name="evaluate_signal", description="评估信号质量")
    async def _evaluate_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """评估信号"""
        confidence = signal.get("confidence", 0.5)
        action = signal.get("action", "hold")

        quality = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"

        return {
            "action": action,
            "quality": quality,
            "confidence": confidence,
            "recommendation": "execute" if quality != "low" else "skip",
        }

    @tool(name="combine_signals", description="组合多个信号")
    async def _combine_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """组合信号"""
        buy_count = sum(1 for s in signals if s.get("action") == "buy")
        sell_count = sum(1 for s in signals if s.get("action") == "sell")
        hold_count = sum(1 for s in signals if s.get("action") == "hold")

        if buy_count > sell_count and buy_count > hold_count:
            final_action = "buy"
            confidence = buy_count / len(signals)
        elif sell_count > buy_count and sell_count > hold_count:
            final_action = "sell"
            confidence = sell_count / len(signals)
        else:
            final_action = "hold"
            confidence = 0.5

        return {
            "action": final_action,
            "confidence": confidence,
            "buy_votes": buy_count,
            "sell_votes": sell_count,
            "hold_votes": hold_count,
        }


__all__ = ["StrategyGenerationAgent"]
