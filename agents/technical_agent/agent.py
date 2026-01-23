"""
技术分析Agent
负责技术指标计算和趋势判断
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base.agent_base import (
    LLMAgent,
    Message,
    MessageType,
    tool,
)
from backtest.engine.indicators import TechnicalIndicators
from utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAnalysisAgent(LLMAgent):
    """
    技术分析Agent
    负责计算技术指标并生成技术分析报告
    """

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="technical_agent",
            description="负责技术指标计算、趋势判断、形态识别",
            llm_config=llm_config,
        )

        self.indicators = TechnicalIndicators()
        self._register_tools()

    def _register_tools(self):
        """注册工具"""
        self.register_tool("calculate_ma", self._calculate_ma)
        self.register_tool("calculate_macd", self._calculate_macd)
        self.register_tool("calculate_rsi", self._calculate_rsi)
        self.register_tool("detect_trend", self._detect_trend)
        self.register_tool("find_support_resistance", self._find_support_resistance)
        self.register_tool("generate_trading_signals", self._generate_signals)

    def _get_system_prompt(self) -> str:
        return """你是quantA系统的技术分析Agent，负责：

1. **技术指标计算**: MA、MACD、RSI、KDJ、布林带等
2. **趋势判断**: 识别上升、下降、震荡趋势
3. **支撑阻力位**: 找出关键的价格支撑和阻力位
4. **形态识别**: 识别头肩顶/底、双顶/底等技术形态
5. **信号生成**: 基于技术分析生成交易信号

你接收市场数据Agent处理后的数据，进行技术分析。
请客观分析，给出清晰的技术判断和建议。
"""

    async def process(self, message: Message) -> Optional[Message]:
        """处理消息"""
        logger.info(f"{self.name}处理消息")

        content = message.content

        try:
            if message.type == MessageType.ANALYSIS_REQUEST:
                result = await self._handle_analysis_request(content)

                response = Message(
                    type=MessageType.ANALYSIS_RESPONSE,
                    sender=self.name,
                    receiver=message.sender,
                    content=result,
                    reply_to=message.message_id,
                )

                return response

        except Exception as e:
            logger.error(f"处理失败: {e}", exc_info=True)
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)},
            )

    async def _handle_analysis_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析请求"""
        task = content.get("task")
        data = content.get("data")

        if task == "analyze":
            return await self._analyze_technical(data)
        elif task == "calculate_indicators":
            return await self._calculate_all_indicators(data)
        elif task == "detect_signals":
            return await self._detect_trading_signals(data)
        else:
            return {"error": f"未知任务: {task}"}

    async def _analyze_technical(self, data: pd.DataFrame) -> Dict[str, Any]:
        """技术分析"""
        if data is None or data.empty:
            return {"error": "数据为空"}

        analysis = {
            "trend": self._detect_trend_dict(data),
            "indicators": self._calculate_key_indicators(data),
            "support_resistance": self._find_support_resistance_dict(data),
            "recommendation": self._generate_recommendation(data),
        }

        return analysis

    async def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算所有指标"""
        if data is None or data.empty:
            return {"error": "数据为空"}

        indicators = self._calculate_key_indicators(data)
        return {"indicators": indicators, "status": "success"}

    async def _detect_trading_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检测交易信号"""
        signals = self._generate_signals(data)
        return {"signals": signals, "status": "success"}

    # ==================== 工具实现 ====================

    @tool(name="calculate_ma", description="计算移动平均线")
    async def _calculate_ma(
        self,
        data: pd.DataFrame,
        periods: List[int] = [5, 10, 20, 60],
    ) -> Dict[str, float]:
        """计算MA"""
        result = {}
        for period in periods:
            ma = self.indicators.sma(data["close"], period)
            result[f"MA{period}"] = ma.iloc[-1]
        return result

    @tool(name="calculate_macd", description="计算MACD指标")
    async def _calculate_macd(
        self,
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Dict[str, float]:
        """计算MACD"""
        macd_line, signal_line, histogram = self.indicators.macd(data["close"], fast, slow, signal)
        return {
            "MACD": macd_line.iloc[-1],
            "Signal": signal_line.iloc[-1],
            "Histogram": histogram.iloc[-1],
        }

    @tool(name="calculate_rsi", description="计算RSI指标")
    async def _calculate_rsi(
        self,
        data: pd.DataFrame,
        period: int = 14,
    ) -> Dict[str, Any]:
        """计算RSI"""
        rsi = self.indicators.rsi(data["close"], period)
        current_rsi = rsi.iloc[-1]

        return {
            "RSI": current_rsi,
            "overbought": current_rsi > 70,
            "oversold": current_rsi < 30,
        }

    @tool(name="detect_trend", description="检测价格趋势")
    async def _detect_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检测趋势"""
        return self._detect_trend_dict(data)

    @tool(name="find_support_resistance", description="寻找支撑阻力位")
    async def _find_support_resistance(
        self,
        data: pd.DataFrame,
        window: int = 20,
    ) -> Dict[str, Any]:
        """寻找支撑阻力位"""
        return self._find_support_resistance_dict(data)

    @tool(name="generate_trading_signals", description="生成交易信号")
    async def _generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成交易信号"""
        return self._generate_signals_dict(data)

    # ==================== 辅助方法 ====================

    def _detect_trend_dict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检测趋势"""
        ma_short = self.indicators.sma(data["close"], 20).iloc[-1]
        ma_long = self.indicators.sma(data["close"], 60).iloc[-1]
        current_price = data["close"].iloc[-1]

        if ma_short > ma_long and current_price > ma_short:
            trend = "uptrend"
            strength = "strong"
        elif ma_short < ma_long and current_price < ma_short:
            trend = "downtrend"
            strength = "strong"
        else:
            trend = "sideways"
            strength = "weak"

        return {
            "trend": trend,
            "strength": strength,
            "current_price": current_price,
            "MA20": ma_short,
            "MA60": ma_long,
        }

    def _calculate_key_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算关键指标"""
        return {
            "price": data["close"].iloc[-1],
            "volume": data["volume"].iloc[-1],
            "MA20": self.indicators.sma(data["close"], 20).iloc[-1],
            "MA60": self.indicators.sma(data["close"], 60).iloc[-1],
            "RSI": self.indicators.rsi(data["close"]).iloc[-1],
        }

    def _find_support_resistance_dict(
        self,
        data: pd.DataFrame,
        window: int = 20,
    ) -> Dict[str, Any]:
        """寻找支撑阻力位"""
        recent = data.tail(window)
        support = recent["low"].min()
        resistance = recent["high"].max()

        return {
            "support": support,
            "resistance": resistance,
            "current_price": data["close"].iloc[-1],
        }

    def _generate_recommendation(self, data: pd.DataFrame) -> str:
        """生成建议"""
        trend_info = self._detect_trend_dict(data)
        rsi = self.indicators.rsi(data["close"]).iloc[-1]

        if trend_info["trend"] == "uptrend" and rsi < 70:
            return "bullish"
        elif trend_info["trend"] == "downtrend" and rsi > 30:
            return "bearish"
        else:
            return "neutral"

    def _generate_signals_dict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成信号"""
        signals = []

        # MA交叉信号
        ma_short = self.indicators.sma(data["close"], 5)
        ma_long = self.indicators.sma(data["close"], 20)

        if ma_short.iloc[-1] > ma_long.iloc[-1] and ma_short.iloc[-2] <= ma_long.iloc[-2]:
            signals.append({"type": "golden_cross", "strength": "strong"})

        # RSI信号
        rsi = self.indicators.rsi(data["close"])
        if rsi.iloc[-1] < 30:
            signals.append({"type": "rsi_oversold", "action": "buy_signal"})
        elif rsi.iloc[-1] > 70:
            signals.append({"type": "rsi_overbought", "action": "sell_signal"})

        return {"signals": signals, "count": len(signals)}


__all__ = ["TechnicalAnalysisAgent"]
