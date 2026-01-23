"""
情绪分析Agent
负责新闻舆情和社交媒体情绪分析
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base.agent_base import LLMAgent, Message, MessageType, tool
from utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAnalysisAgent(LLMAgent):
    """情绪分析Agent"""

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="sentiment_agent",
            description="负责新闻舆情和社交媒体情绪分析",
            llm_config=llm_config,
        )
        self._register_tools()

    def _register_tools(self):
        self.register_tool("analyze_news", self._analyze_news)
        self.register_tool("analyze_social_media", self._analyze_social_media)
        self.register_tool("calculate_sentiment_score", self._calculate_sentiment_score)

    def _get_system_prompt(self) -> str:
        return """你是quantA系统的情绪分析Agent，负责：

1. **新闻分析**: 分析财经新闻的市场情绪
2. **社交媒体监控**: 监控雪球、股吧等平台情绪
3. **情绪量化**: 计算情绪得分（-1到1）
4. **异常检测**: 识别异常情绪波动

你接收文本数据，分析其中的情绪倾向（正面/负面/中性）。
输出结构化的情绪分析结果。"""

    async def process(self, message: Message) -> Optional[Message]:
        """处理消息"""
        try:
            result = await self._analyze_sentiment(message.content)

            return Message(
                type=MessageType.ANALYSIS_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                reply_to=message.message_id,
            )
        except Exception as e:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)},
            )

    async def _analyze_sentiment(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """分析情绪"""
        news = content.get("news", [])
        social_posts = content.get("social_posts", [])

        sentiment = {
            "overall_score": 0.0,  # -1到1
            "news_sentiment": await self._analyze_news(news),
            "social_sentiment": await self._analyze_social_media(social_posts),
            "timestamp": datetime.now().isoformat(),
        }

        # 计算综合情绪
        news_score = sentiment["news_sentiment"].get("score", 0)
        social_score = sentiment["social_sentiment"].get("score", 0)
        sentiment["overall_score"] = (news_score + social_score) / 2

        return sentiment

    @tool(name="analyze_news", description="分析新闻情绪")
    async def _analyze_news(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析新闻"""
        if not news:
            return {"count": 0, "score": 0, "analysis": "无新闻数据"}

        # 简化版情绪分析
        positive_keywords = ["上涨", "利好", "突破", "增长", "看好"]
        negative_keywords = ["下跌", "利空", "暴跌", "风险", "担忧"]

        positive_count = 0
        negative_count = 0

        for item in news:
            text = item.get("title", "") + item.get("content", "")
            for word in positive_keywords:
                if word in text:
                    positive_count += 1
            for word in negative_keywords:
                if word in text:
                    negative_count += 1

        total = positive_count + negative_count
        score = (positive_count - negative_count) / total if total > 0 else 0

        return {
            "count": len(news),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "score": score,  # -1到1
        }

    @tool(name="analyze_social_media", description="分析社交媒体情绪")
    async def _analyze_social_media(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析社交媒体"""
        if not posts:
            return {"count": 0, "score": 0}

        # 简化版：分析帖子情绪
        positive = 0
        negative = 0

        for post in posts:
            text = post.get("content", "")
            # 简单情绪判断
            if any(word in text for word in ["看多", "买入", "涨"]):
                positive += 1
            elif any(word in text for word in ["看空", "卖出", "跌"]):
                negative += 1

        score = (positive - negative) / len(posts) if posts else 0

        return {"count": len(posts), "score": score}

    @tool(name="calculate_sentiment_score", description="计算情绪得分")
    async def _calculate_sentiment_score(
        self,
        news_sentiment: float,
        social_sentiment: float,
    ) -> float:
        """计算综合情绪得分"""
        return (news_sentiment + social_sentiment) / 2


__all__ = ["SentimentAnalysisAgent"]
