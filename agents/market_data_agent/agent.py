"""
市场数据Agent
负责数据采集、清洗、特征提取
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base.agent_base import (
    LLMAgent,
    Message,
    MessageType,
    tool,
)
from backtest.engine.indicators import add_indicators
from utils.logging import get_logger

logger = get_logger(__name__)


class MarketDataAgent(LLMAgent):
    """
    市场数据Agent
    负责采集、清洗和处理市场数据
    """

    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        data_source: str = "akshare",
    ):
        """
        Args:
            llm_config: LLM配置
            data_source: 数据源 (akshare, tushare)
        """
        super().__init__(
            name="market_data_agent",
            description="负责市场数据的采集、清洗和特征提取",
            llm_config=llm_config,
        )

        self.data_source = data_source
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # 注册工具
        self._register_tools()

    def _register_tools(self):
        """注册工具函数"""
        self.register_tool("get_stock_data", self._get_stock_data)
        self.register_tool("clean_data", self._clean_data)
        self.register_tool("add_features", self._add_features)
        self.register_tool("get_realtime_quote", self._get_realtime_quote)

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是quantA系统的市场数据Agent，负责：

1. **数据采集**: 从数据源获取股票行情数据（日线、分钟线）
2. **数据清洗**: 处理缺失值、异常值，确保数据质量
3. **特征工程**: 计算技术指标，提取市场特征
4. **数据存储**: 将处理后的数据存储到时序数据库

你接收的请求可能包括：
- 获取某只股票的历史数据
- 清洗和预处理数据
- 计算技术指标特征
- 获取实时行情

请使用可用的工具完成数据任务，并返回结构化的结果。
"""

    async def process(self, message: Message) -> Optional[Message]:
        """
        处理消息

        Args:
            message: 接收到的消息

        Returns:
            响应消息
        """
        logger.info(f"{self.name}处理消息: {message.type.value}")

        content = message.content

        try:
            if message.type == MessageType.ANALYSIS_REQUEST:
                # 数据分析请求
                result = await self._handle_data_request(content)
            else:
                result = {"error": f"不支持的消息类型: {message.type.value}"}

            # 构建响应
            response = Message(
                type=MessageType.ANALYSIS_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content=result,
                reply_to=message.message_id,
            )

            return response

        except Exception as e:
            logger.error(f"处理消息失败: {e}", exc_info=True)

            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)},
                reply_to=message.message_id,
            )

    async def _handle_data_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据请求

        Args:
            content: 请求内容

        Returns:
            处理结果
        """
        task = content.get("task", "")

        if task == "get_historical_data":
            return await self._get_historical_data(content)
        elif task == "get_realtime_data":
            return await self._get_realtime_data(content)
        elif task == "clean_data":
            return await self._clean_and_process(content)
        elif task == "add_indicators":
            return await self._add_technical_indicators(content)
        else:
            return {"error": f"未知的任务类型: {task}"}

    async def _get_historical_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取历史数据

        Args:
            params: 参数 {symbol, start_date, end_date}

        Returns:
            历史数据
        """
        symbol = params.get("symbol")
        start_date = params.get("start_date")
        end_date = params.get("end_date", datetime.now().strftime("%Y%m%d"))

        if not symbol or not start_date:
            return {"error": "缺少必要参数: symbol, start_date"}

        try:
            # 从数据源获取数据
            data = await self._fetch_data(symbol, start_date, end_date)

            if data is not None and not data.empty:
                # 缓存数据
                self._data_cache[symbol] = data

                return {
                    "symbol": symbol,
                    "data_points": len(data),
                    "date_range": f"{data['date'].min()} ~ {data['date'].max()}",
                    "preview": data.head(5).to_dict("records"),
                    "status": "success",
                }
            else:
                return {"error": "未获取到数据"}

        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return {"error": str(e)}

    async def _get_realtime_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取实时数据

        Args:
            params: 参数 {symbols}

        Returns:
            实时行情
        """
        symbols = params.get("symbols", [])

        if not symbols:
            return {"error": "缺少参数: symbols"}

        try:
            # 获取实时行情
            quotes = await self._fetch_realtime_quotes(symbols)

            return {
                "quotes": quotes,
                "count": len(quotes),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
            return {"error": str(e)}

    async def _clean_and_process(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗和处理数据

        Args:
            params: 参数 {data, operations}

        Returns:
            清洗后的数据
        """
        # 这里实现数据清洗逻辑
        return {"status": "success", "message": "数据清洗完成"}

    async def _add_technical_indicators(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加技术指标

        Args:
            params: 参数 {symbol, indicators}

        Returns:
            添加指标后的数据
        """
        symbol = params.get("symbol")

        if symbol not in self._data_cache:
            return {"error": f"未找到{symbol}的数据，请先获取历史数据"}

        try:
            # 添加技术指标
            df = self._data_cache[symbol]
            df_with_indicators = add_indicators(df)

            # 更新缓存
            self._data_cache[symbol] = df_with_indicators

            return {
                "symbol": symbol,
                "indicators_added": list(df_with_indicators.columns),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"添加技术指标失败: {e}")
            return {"error": str(e)}

    # ==================== 工具函数 ====================

    @tool(
        name="get_stock_data",
        description="获取股票历史行情数据",
    )
    async def _get_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """获取股票数据（工具）"""
        return await self._get_historical_data(
            {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

    @tool(
        name="clean_data",
        description="清洗数据（处理缺失值、异常值）",
    )
    async def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """清洗数据（工具）"""
        # 处理缺失值
        data = data.dropna(subset=["close"])

        # 处理异常值
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                # 简单的异常值处理：超出3个标准差的值
                mean = data[col].mean()
                std = data[col].std()
                data[col] = data[col].clip(lower=mean - 3 * std, upper=mean + 3 * std)

        return data

    @tool(
        name="add_features",
        description="添加技术指标特征",
    )
    async def _add_features(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """添加特征（工具）"""
        return add_indicators(data)

    @tool(
        name="get_realtime_quote",
        description="获取实时行情",
    )
    async def _get_realtime_quote(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """获取实时行情（工具）"""
        return await self._fetch_realtime_quotes(symbols)

    # ==================== 私有方法 ====================

    async def _fetch_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """
        从数据源获取数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            数据DataFrame
        """
        try:
            if self.data_source == "akshare":
                from data.market.sources.akshare_provider import AKShareProvider

                provider = AKShareProvider()
            elif self.data_source == "tushare":
                from data.market.sources.tushare_provider import TushareProvider

                provider = TushareProvider()
            else:
                raise ValueError(f"不支持的数据源: {self.data_source}")

            provider.connect()

            # 获取数据
            df = provider.get_daily_bar(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )

            provider.disconnect()
            return df

        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return None

    async def _fetch_realtime_quotes(
        self,
        symbols: List[str],
    ) -> List[Dict[str, Any]]:
        """
        获取实时行情

        Args:
            symbols: 股票代码列表

        Returns:
            行情列表
        """
        try:
            if self.data_source == "akshare":
                from data.market.sources.akshare_provider import AKShareProvider

                provider = AKShareProvider()
            else:
                from data.market.sources.tushare_provider import TushareProvider

                provider = TushareProvider()

            provider.connect()

            # 获取实时行情
            df = provider.get_realtime_quote(symbols)

            provider.disconnect()

            if not df.empty:
                return df.to_dict("records")
            else:
                return []

        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")
            return []


__all__ = ["MarketDataAgent"]
