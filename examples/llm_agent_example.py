"""
LLM Agent使用示例

展示如何使用GLM-4驱动的AI Agent进行量化交易分析
"""

import asyncio
import json
from datetime import datetime
from agents.base.agent_base import LLMAgent, MessageType, Message
from agents.base.glm4_integration import GLM4Client, create_glm_agent


class MarketAnalysisAgent(LLMAgent):
    """市场分析Agent - 使用GLM-4分析市场数据"""

    def __init__(self):
        super().__init__(
            name="MarketAnalyzer",
            description="你是一个专业的股市分析师，擅长技术分析和趋势判断。",
            llm_config={'use_glm4': True}
        )

        # 注册工具
        self.register_tool("calculate_ma", self.calculate_ma, "计算移动平均线")
        self.register_tool("calculate_rsi", self.calculate_rsi, "计算RSI指标")

    async def process(self, message: Message) -> Message:
        """处理消息"""
        if message.type == MessageType.ANALYSIS_REQUEST:
            return await self._analyze_market(message)
        else:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": "不支持的消息类型"}
            )

    async def _analyze_market(self, message: Message) -> Message:
        """分析市场"""
        try:
            # 获取市场数据
            market_data = message.content.get('market_data', {})

            # 构建提示词
            prompt = self._build_analysis_prompt(market_data)

            # 调用LLM
            system_prompt = """你是一个专业的股市分析师。请根据提供的技术指标数据进行分析：
1. 评估当前市场趋势（上涨/下跌/震荡）
2. 识别关键支撑位和阻力位
3. 给出交易建议（买入/卖出/观望）
4. 提供风险提示

请以JSON格式返回分析结果，包含以下字段：
{
    "trend": "上涨/下跌/震荡",
    "support": "支撑位价格",
    "resistance": "阻力位价格",
    "recommendation": "买入/卖出/观望",
    "confidence": 0.0-1.0,
    "reason": "分析理由",
    "risk": "风险提示"
}"""

            response = await self._call_llm(prompt, system_prompt=system_prompt)

            # 解析响应
            try:
                analysis = json.loads(response)
            except json.JSONDecodeError:
                # 如果LLM没有返回JSON，创建一个默认响应
                analysis = {
                    "trend": "震荡",
                    "support": "N/A",
                    "resistance": "N/A",
                    "recommendation": "观望",
                    "confidence": 0.5,
                    "reason": response,
                    "risk": "请谨慎投资"
                }

            return Message(
                type=MessageType.ANALYSIS_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content={
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)}
            )

    def _build_analysis_prompt(self, market_data: dict) -> str:
        """构建分析提示词"""
        symbol = market_data.get('symbol', '未知')
        current_price = market_data.get('close', 0)
        volume = market_data.get('volume', 0)

        # 技术指标
        ma5 = market_data.get('ma5', 0)
        ma10 = market_data.get('ma10', 0)
        ma20 = market_data.get('ma20', 0)
        rsi = market_data.get('rsi', 50)

        prompt = f"""请分析以下股票的技术面：

股票代码：{symbol}
当前价格：{current_price}
成交量：{volume}

技术指标：
- MA5：{ma5}
- MA10：{ma10}
- MA20：{ma20}
- RSI：{rsi}

请基于这些指标进行综合分析。"""

        return prompt

    # 工具函数
    def calculate_ma(self, prices: list, period: int = 5) -> float:
        """计算移动平均线"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        return sum(prices[-period:]) / period

    def calculate_rsi(self, prices: list, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0

        gains = []
        losses = []

        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class StrategyAgent(LLMAgent):
    """策略生成Agent - 使用GLM-4生成交易策略"""

    def __init__(self):
        super().__init__(
            name="StrategyGenerator",
            description="你是一个量化交易策略生成专家，擅长根据市场分析结果制定交易策略。",
            llm_config={'use_glm4': True}
        )

    async def process(self, message: Message) -> Message:
        """处理消息"""
        if message.type == MessageType.SIGNAL_REQUEST:
            return await self._generate_signal(message)
        else:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": "不支持的消息类型"}
            )

    async def _generate_signal(self, message: Message) -> Message:
        """生成交易信号"""
        try:
            # 获取分析结果
            analysis = message.content.get('analysis', {})
            account_info = message.content.get('account_info', {})

            # 构建提示词
            prompt = f"""基于以下市场分析结果，请生成具体的交易信号：

市场分析：
{json.dumps(analysis, ensure_ascii=False, indent=2)}

当前账户状态：
{json.dumps(account_info, ensure_ascii=False, indent=2)}

请生成交易信号，包含以下信息：
1. 交易方向（买入/卖出/观望）
2. 建议数量（股数）
3. 建议价格（或市价）
4. 止损价
5. 止盈价
6. 仓位比例（0-1）

请以JSON格式返回。"""

            system_prompt = """你是一个专业的交易员。请根据分析结果制定具体的交易计划。
考虑风险控制，遵循以下原则：
1. 严格控制单笔交易风险（不超过总资金的2%）
2. 设置合理的止损止盈位
3. 考虑当前仓位情况
4. 优先保本，其次获利"""

            response = await self._call_llm(prompt, system_prompt=system_prompt)

            # 解析响应
            try:
                signal = json.loads(response)
            except json.JSONDecodeError:
                signal = {
                    "action": "观望",
                    "quantity": 0,
                    "price": "市价",
                    "stop_loss": "N/A",
                    "take_profit": "N/A",
                    "position_ratio": 0,
                    "reason": response
                }

            return Message(
                type=MessageType.SIGNAL_RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content={
                    "signal": signal,
                    "timestamp": datetime.now().isoformat()
                }
            )

        except Exception as e:
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)}
            )


async def example_simple_chat():
    """示例1：简单的GLM-4对话"""
    print("\n" + "="*50)
    print("示例1：简单的GLM-4对话")
    print("="*50)

    try:
        client = GLM4Client()

        messages = [
            {"role": "user", "content": "你好，请简单介绍一下量化交易。"}
        ]

        response = await client.chat(messages)
        print(f"\nGLM-4响应：\n{response}")

    except Exception as e:
        print(f"\n错误：{e}")
        print("\n请确保：")
        print("1. 已安装 zhipuai: pip install zhipuai")
        print("2. 已在 .env 文件中设置 GLM_API_KEY")


async def example_market_analysis():
    """示例2：市场分析Agent"""
    print("\n" + "="*50)
    print("示例2：市场分析Agent")
    print("="*50)

    # 创建市场分析Agent
    analyzer = MarketAnalysisAgent()
    await analyzer.start()

    # 模拟市场数据
    market_data = {
        'symbol': '000001.SZ',
        'close': 10.50,
        'volume': 1000000,
        'ma5': 10.45,
        'ma10': 10.40,
        'ma20': 10.35,
        'rsi': 55.0
    }

    # 创建分析请求
    message = Message(
        type=MessageType.ANALYSIS_REQUEST,
        sender="User",
        receiver="MarketAnalyzer",
        content={'market_data': market_data}
    )

    # 发送消息并获取响应
    response = await analyzer.process(message)

    print(f"\n分析结果：")
    print(json.dumps(response.content, ensure_ascii=False, indent=2))

    await analyzer.stop()


async def example_agent_collaboration():
    """示例3：Agent协作"""
    print("\n" + "="*50)
    print("示例3：Agent协作（分析 + 策略）")
    print("="*50)

    # 创建两个Agent
    analyzer = MarketAnalysisAgent()
    strategist = StrategyAgent()

    await analyzer.start()
    await strategist.start()

    # 模拟市场数据
    market_data = {
        'symbol': '600519.SH',
        'close': 1850.00,
        'volume': 5000000,
        'ma5': 1845.00,
        'ma10': 1840.00,
        'ma20': 1835.00,
        'rsi': 65.0
    }

    account_info = {
        'cash': 1000000,
        'total_value': 1000000,
        'positions': []
    }

    # 第一步：市场分析
    print("\n[步骤1] 市场分析Agent分析...")
    analysis_request = Message(
        type=MessageType.ANALYSIS_REQUEST,
        sender="User",
        receiver="MarketAnalyzer",
        content={'market_data': market_data}
    )

    analysis_response = await analyzer.process(analysis_request)
    analysis = analysis_response.content.get('analysis', {})

    print(f"\n分析结果：")
    print(json.dumps(analysis, ensure_ascii=False, indent=2))

    # 第二步：生成交易信号
    print("\n[步骤2] 策略Agent生成交易信号...")

    signal_request = Message(
        type=MessageType.SIGNAL_REQUEST,
        sender="User",
        receiver="StrategyGenerator",
        content={
            'analysis': analysis,
            'account_info': account_info
        }
    )

    signal_response = await strategist.process(signal_request)
    signal = signal_response.content.get('signal', {})

    print(f"\n交易信号：")
    print(json.dumps(signal, ensure_ascii=False, indent=2))

    # 关闭Agent
    await analyzer.stop()
    await strategist.stop()


async def example_custom_agent():
    """示例4：创建自定义Agent"""
    print("\n" + "="*50)
    print("示例4：创建自定义Agent")
    print("="*50)

    # 定义工具函数
    def get_stock_price(symbol: str) -> dict:
        """获取股票价格（模拟）"""
        return {
            'symbol': symbol,
            'price': 100.0,
            'change': 2.5
        }

    def calculate_rsi(prices: list) -> float:
        """计算RSI"""
        # 简化版RSI计算
        return 65.0

    # 创建Agent
    agent = create_glm_agent(
        name="CustomAgent",
        system_prompt="你是一个专业的量化交易助手，可以帮助用户分析股票。",
        tools={
            "get_stock_price": get_stock_price,
            "calculate_rsi": calculate_rsi
        }
    )

    # 使用Agent
    prompt = "请帮我查询000001.SZ的价格，并计算其RSI指标。"

    response = await agent.call_llm_with_tools(
        prompt=prompt,
        tools=agent.get_tools()
    )

    print(f"\nAgent响应：\n{response}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA LLM Agent使用示例")
    print("="*70)

    # 运行示例
    try:
        # 示例1：简单对话
        asyncio.run(example_simple_chat())

        # 示例2：市场分析Agent
        # asyncio.run(example_market_analysis())

        # 示例3：Agent协作
        # asyncio.run(example_agent_collaboration())

        # 示例4：自定义Agent
        # asyncio.run(example_custom_agent())

        print("\n" + "="*70)
        print("示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        print("\n请检查：")
        print("1. 是否安装了zhipuai包: pip install zhipuai")
        print("2. 是否设置了API密钥（在.env文件中）")
        print("3. 网络连接是否正常")


if __name__ == "__main__":
    main()
