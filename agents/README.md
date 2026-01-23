# LLM Agent系统使用指南

## 概述

quantA的Agent系统是一个基于GLM-4（智谱AI）的多Agent协作框架，用于自动化量化交易决策。

## 架构设计

### Agent类型

1. **MarketDataAgent** - 市场数据Agent
   - 获取实时和历史市场数据
   - 数据清洗和预处理

2. **TechnicalAgent** - 技术分析Agent
   - 计算技术指标
   - 技术形态识别

3. **SentimentAgent** - 情绪分析Agent
   - 分析市场情绪
   - 新闻和公告解读

4. **StrategyAgent** - 策略生成Agent
   - 综合各Agent的分析结果
   - 生成交易信号

5. **RiskAgent** - 风控Agent
   - 风险评估
   - 仓位管理建议

### 消息协议

Agent之间通过消息进行通信：

```python
class MessageType(Enum):
    MARKET_DATA = "market_data"          # 市场数据
    TECHNICAL_DATA = "technical_data"    # 技术指标
    SENTIMENT_DATA = "sentiment_data"    # 情绪数据
    ANALYSIS_REQUEST = "analysis_request"  # 分析请求
    ANALYSIS_RESPONSE = "analysis_response"  # 分析响应
    SIGNAL_REQUEST = "signal_request"    # 信号请求
    SIGNAL_RESPONSE = "signal_response"  # 信号响应
    RISK_CHECK_REQUEST = "risk_check_request"  # 风控检查
    RISK_CHECK_RESPONSE = "risk_check_response"  # 风控响应
```

## 快速开始

### 1. 安装依赖

```bash
pip install zhipuai
```

### 2. 配置API密钥

在 `.env` 文件中添加：

```
GLM_API_KEY=your_api_key_here
```

获取API密钥：
1. 访问 https://open.bigmodel.cn/
2. 注册账号
3. 在控制台获取API密钥

### 3. 使用示例

```python
import asyncio
from agents.base.agent_base import LLMAgent, MessageType, Message

class MyAgent(LLMAgent):
    def __init__(self):
        super().__init__(
            name="MyAgent",
            description="我的第一个量化交易Agent"
        )

    async def process(self, message: Message) -> Message:
        # 处理消息
        prompt = "请分析以下数据..."
        response = await self._call_llm(prompt)

        return Message(
            type=MessageType.ANALYSIS_RESPONSE,
            sender=self.name,
            receiver=message.sender,
            content={"result": response}
        )

# 使用Agent
async def main():
    agent = MyAgent()
    await agent.start()

    # 发送消息
    message = Message(
        type=MessageType.ANALYSIS_REQUEST,
        sender="User",
        receiver="MyAgent",
        content={"data": "..."}
    )

    response = await agent.process(message)
    print(response.content)

    await agent.stop()

asyncio.run(main())
```

## 高级用法

### 1. Agent协作

多个Agent协作完成复杂任务：

```python
# 创建多个Agent
market_agent = MarketDataAgent()
technical_agent = TechnicalAgent()
strategy_agent = StrategyAgent()

# Agent链式调用
# 1. 获取市场数据
market_data_msg = await market_agent.get_data("000001.SZ")

# 2. 技术分析
technical_msg = await technical_agent.analyze(market_data_msg)

# 3. 生成策略
strategy_msg = await strategy_agent.generate(technical_msg)
```

### 2. 使用工具

Agent可以调用工具函数：

```python
class AnalysisAgent(LLMAgent):
    def __init__(self):
        super().__init__(name="Analyzer")

        # 注册工具
        self.register_tool("calculate_ma", self.calculate_ma)
        self.register_tool("calculate_rsi", self.calculate_rsi)

    def calculate_ma(self, prices: list, period: int = 5) -> float:
        """计算移动平均线"""
        return sum(prices[-period:]) / period

    def calculate_rsi(self, prices: list) -> float:
        """计算RSI"""
        # RSI计算逻辑
        return 65.0
```

### 3. 自定义提示词

```python
agent = MyAgent()
agent.set_system_prompt("""
你是一个保守的量化交易专家。
请遵循以下原则：
1. 优先保本
2. 严格止损
3. 不过度交易
4. 优先选择低风险机会
""")
```

## 完整示例

### 示例：自动化交易流程

```python
import asyncio
from agents.base.agent_base import LLMAgent, MessageType, Message

class TradingBot:
    """自动化交易机器人"""

    def __init__(self):
        self.agents = {
            'market': MarketDataAgent(),
            'technical': TechnicalAgent(),
            'strategy': StrategyAgent(),
            'risk': RiskAgent()
        }

    async def run(self, symbol: str):
        """运行交易流程"""

        # 1. 获取市场数据
        market_msg = Message(
            type=MessageType.MARKET_DATA,
            sender="Bot",
            receiver="MarketDataAgent",
            content={'symbol': symbol}
        )
        market_response = await self.agents['market'].process(market_msg)

        # 2. 技术分析
        tech_msg = Message(
            type=MessageType.ANALYSIS_REQUEST,
            sender="Bot",
            receiver="TechnicalAgent",
            content=market_response.content
        )
        tech_response = await self.agents['technical'].process(tech_msg)

        # 3. 生成策略
        strategy_msg = Message(
            type=MessageType.SIGNAL_REQUEST,
            sender="Bot",
            receiver="StrategyAgent",
            content=tech_response.content
        )
        strategy_response = await self.agents['strategy'].process(strategy_msg)

        # 4. 风控检查
        risk_msg = Message(
            type=MessageType.RISK_CHECK_REQUEST,
            sender="Bot",
            receiver="RiskAgent",
            content=strategy_response.content
        )
        risk_response = await self.agents['risk'].process(risk_msg)

        # 5. 执行交易（如果风控通过）
        if risk_response.content.get('approved', False):
            print(f"执行交易：{strategy_response.content}")
        else:
            print(f"交易被风控拒绝：{risk_response.content.get('reason')}")

# 使用
async def main():
    bot = TradingBot()
    await bot.run("000001.SZ")

asyncio.run(main())
```

## 提示词工程

### 好的提示词示例

```python
system_prompt = """
你是一个专业的量化交易分析师，具有以下能力：
1. 精通各种技术分析指标
2. 熟悉A股市场特性
3. 理解风险管理的重要性

请遵循以下原则：
- 客观分析，不带偏见
- 关注风险控制
- 提供清晰的买卖理由
- 给出具体的建议价格
"""

user_prompt = """
请分析以下股票：
股票代码：000001.SZ
当前价格：10.50元
技术指标：
- MA5: 10.45
- MA10: 10.40
- MA20: 10.35
- RSI: 55

请给出：
1. 趋势判断
2. 支撑位和阻力位
3. 交易建议（买入/卖出/观望）
4. 止损止盈建议
"""
```

### 结构化输出

要求LLM返回结构化的JSON：

```python
prompt = """
请以JSON格式返回分析结果：

{
    "trend": "上涨/下跌/震荡",
    "support": 10.40,
    "resistance": 10.60,
    "recommendation": "买入",
    "confidence": 0.75,
    "stop_loss": 10.35,
    "take_profit": 10.70,
    "reason": "理由...",
    "risk": "风险提示..."
}
"""
```

## 调试和监控

### 日志配置

```python
from utils.logging import get_logger

logger = get_logger(__name__)

# 在Agent中使用
logger.info("Agent启动")
logger.debug(f"处理消息: {message}")
logger.error(f"处理失败: {e}")
```

### 消息追踪

```python
# 记录消息历史
message_id = agent.send_message(
    receiver="OtherAgent",
    message_type=MessageType.ANALYSIS_REQUEST,
    content={"data": "..."}
)

# 查看消息历史
history = agent.message_queue.get_by_id(message_id)
```

## 性能优化

### 1. 批量处理

```python
# 批量分析多个股票
symbols = ["000001.SZ", "000002.SZ", "600000.SH"]

tasks = [agent.analyze(symbol) for symbol in symbols]
results = await asyncio.gather(*tasks)
```

### 2. 缓存

```python
from functools import lru_cache

class CachedAgent(LLMAgent):
    @lru_cache(maxsize=100)
    async def _analyze_cached(self, symbol: str, date: str):
        # 带缓存的分析
        return await self._analyze(symbol, date)
```

### 3. 异步处理

```python
# 并发运行多个Agent
async def run_parallel():
    tasks = [
        agent1.process(msg1),
        agent2.process(msg2),
        agent3.process(msg3)
    ]
    results = await asyncio.gather(*tasks)
    return results
```

## 最佳实践

1. **模块化设计**：每个Agent专注于单一职责
2. **错误处理**：完善的异常处理和降级策略
3. **日志记录**：详细的日志便于调试
4. **测试**：为Agent编写单元测试
5. **提示词优化**：持续优化提示词提高准确度
6. **成本控制**：控制API调用频率和成本

## 常见问题

### Q1: API调用失败？

**A:** 检查：
1. API密钥是否正确
2. 网络连接是否正常
3. API余额是否充足
4. 请求频率是否过高

### Q2: LLM响应不准确？

**A:**
1. 优化提示词
2. 提供更多上下文
3. 使用few-shot示例
4. 要求结构化输出

### Q3: 如何降低成本？

**A:**
1. 使用较小的模型（如glm-4-air）
2. 批量处理减少调用次数
3. 缓存常用结果
4. 限制max_tokens

## 相关资源

- GLM-4文档：https://open.bigmodel.cn/dev/api
- 示例代码：`examples/llm_agent_example.py`
- Agent基类：`agents/base/agent_base.py`
- GLM-4集成：`agents/base/glm4_integration.py`
