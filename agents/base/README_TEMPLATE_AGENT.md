# Agent模板使用指南

## 概述

`TemplateAgent` 是一个使用模板方法模式的基类，用于消除所有Agent子类的重复代码。

## 设计模式

### 模板方法模式

`TemplateAgent` 将Agent的处理流程标准化为以下步骤：

1. **路由消息** (`_route_message`) - 根据消息类型分发到对应处理方法
2. **执行处理** (`_handle_*_request`) - 子类实现具体业务逻辑
3. **构建响应** (`_build_response`) - 统一构建响应消息
4. **错误处理** (`_build_error_response`) - 统一异常处理

## 使用方式

### 最简单的Agent

只需定义类属性和系统提示词：

```python
from agents.base.agent_template import TemplateAgent
from agents.base.agent_base import Message, MessageType

class SimpleAgent(TemplateAgent):
    """简单Agent示例"""

    # 定义类属性
    agent_name = "simple_agent"
    agent_description = "这是一个简单Agent"

    # 实现抽象方法
    def _get_system_prompt(self) -> str:
        return """你是一个简单的AI助手。"""
```

### 带工具的Agent

添加工具注册：

```python
from agents.base.agent_template import TemplateAgent
from agents.base.agent_base import tool
from typing import Dict, Any

class ToolAgent(TemplateAgent):
    """带工具的Agent示例"""

    agent_name = "tool_agent"
    agent_description = "带工具的Agent"

    def _get_system_prompt(self) -> str:
        return """你是quantA系统的工具Agent，负责示例任务。"""

    def _register_tools(self):
        """注册工具"""
        super()._register_tools()  # 调用父类（可选）
        self.register_tool("my_tool", self._my_tool)

    @tool(name="my_tool", description="我的工具")
    async def _my_tool(self, param: str) -> Dict[str, Any]:
        """工具实现"""
        return {"result": f"处理了: {param}"}
```

### 处理分析请求

覆盖分析请求处理方法：

```python
class AnalysisAgent(TemplateAgent):
    """分析Agent示例"""

    agent_name = "analysis_agent"
    agent_description = "分析Agent"

    def _get_system_prompt(self) -> str:
        return """你是分析Agent，负责数据分析。"""

    async def _handle_analysis_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析请求"""
        task = self._extract_task(content)
        data = self._extract_data(content)

        if task == "analyze":
            return self._build_success_response(
                data=self._perform_analysis(data),
                message="分析完成"
            )
        else:
            return self._build_error_response_dict(f"未知任务: {task}")

    def _perform_analysis(self, data):
        """实际分析逻辑"""
        # 实现分析逻辑
        return {"analysis_result": "..."}
```

### 多消息类型支持

支持多种消息类型：

```python
class MultiTypeAgent(TemplateAgent):
    """多消息类型Agent示例"""

    agent_name = "multi_type_agent"
    agent_description = "支持多种消息类型的Agent"

    def _get_system_prompt(self) -> str:
        return """你是多类型Agent。"""

    async def _handle_analysis_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析请求"""
        return self._build_success_response(
            data={"analysis": "..."},
            message="分析完成"
        )

    async def _handle_signal_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理信号请求"""
        return self._build_success_response(
            data={"signal": "buy"},
            message="信号生成完成"
        )

    async def _handle_risk_check_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理风控请求"""
        return self._build_success_response(
            data={"risk": "low"},
            message="风控检查通过"
        )
```

### 自定义消息路由

覆盖路由方法：

```python
class CustomRoutingAgent(TemplateAgent):
    """自定义路由Agent示例"""

    agent_name = "custom_routing_agent"
    agent_description = "自定义路由的Agent"

    def _get_system_prompt(self) -> str:
        return """你是自定义路由Agent。"""

    def _route_message(self, message):
        """自定义路由逻辑"""
        # 可以根据消息内容进行更复杂的路由
        if message.type == MessageType.ANALYSIS_REQUEST:
            task = message.content.get("task", "")
            if task == "advanced_analysis":
                return self._handle_advanced_analysis
            else:
                return self._handle_analysis_request
        # ...其他路由逻辑

    async def _handle_advanced_analysis(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理高级分析"""
        return self._build_success_response(
            data={"advanced_analysis": "..."},
            message="高级分析完成"
        )
```

## 内置辅助方法

### 消息提取

```python
# 提取任务类型
task = self._extract_task(content)

# 提取数据
data = self._extract_data(content)
```

### 响应构建

```python
# 构建成功响应
response = self._build_success_response(
    data={"key": "value"},
    message="操作成功",
    extra_field="extra_value"
)

# 构建错误响应
error = self._build_error_response_dict(
    error_message="操作失败",
    error_code="ERR_001",
    details="..."
)
```

## 完整示例：重写现有Agent

### 重写MarketDataAgent

```python
from agents.base.agent_template import TemplateAgent
from agents.base.agent_base import Message, MessageType, tool
from typing import Dict, Any, Optional
import pandas as pd

class MarketDataAgent(TemplateAgent):
    """市场数据Agent - 使用模板"""

    agent_name = "market_data_agent"
    agent_description = "负责市场数据的采集、清洗和特征提取"

    def __init__(self, llm_config: Optional[Dict[str, Any]] = None, data_source: str = "akshare"):
        super().__init__(llm_config)
        self.data_source = data_source
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _get_system_prompt(self) -> str:
        return """你是quantA系统的市场数据Agent，负责：

1. **数据采集**: 从数据源获取股票行情数据（日线、分钟线）
2. **数据清洗**: 处理缺失值、异常值，确保数据质量
3. **特征工程**: 计算技术指标，提取市场特征
4. **数据存储**: 将处理后的数据存储到时序数据库

请使用可用的工具完成数据任务，并返回结构化的结果。
"""

    def _register_tools(self):
        """注册工具"""
        self.register_tool("get_stock_data", self._get_stock_data)
        self.register_tool("clean_data", self._clean_data)
        self.register_tool("add_features", self._add_features)
        self.register_tool("get_realtime_quote", self._get_realtime_quote)

    async def _handle_analysis_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """处理分析请求"""
        task = self._extract_task(content)

        if task == "get_historical_data":
            return await self._get_historical_data(content)
        elif task == "get_realtime_data":
            return await self._get_realtime_data(content)
        elif task == "clean_data":
            return await self._clean_and_process(content)
        elif task == "add_indicators":
            return await self._add_technical_indicators(content)
        else:
            return self._build_error_response_dict(f"未知任务: {task}")

    # ... 其他业务逻辑方法
```

## 优势对比

### 使用模板前

每个Agent都需要重复编写：
- ✗ `__init__` 方法（30+ 行）
- ✗ `_register_tools` 调用
- ✗ `process` 方法（40+ 行，包括路由、错误处理）
- ✗ 响应构建逻辑
- ✗ 错误处理逻辑

**总计重复代码：100+ 行**

### 使用模板后

每个Agent只需编写：
- ✓ 2个类属性（2行）
- ✓ `_get_system_prompt` 方法（10-20行）
- ✓ 具体的业务逻辑方法

**总计代码减少：70-80%**

## 迁移指南

### 步骤1：更改基类

```python
# 之前
from agents.base.agent_base import LLMAgent

class MyAgent(LLMAgent):
    ...

# 之后
from agents.base.agent_template import TemplateAgent

class MyAgent(TemplateAgent):
    ...
```

### 步骤2：删除重复代码

删除以下重复代码：
- `__init__` 方法中的 `super().__init__()` 调用
- 整个 `process` 方法
- 手动构建响应的代码
- 手动错误处理代码

### 步骤3：添加类属性

```python
class MyAgent(TemplateAgent):
    agent_name = "my_agent"
    agent_description = "我的Agent描述"
    ...
```

### 步骤4：重命名处理方法

将处理逻辑从私有方法改为模板方法：

```python
# 之前
async def process(self, message: Message) -> Optional[Message]:
    if message.type == MessageType.ANALYSIS_REQUEST:
        result = await self._do_something(message.content)
        return Message(...)
    ...

# 之后
async def _handle_analysis_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
    return await self._do_something(content)
```

## 注意事项

1. **必须覆盖的成员**：
   - `agent_name` (类属性)
   - `agent_description` (类属性)
   - `_get_system_prompt()` (方法)

2. **可选覆盖的成员**：
   - `_register_tools()` - 如果需要工具
   - `_handle_*_request()` - 根据需要处理的消息类型
   - `_route_message()` - 如果需要自定义路由

3. **不要覆盖的成员**：
   - `__init__()` - 模板已实现标准初始化
   - `process()` - 这是模板方法，已实现标准流程

4. **使用辅助方法**：
   - 使用 `_extract_task()` 和 `_extract_data()` 提取消息内容
   - 使用 `_build_success_response()` 和 `_build_error_response_dict()` 构建响应

## 总结

`TemplateAgent` 通过模板方法模式实现了：

1. **代码复用**：消除70-80%的重复代码
2. **统一标准**：所有Agent遵循相同的处理流程
3. **易于维护**：修改一处，所有子类受益
4. **灵活扩展**：子类可以按需覆盖特定方法
5. **清晰职责**：子类只需关注业务逻辑

这使得编写新Agent变得简单快捷，同时保持了代码的一致性和可维护性。
