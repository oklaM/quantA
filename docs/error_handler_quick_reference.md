# 错误处理快速参考

## 导入

```python
from utils.error_handler import (
    QuantAError, DataError, AgentError, BacktestError, RLError,
    handle_errors, ErrorHandler,
    create_error, format_error_for_api, log_and_raise
)
```

## 快速示例

### 1. 抛出异常

```python
# 基本用法
raise DataError("数据获取失败", symbol="600000.SH")

# 带详情
raise AgentError(
    "Agent执行失败",
    agent_name="strategy_agent",
    details={"retry_count": 3}
)
```

### 2. 装饰器使用

```python
# 不重新抛出，返回默认值
@handle_errors(error_type=DataError, default_return=None)
def fetch_data(symbol):
    # 函数逻辑
    pass

# 重新抛出异常
@handle_errors(error_type=AgentError, reraise=True)
async def process(message):
    # 函数逻辑
    pass

# 带上下文
@handle_errors(
    error_type=BacktestError,
    default_return={},
    context={"module": "backtest"}
)
def run_backtest(config):
    # 函数逻辑
    pass
```

### 3. 上下文管理器

```python
# 基本使用
with ErrorHandler() as handler:
    risky_operation()

if handler.has_error():
    print(f"错误: {handler.get_error()}")

# 重新抛出
try:
    with ErrorHandler(reraise=True) as handler:
        risky_operation()
except Exception as e:
    print(f"捕获: {e}")
```

### 4. 辅助函数

```python
# 创建错误
error = create_error(DataError, "失败", symbol="600000")

# API格式化
response = format_error_for_api(error)
# {"success": False, "error": {...}}

# 记录并抛出
log_and_raise(BacktestError, "失败", symbol="600000")
```

## 异常类型速查表

| 异常类 | 用途 | 特定参数 |
|--------|------|----------|
| `QuantAError` | 基类 | 无 |
| `DataError` | 数据层 | `data_source`, `symbol` |
| `AgentError` | Agent层 | `agent_name`, `agent_type` |
| `BacktestError` | 回测引擎 | `symbol`, `timestamp` |
| `RLError` | 强化学习 | `algorithm`, `env_id` |

## 装饰器参数速查表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `error_type` | Type | QuantAError | 捕获的异常类型 |
| `default_return` | Any | None | 错误时返回值 |
| `reraise` | bool | False | 是否重新抛出 |
| `log_level` | str | "ERROR" | 日志级别 |
| `context` | dict | None | 额外上下文 |

## 常见模式

### 数据获取
```python
@handle_errors(error_type=DataError, default_return=None)
def fetch_data(symbol: str):
    if not symbol:
        raise DataError("股票代码为空", symbol=symbol)
    # 获取逻辑
    return data
```

### Agent处理
```python
@handle_errors(error_type=AgentError, reraise=True)
async def process_message(self, message: str):
    if not message:
        raise AgentError("消息为空", agent_name=self.name)
    # 处理逻辑
    return result
```

### 回测执行
```python
with ErrorHandler(reraise=True, context={"symbol": symbol}) as handler:
    execute_backtest(config)
```

## 完整文档

详细文档请参考: `/Users/rowan/Projects/quantA/docs/error_handler_guide.md`
