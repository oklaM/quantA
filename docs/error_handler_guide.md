# 统一错误处理机制文档

## 概述

`utils/error_handler.py` 提供了quantA系统的统一错误处理机制，包括自定义异常类层次结构、错误处理装饰器和上下文管理器。

## 功能特性

- **自定义异常类层次**: 针对不同模块的专用异常类
- **错误处理装饰器**: 支持同步和异步函数
- **上下文管理器**: 用于with语句的错误处理
- **自动日志记录**: 错误发生时自动记录详细信息
- **灵活的错误处理**: 支持返回默认值或重新抛出异常

## 异常类层次

### 基类: QuantAError

所有自定义异常的基类，提供统一的错误格式和详情管理。

```python
class QuantAError(Exception):
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        """转换为字典格式"""
```

### 专用异常类

#### 1. DataError - 数据层异常

用于数据获取、存储、处理过程中的错误。

```python
DataError(
    message: str,
    data_source: Optional[str] = None,  # tushare, akshare, duckdb等
    symbol: Optional[str] = None,        # 股票代码
    details: Optional[dict] = None
)
```

**使用场景**:
- 数据源连接失败
- 数据格式错误
- 数据获取超时
- 数据存储失败

#### 2. AgentError - Agent层异常

用于LLM Agent执行过程中的错误。

```python
AgentError(
    message: str,
    agent_name: Optional[str] = None,     # Agent名称
    agent_type: Optional[str] = None,     # market_data, technical等
    details: Optional[dict] = None
)
```

**使用场景**:
- Agent初始化失败
- 消息处理失败
- LLM调用失败
- Agent协调失败

#### 3. BacktestError - 回测引擎异常

用于回测执行、指标计算、订单处理等错误。

```python
BacktestError(
    message: str,
    symbol: Optional[str] = None,         # 股票代码
    timestamp: Optional[str] = None,      # 错误发生时间
    details: Optional[dict] = None
)
```

**使用场景**:
- 回测配置错误
- 订单执行失败
- 指标计算错误
- 数据回放失败

#### 4. RLError - 强化学习异常

用于RL训练、环境、策略等错误。

```python
RLError(
    message: str,
    algorithm: Optional[str] = None,      # ppo, dqn, a2c等
    env_id: Optional[str] = None,         # 环境ID
    details: Optional[dict] = None
)
```

**使用场景**:
- 环境初始化失败
- 训练过程错误
- 模型保存/加载失败
- 超参数优化失败

## 错误处理装饰器

### @handle_errors 装饰器

统一的错误处理装饰器，支持同步和异步函数。

#### 参数说明

```python
@handle_errors(
    error_type: Type[QuantAError] = QuantAError,  # 捕获的异常类型
    default_return: Any = None,                    # 错误时的默认返回值
    reraise: bool = False,                         # 是否重新抛出异常
    log_level: str = "ERROR",                      # 日志级别
    context: Optional[dict] = None                 # 额外上下文信息
)
```

#### 使用示例

##### 1. 基本使用 - 同步函数

```python
from utils.error_handler import handle_errors, DataError

@handle_errors(error_type=DataError, default_return=None)
def fetch_stock_data(symbol: str):
    if not symbol:
        raise DataError("股票代码不能为空", symbol=symbol)
    # 数据获取逻辑
    return {"symbol": symbol, "price": 100.0}

# 使用
result = fetch_stock_data("")  # 返回 None，错误被记录
```

##### 2. 重新抛出异常

```python
@handle_errors(error_type=AgentError, reraise=True)
async def process_message(message: str):
    if not message:
        raise AgentError("消息不能为空")
    # 处理逻辑
    return {"status": "success"}

# 使用
try:
    await process_message("")
except AgentError as e:
    print(f"捕获异常: {e}")
```

##### 3. 异步函数

```python
@handle_errors(error_type=RLError, default_return=None)
async def train_model(algorithm: str):
    if algorithm not in ["ppo", "dqn"]:
        raise RLError("不支持的算法", algorithm=algorithm)
    # 训练逻辑
    return {"status": "success"}

# 使用
result = await train_model("invalid")  # 返回 None
```

##### 4. 带上下文信息

```python
@handle_errors(
    error_type=BacktestError,
    default_return=None,
    context={"module": "backtest_engine", "version": "1.0"}
)
def run_backtest(config: dict):
    if not config:
        raise BacktestError("配置不能为空")
    # 回测逻辑
    return {"result": "success"}
```

##### 5. 特定异常类型捕获

```python
@handle_errors(error_type=DataError, default_return="data_error")
def mixed_errors(error_type: str):
    if error_type == "data":
        raise DataError("数据错误")
    elif error_type == "agent":
        raise AgentError("Agent错误")  # 不会被捕获
    return "success"

# 使用
result = mixed_errors("data")      # 返回 "data_error"
result = mixed_errors("agent")     # 抛出 AgentError
```

## 上下文管理器

### ErrorHandler 类

用于with语句中的错误处理。

#### 参数说明

```python
ErrorHandler(
    error_type: Type[QuantAError] = QuantAError,
    reraise: bool = False,
    log_level: str = "ERROR",
    context: Optional[dict] = None
)
```

#### 使用示例

##### 1. 基本使用

```python
from utils.error_handler import ErrorHandler, DataError

with ErrorHandler() as handler:
    # 可能出错的代码
    result = some_operation()

if handler.has_error():
    print(f"发生错误: {handler.get_error()}")
```

##### 2. 不重新抛出异常

```python
with ErrorHandler(reraise=False) as handler:
    raise DataError("数据获取失败")

# 程序继续执行
print(f"错误发生: {handler.has_error()}")  # True
print(f"错误对象: {handler.get_error()}")  # DataError实例
```

##### 3. 重新抛出异常

```python
try:
    with ErrorHandler(reraise=True) as handler:
        raise DataError("数据获取失败")
except DataError as e:
    print(f"捕获异常: {e}")
```

##### 4. 带上下文信息

```python
with ErrorHandler(
    reraise=False,
    context={"module": "data_fetcher", "version": "1.0"}
) as handler:
    fetch_data()

if handler.has_error():
    error = handler.get_error()
    print(f"错误: {error}")
```

## 辅助函数

### 1. create_error

创建错误实例的辅助函数。

```python
from utils.error_handler import create_error, DataError

error = create_error(
    DataError,
    "数据获取失败",
    symbol="600000.SH",
    data_source="tushare",
    details={"timeout": 30}
)
# 返回 DataError 实例
```

### 2. format_error_for_api

将错误格式化为API响应格式。

```python
from utils.error_handler import format_error_for_api, DataError

error = DataError("数据获取失败", symbol="600000.SH")
api_response = format_error_for_api(error)

# 返回:
# {
#     "success": False,
#     "error": {
#         "error_type": "DataError",
#         "message": "数据获取失败",
#         "details": {"symbol": "600000.SH"}
#     }
# }
```

### 3. log_and_raise

记录日志并抛出异常。

```python
from utils.error_handler import log_and_raise, BacktestError

# 记录ERROR级别日志并抛出异常
log_and_raise(
    BacktestError,
    "回测执行失败",
    symbol="600000.SH",
    timestamp="2024-01-01 10:00:00",
    log_level="ERROR"
)
# 抛出 BacktestError
```

## 最佳实践

### 1. 选择合适的异常类型

```python
# ✓ 正确 - 使用特定异常类型
raise DataError("数据获取失败", symbol="600000.SH")

# ✗ 错误 - 使用通用异常
raise Exception("数据获取失败")
```

### 2. 提供详细的错误信息

```python
# ✓ 正确 - 包含上下文信息
raise DataError(
    "数据获取失败",
    data_source="tushare",
    symbol="600000.SH",
    details={"timeout": 30, "retry_count": 3}
)

# ✗ 错误 - 信息不详细
raise DataError("失败")
```

### 3. 合理使用装饰器参数

```python
# ✓ 正确 - 关键操作重新抛出异常
@handle_errors(error_type=BacktestError, reraise=True)
def execute_trade(order):
    # 交易执行逻辑
    pass

# ✓ 正确 - 非关键操作返回默认值
@handle_errors(error_type=DataError, default_return={})
def get_cached_data(symbol):
    # 缓存数据获取
    pass
```

### 4. 使用上下文管理器处理复杂逻辑

```python
# ✓ 正确 - 使用上下文管理器
with ErrorHandler(reraise=False, context={"operation": "batch_import"}) as handler:
    for symbol in symbols:
        process_symbol(symbol)
        if handler.has_error():
            # 记录错误但继续处理
            logger.warning(f"跳过 {symbol}: {handler.get_error()}")
            handler.exception_occurred = False  # 重置状态
```

### 5. 异步函数的错误处理

```python
# ✓ 正确 - 异步函数使用装饰器
@handle_errors(error_type=AgentError, default_return=None)
async def process_agent_message(message: str):
    # 异步处理逻辑
    await async_operation()
    return result
```

## 错误日志格式

错误日志包含以下信息：

1. **时间戳**: 错误发生时间
2. **模块**: 错误发生的模块和函数
3. **错误类型**: 异常类名称
4. **错误消息**: 主要错误描述
5. **详细信息**: 错误详情字典
6. **上下文**: 额外的上下文信息（如果提供）
7. **参数信息**: 函数参数的类型信息

示例日志：

```
2026-02-09 23:18:28 [ERROR] Error in __main__.fetch_stock_data | [DataError] 无效的股票代码 | Details: {'data_source': 'tushare', 'symbol': 'INVALID'} | Args types: ['str']
```

## 集成示例

### 在数据层使用

```python
# data/market/data_manager.py
from utils.error_handler import handle_errors, DataError

@handle_errors(error_type=DataError, default_return=None, reraise=True)
def fetch_daily_data(symbol: str, start_date: str, end_date: str):
    """获取日线数据"""
    try:
        data = ts.pro_bar(ts_code=symbol, start_date=start_date, end_date=end_date)
        if data is None or data.empty:
            raise DataError(
                "未获取到数据",
                symbol=symbol,
                data_source="tushare",
                details={"start_date": start_date, "end_date": end_date}
            )
        return data
    except DataError:
        raise
    except Exception as e:
        raise DataError(
            f"数据获取异常: {str(e)}",
            symbol=symbol,
            data_source="tushare"
        )
```

### 在Agent层使用

```python
# agents/base/agent_base.py
from utils.error_handler import handle_errors, AgentError

@handle_errors(error_type=AgentError, default_return=None, reraise=False)
async def process_message(self, message: str) -> Optional[dict]:
    """处理消息"""
    if not message:
        raise AgentError(
            "消息不能为空",
            agent_name=self.name,
            agent_type=self.__class__.__name__
        )

    try:
        response = await self.llm.generate(message)
        return response
    except Exception as e:
        raise AgentError(
            f"消息处理失败: {str(e)}",
            agent_name=self.name,
            agent_type=self.__class__.__name__
        )
```

### 在回测引擎使用

```python
# backtest/engine/backtest.py
from utils.error_handler import ErrorHandler, BacktestError

def execute_order(self, order: Order):
    """执行订单"""
    with ErrorHandler(reraise=True, context={"symbol": order.symbol}) as handler:
        # 验证订单
        if not self.validate_order(order):
            raise BacktestError(
                "订单验证失败",
                symbol=order.symbol,
                timestamp=self.current_time,
                details={"order": order.to_dict()}
            )

        # 执行订单
        self._execute_order_internal(order)
```

### 在RL训练使用

```python
# rl/training/trainer.py
from utils.error_handler import log_and_raise, RLError

def train(self, total_timesteps: int):
    """训练模型"""
    try:
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.callback
        )
    except Exception as e:
        log_and_raise(
            RLError,
            f"训练失败: {str(e)}",
            algorithm=self.algorithm,
            env_id=self.env.__class__.__name__,
            details={"total_timesteps": total_timesteps}
        )
```

## 测试

运行错误处理测试：

```bash
# 运行所有错误处理测试
pytest tests/utils/test_error_handler.py -v

# 运行特定测试类
pytest tests/utils/test_error_handler.py::TestHandleErrorsDecorator -v

# 运行使用示例
python examples/error_handler_usage.py
```

## 注意事项

1. **异常继承**: 所有自定义异常都继承自 `QuantAError`，可以用基类捕获所有子类异常
2. **日志级别**: 根据错误严重程度选择合适的日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
3. **性能影响**: 装饰器会有轻微的性能开销，建议在关键路径外使用
4. **异步支持**: 装饰器自动检测同步/异步函数，无需特殊处理
5. **错误详情**: 尽可能提供详细的错误信息，便于调试和监控

## 相关文件

- 实现文件: `utils/error_handler.py`
- 测试文件: `tests/utils/test_error_handler.py`
- 使用示例: `examples/error_handler_usage.py`
- 配置文件: `config/settings.py` (日志配置)

## 总结

统一错误处理机制提供了：

1. **一致的错误格式**: 所有错误都包含类型、消息和详情
2. **自动化日志记录**: 错误发生时自动记录详细信息
3. **灵活的错误处理**: 支持返回默认值或重新抛出异常
4. **同步/异步支持**: 装饰器自动适配函数类型
5. **上下文信息**: 可添加额外的上下文信息便于调试

通过使用这套机制，可以提高代码的健壮性和可维护性，便于错误追踪和问题定位。
