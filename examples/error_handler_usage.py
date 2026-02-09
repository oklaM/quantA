"""
错误处理机制使用示例
演示如何使用统一的错误处理机制
"""

import asyncio
from utils.error_handler import (
    AgentError,
    BacktestError,
    DataError,
    ErrorHandler,
    QuantAError,
    RLError,
    create_error,
    format_error_for_api,
    handle_errors,
    log_and_raise,
)


# ==================== 示例 1: 基本异常使用 ====================

def example_basic_exceptions():
    """演示基本异常使用"""
    print("=== 示例 1: 基本异常使用 ===\n")

    # 1. 创建基本异常
    try:
        raise QuantAError("系统发生错误")
    except QuantAError as e:
        print(f"捕获到异常: {e}")
        print(f"异常详情: {e.to_dict()}\n")

    # 2. 创建带详情的异常
    try:
        raise DataError(
            "数据获取失败",
            data_source="tushare",
            symbol="600000.SH",
        )
    except DataError as e:
        print(f"捕获到数据异常: {e}")
        print(f"异常类型: {e.__class__.__name__}")
        print(f"详细信息: {e.details}\n")


# ==================== 示例 2: 装饰器使用 ====================

@handle_errors(error_type=DataError, default_return=None, reraise=False)
def fetch_stock_data(symbol: str):
    """模拟获取股票数据"""
    if symbol == "INVALID":
        raise DataError(
            "无效的股票代码",
            symbol=symbol,
            data_source="tushare",
        )
    return {"symbol": symbol, "price": 100.0}


@handle_errors(error_type=AgentError, reraise=True)
async def process_agent_message(agent_name: str, message: str):
    """模拟Agent处理消息"""
    if not message:
        raise AgentError(
            "消息不能为空",
            agent_name=agent_name,
            agent_type="strategy",
        )
    return f"处理结果: {message}"


def example_decorator_usage():
    """演示装饰器使用"""
    print("=== 示例 2: 装饰器使用 ===\n")

    # 1. 不重新抛出异常，返回默认值
    result = fetch_stock_data("INVALID")
    print(f"获取无效股票数据 (返回默认值): {result}\n")

    # 2. 正常情况
    result = fetch_stock_data("600000.SH")
    print(f"获取有效股票数据: {result}\n")

    # 3. 异步函数 - 重新抛出异常
    try:
        asyncio.run(process_agent_message("strategy_agent", ""))
    except AgentError as e:
        print(f"捕获到Agent异常: {e}\n")


# ==================== 示例 3: 上下文管理器使用 ====================

def example_context_manager():
    """演示上下文管理器使用"""
    print("=== 示例 3: 上下文管理器使用 ===\n")

    # 1. 无错误情况
    with ErrorHandler() as handler:
        result = 1 + 1
    print(f"无错误 - has_error: {handler.has_error()}\n")

    # 2. 有错误但不重新抛出
    with ErrorHandler(reraise=False, context={"module": "test"}) as handler:
        raise BacktestError("回测失败", symbol="600000.SH")

    print(f"有错误 - has_error: {handler.has_error()}")
    print(f"错误类型: {handler.get_error().__class__.__name__}")
    print(f"错误消息: {handler.get_error()}\n")

    # 3. 有错误并重新抛出
    try:
        with ErrorHandler(reraise=True) as handler:
            raise RLError("训练失败", algorithm="ppo")
    except RLError as e:
        print(f"重新抛出的异常: {e}\n")


# ==================== 示例 4: 辅助函数使用 ====================

def example_helper_functions():
    """演示辅助函数使用"""
    print("=== 示例 4: 辅助函数使用 ===\n")

    # 1. create_error - 创建错误实例
    error = create_error(
        DataError,
        "数据获取超时",
        symbol="600000.SH",
        data_source="akshare",
        details={"timeout": 30, "retry_count": 3},
    )
    print(f"创建的错误: {error}\n")

    # 2. format_error_for_api - 格式化为API响应
    api_response = format_error_for_api(error)
    print("API响应格式:")
    print(f"  成功: {api_response['success']}")
    print(f"  错误类型: {api_response['error']['error_type']}")
    print(f"  错误消息: {api_response['error']['message']}")
    print(f"  详细信息: {api_response['error']['details']}\n")

    # 3. log_and_raise - 记录日志并抛出
    try:
        log_and_raise(
            BacktestError,
            "回测执行失败",
            symbol="600000.SH",
            timestamp="2024-01-01 10:00:00",
            log_level="ERROR",
        )
    except BacktestError as e:
        print(f"log_and_raise 抛出的异常: {e}\n")


# ==================== 示例 5: 实际应用场景 ====================

@handle_errors(
    error_type=DataError,
    default_return={"status": "error", "data": None},
    reraise=False,
    context={"module": "data_fetcher", "version": "1.0"},
)
def fetch_market_data(symbols: list):
    """
    实际应用：获取市场数据
    演示如何在实际函数中使用错误处理装饰器
    """
    if not symbols:
        raise DataError(
            "股票代码列表不能为空",
            details={"provided_symbols": symbols},
        )

    if "INVALID" in symbols:
        raise DataError(
            "包含无效的股票代码",
            data_source="tushare",
            details={"invalid_symbols": ["INVALID"]},
        )

    return {
        "status": "success",
        "data": {symbol: {"price": 100.0} for symbol in symbols},
    }


@handle_errors(error_type=BacktestError, reraise=True)
def run_backtest(strategy_config: dict, symbols: list):
    """
    实际应用：运行回测
    演示如何在关键操作中启用异常重新抛出
    """
    if not strategy_config:
        raise BacktestError(
            "策略配置不能为空",
            details={"config": strategy_config},
        )

    if not symbols:
        raise BacktestError(
            "回测股票列表不能为空",
            details={"symbols_count": len(symbols)},
        )

    # 模拟回测逻辑
    return {
        "status": "success",
        "result": {"total_return": 0.15, "sharpe_ratio": 1.5},
    }


def example_real_world_usage():
    """演示实际应用场景"""
    print("=== 示例 5: 实际应用场景 ===\n")

    # 1. 正常数据获取
    result = fetch_market_data(["600000.SH", "000001.SZ"])
    print(f"正常数据获取: {result['status']}\n")

    # 2. 空列表错误（返回默认值）
    result = fetch_market_data([])
    print(f"空列表错误处理: {result['status']}\n")

    # 3. 回测失败（重新抛出异常）
    try:
        run_backtest({}, [])
    except BacktestError as e:
        print(f"回测失败捕获: {e}")
        print(f"错误详情: {e.details}\n")


# ==================== 示例 6: 异步函数错误处理 ====================

@handle_errors(error_type=AgentError, default_return=None, reraise=False)
async def agent_coordinate(agents: list, task: str):
    """
    异步函数：Agent协调
    演示异步函数的错误处理
    """
    if not agents:
        raise AgentError(
            "Agent列表不能为空",
            agent_type="coordinator",
            details={"task": task},
        )

    # 模拟异步处理
    await asyncio.sleep(0.1)
    return {"status": "success", "agents_count": len(agents)}


async def example_async_usage():
    """演示异步函数错误处理"""
    print("=== 示例 6: 异步函数错误处理 ===\n")

    # 1. 正常情况
    result = await agent_coordinate(["agent1", "agent2"], "分析市场")
    print(f"正常协调: {result}\n")

    # 2. 错误情况
    result = await agent_coordinate([], "分析市场")
    print(f"错误处理 (返回默认值): {result}\n")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    example_basic_exceptions()
    example_decorator_usage()
    example_context_manager()
    example_helper_functions()
    example_real_world_usage()
    asyncio.run(example_async_usage())

    print("=== 所有示例运行完成 ===")


if __name__ == "__main__":
    main()
