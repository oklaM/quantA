"""
统一错误处理机制
提供自定义异常类和错误处理装饰器
"""

import asyncio
import functools
import inspect
import logging
import traceback
from typing import Any, Callable, Optional, Type, TypeVar, Union

from utils.logging import get_logger

logger = get_logger(__name__)


# ==================== 自定义异常类层次 ====================


class QuantAError(Exception):
    """
    quantA 系统基础异常类
    所有自定义异常的基类
    """

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        初始化异常

        Args:
            message: 错误消息
            details: 额外的错误详情字典
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def to_dict(self) -> dict:
        """转换为字典格式，便于日志记录和API响应"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class DataError(QuantAError):
    """
    数据层异常
    用于数据获取、存储、处理过程中的错误
    """

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化数据异常

        Args:
            message: 错误消息
            data_source: 数据源名称 (tushare, akshare, duckdb等)
            symbol: 相关股票代码
            details: 额外详情
        """
        details = details or {}
        if data_source:
            details["data_source"] = data_source
        if symbol:
            details["symbol"] = symbol
        super().__init__(message, details)


class AgentError(QuantAError):
    """
    Agent层异常
    用于LLM Agent执行过程中的错误
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化Agent异常

        Args:
            message: 错误消息
            agent_name: Agent名称
            agent_type: Agent类型 (market_data, technical, sentiment, strategy, risk)
            details: 额外详情
        """
        details = details or {}
        if agent_name:
            details["agent_name"] = agent_name
        if agent_type:
            details["agent_type"] = agent_type
        super().__init__(message, details)


class BacktestError(QuantAError):
    """
    回测引擎异常
    用于回测执行、指标计算、订单处理等错误
    """

    def __init__(
        self,
        message: str,
        symbol: Optional[str] = None,
        timestamp: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化回测异常

        Args:
            message: 错误消息
            symbol: 相关股票代码
            timestamp: 错误发生时间
            details: 额外详情
        """
        details = details or {}
        if symbol:
            details["symbol"] = symbol
        if timestamp:
            details["timestamp"] = timestamp
        super().__init__(message, details)


class RLError(QuantAError):
    """
    强化学习异常
    用于RL训练、环境、策略等错误
    """

    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        env_id: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        初始化RL异常

        Args:
            message: 错误消息
            algorithm: 算法名称 (ppo, dqn, a2c等)
            env_id: 环境ID
            details: 额外详情
        """
        details = details or {}
        if algorithm:
            details["algorithm"] = algorithm
        if env_id:
            details["env_id"] = env_id
        super().__init__(message, details)


# ==================== 错误处理装饰器 ====================


T = TypeVar("T")


def handle_errors(
    error_type: Type[QuantAError] = QuantAError,
    default_return: Any = None,
    reraise: bool = False,
    log_level: str = "ERROR",
    context: Optional[dict] = None,
) -> Callable:
    """
    统一错误处理装饰器
    支持同步和异步函数，自动记录错误日志

    Args:
        error_type: 捕获的异常类型（默认QuantAError及其子类）
        default_return: 发生错误时的默认返回值
        reraise: 是否重新抛出异常（False则返回default_return）
        log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        context: 额外的上下文信息，会记录到日志中

    Returns:
        装饰器函数

    Examples:
        >>> # 同步函数使用示例
        >>> @handle_errors(error_type=DataError, reraise=True)
        >>> def fetch_data(symbol: str):
        ...     # 数据获取逻辑
        ...     pass

        >>> # 异步函数使用示例
        >>> @handle_errors(error_type=AgentError, default_return={"status": "failed"})
        >>> async def process_message(self, message: str):
        ...     # 处理逻辑
        ...     pass

        >>> # 带上下文信息
        >>> @handle_errors(context={"module": "backtest", "version": "1.0"})
        >>> def run_backtest(config):
        ...     # 回测逻辑
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        # 判断是否是异步函数
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except error_type as e:
                # 只捕获指定类型的异常
                _handle_exception(
                    exception=e,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    error_type=error_type,
                    default_return=default_return,
                    reraise=reraise,
                    log_level=log_level,
                    context=context,
                )

                # 返回默认值或重新抛出
                if reraise:
                    raise
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except error_type as e:
                # 只捕获指定类型的异常
                _handle_exception(
                    exception=e,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    error_type=error_type,
                    default_return=default_return,
                    reraise=reraise,
                    log_level=log_level,
                    context=context,
                )

                # 返回默认值或重新抛出
                if reraise:
                    raise
                return default_return

        # 根据函数类型返回对应的wrapper
        return async_wrapper if is_async else sync_wrapper

    return decorator


def _handle_exception(
    exception: Exception,
    func: Callable,
    args: tuple,
    kwargs: dict,
    error_type: Type[QuantAError],
    default_return: Any,
    reraise: bool,
    log_level: str,
    context: Optional[dict],
) -> None:
    """
    内部异常处理逻辑
    记录日志并处理异常
    """
    # 获取日志级别对应的logger方法
    log_method = getattr(logger, log_level.lower(), logger.error)

    # 构建错误信息
    func_name = func.__qualname__
    module_name = func.__module__

    # 如果是QuantAError或其子类，直接使用
    if isinstance(exception, QuantAError):
        error_info = exception.to_dict()
        error_msg = f"[{error_info['error_type']}] {error_info['message']}"
        if error_info["details"]:
            error_msg += f" | Details: {error_info['details']}"
    else:
        # 对于其他异常，包装为QuantAError
        error_msg = str(exception)
        error_info = {
            "error_type": exception.__class__.__name__,
            "message": error_msg,
            "details": context or {},
        }

    # 构建完整的日志消息
    log_msg_text = f"Error in {module_name}.{func_name} | {error_msg}"

    # 添加上下文信息
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        log_msg_text += f" | Context: {context_str}"

    # 添加参数信息（仅记录类型，避免敏感数据）
    if args:
        arg_types = [type(arg).__name__ for arg in args]
        log_msg_text += f" | Args types: {arg_types}"
    if kwargs:
        kwarg_keys = list(kwargs.keys())
        log_msg_text += f" | Kwargs keys: {kwarg_keys}"

    # 记录日志
    log_method(log_msg_text)

    # 如果是ERROR或CRITICAL级别，记录堆栈跟踪
    if log_level.upper() in ["ERROR", "CRITICAL"]:
        logger.debug(f"Stack trace for {func_name}:\n{traceback.format_exc()}")


# ==================== 错误处理上下文管理器 ====================


class ErrorHandler:
    """
    错误处理上下文管理器
    用于with语句中的错误处理
    """

    def __init__(
        self,
        error_type: Type[QuantAError] = QuantAError,
        reraise: bool = False,
        log_level: str = "ERROR",
        context: Optional[dict] = None,
    ):
        """
        初始化错误处理器

        Args:
            error_type: 捕获的异常类型
            reraise: 是否重新抛出异常
            log_level: 日志级别
            context: 额外的上下文信息
        """
        self.error_type = error_type
        self.reraise = reraise
        self.log_level = log_level
        self.context = context
        self.exception_occurred = False
        self.exception = None

    def __enter__(self):
        """进入上下文"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if exc_type is not None:
            self.exception_occurred = True
            self.exception = exc_val

            # 处理异常
            _handle_exception(
                exception=exc_val,
                func=lambda: None,  # 占位符
                args=(),
                kwargs={},
                error_type=self.error_type,
                default_return=None,
                reraise=self.reraise,
                log_level=self.log_level,
                context=self.context,
            )

            # 如果不重新抛出，返回True抑制异常
            return not self.reraise
        return False

    def has_error(self) -> bool:
        """检查是否发生错误"""
        return self.exception_occurred

    def get_error(self) -> Optional[Exception]:
        """获取发生的异常"""
        return self.exception


# ==================== 辅助函数 ====================


def create_error(
    error_class: Type[QuantAError],
    message: str,
    **kwargs,
) -> QuantAError:
    """
    创建错误实例的辅助函数

    Args:
        error_class: 错误类
        message: 错误消息
        **kwargs: 错误类特定的参数（如symbol, agent_name等）

    Returns:
        错误实例

    Examples:
        >>> error = create_error(DataError, "数据获取失败", symbol="600000", data_source="tushare")
        >>> raise error
    """
    return error_class(message=message, **kwargs)


def format_error_for_api(error: QuantAError) -> dict:
    """
    将错误格式化为API响应格式

    Args:
        error: QuantAError实例

    Returns:
        API响应字典
    """
    return {
        "success": False,
        "error": error.to_dict(),
    }


def log_and_raise(
    error_class: Type[QuantAError],
    message: str,
    log_level: str = "ERROR",
    **kwargs,
) -> None:
    """
    记录日志并抛出异常

    Args:
        error_class: 错误类
        message: 错误消息
        log_level: 日志级别
        **kwargs: 错误类特定的参数

    Raises:
        error_class实例
    """
    error = error_class(message=message, **kwargs)

    # 记录日志
    log_method = getattr(logger, log_level.lower(), logger.error)
    log_method(f"[{error.__class__.__name__}] {error}")

    # 抛出异常
    raise error


# ==================== 导出 ====================

__all__ = [
    # 异常类
    "QuantAError",
    "DataError",
    "AgentError",
    "BacktestError",
    "RLError",
    # 装饰器
    "handle_errors",
    # 上下文管理器
    "ErrorHandler",
    # 辅助函数
    "create_error",
    "format_error_for_api",
    "log_and_raise",
]
