"""
工具模块
提供日志、时间处理、辅助函数、错误处理等
"""

import logging

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
from utils.helpers import (
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_volatility,
    format_money,
    format_number,
    normalize,
    retry_on_exception,
    timing_decorator,
)
from utils.logging import get_logger, logger, setup_logger
from utils.time_utils import (
    get_current_trading_session,
    get_next_trading_day,
    get_previous_trading_day,
    get_trading_days,
    get_trading_sessions,
    is_trading_day,
    is_trading_time,
    parse_period,
)

__all__ = [
    # logging
    "setup_logger",
    "get_logger",
    "logger",
    # error_handler
    "QuantAError",
    "DataError",
    "AgentError",
    "BacktestError",
    "RLError",
    "handle_errors",
    "ErrorHandler",
    "create_error",
    "format_error_for_api",
    "log_and_raise",
    # time_utils
    "is_trading_day",
    "is_trading_time",
    "get_trading_days",
    "get_previous_trading_day",
    "get_next_trading_day",
    "get_trading_sessions",
    "get_current_trading_session",
    "parse_period",
    # helpers
    "timing_decorator",
    "format_number",
    "format_money",
    "calculate_returns",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_volatility",
    "normalize",
    "retry_on_exception",
]
