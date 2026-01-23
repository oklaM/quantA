"""
Backtest engine module
"""

from .a_share_rules import AShareRulesEngine
from .backtest import BacktestEngine
from .data_handler import DataHandler, SimpleDataHandler
from .event_engine import BacktestEngine as EventDrivenBacktestEngine
from .event_engine import (
    BarEvent,
    DataHandler,
    Event,
    EventQueue,
    EventType,
    ExecutionHandler,
    FillEvent,
    OrderEvent,
    Strategy,
)
from .execution import ExecutionHandler
from .indicators import TechnicalIndicators
from .portfolio import Portfolio
from .rust_engine import RustBacktestEngine
from .strategy import MovingAverageCrossStrategy, Strategy

__all__ = [
    # Event-driven engine
    "Event",
    "EventType",
    "BarEvent",
    "OrderEvent",
    "FillEvent",
    "EventQueue",
    "EventDrivenBacktestEngine",
    # Main backtest engine
    "BacktestEngine",
    "RustBacktestEngine",
    # Components
    "TechnicalIndicators",
    "AShareRulesEngine",
    "DataHandler",
    "SimpleDataHandler",
    "Strategy",
    "MovingAverageCrossStrategy",
    "Portfolio",
    "ExecutionHandler",
]
