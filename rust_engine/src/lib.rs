/*!
quantA Rust回测引擎
高性能事件驱动回测引擎
*/

pub mod engine;
pub mod event;
pub mod data;
pub mod portfolio;
pub mod execution;
pub mod indicators;
pub mod strategy;
pub mod analysis;
pub mod error;
pub mod python;
pub mod order_manager;
pub mod order_book;

// Re-export main types
pub use engine::BacktestEngine;
pub use event::{Event, EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent};
pub use data::MarketData;
pub use portfolio::{Portfolio, Position, PortfolioMetrics};
pub use execution::{ExecutionHandler, ExecutionResult};
pub use indicators::{Indicator, IndicatorResult};
pub use error::{EngineError, EngineResult};
pub use python::{
    PyBacktestEngine,
    PyOrderManager,
    PyOrderBook,
    PyOrder,
    PyOrderSide,
    PyOrderType,
    PyOrderStatus,
};

use pyo3::prelude::*;

/// Python模块定义
#[pymodule]
fn quanta_rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBacktestEngine>()?;
    m.add_class::<python::PyMarketData>()?;
    m.add_class::<python::PyEvent>()?;
    m.add_class::<python::PyPortfolio>()?;
    m.add_class::<PyOrderManager>()?;
    m.add_class::<PyOrderBook>()?;
    m.add_class::<PyOrder>()?;
    m.add_class::<PyOrderSide>()?;
    m.add_class::<PyOrderType>()?;
    m.add_class::<PyOrderStatus>()?;
    Ok(())
}
