/*!
Python绑定模块
使用PyO3为Python提供FFI接口
*/

use crate::{
    event::{EventQueue, MarketEvent, SignalType},
    BacktestEngine,
    order_manager::{OrderManager, ThreadSafeOrderManager, OrderStats},
    order_book::{OrderBook, ThreadSafeOrderBook, OrderBookDepth},
    execution::{Order, OrderSide, OrderType, OrderStatus},
};
use chrono::{DateTime, NaiveDateTime, Utc};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Python包装的市场数据
#[pyclass(name = "MarketData")]
pub struct PyMarketData {
    pub inner: HashMap<String, Vec<MarketEvent>>,
}

#[pymethods]
impl PyMarketData {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn add_bar(
        &mut self,
        symbol: String,
        timestamp: i64,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
    ) -> PyResult<()> {
        let dt = DateTime::from_timestamp(timestamp, 0)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid timestamp"))?;

        let event = MarketEvent::new(dt, symbol.clone(), open, high, low, close, volume);

        self.inner
            .entry(symbol)
            .or_insert_with(Vec::new)
            .push(event);

        Ok(())
    }
}

/// Python包装的回测引擎
#[pyclass(name = "BacktestEngine")]
pub struct PyBacktestEngine {
    pub inner: BacktestEngine,
}

#[pymethods]
impl PyBacktestEngine {
    #[new]
    #[pyo3(signature = (initial_cash=1000000.0, commission=0.0003, slippage=0.0001))]
    pub fn new(initial_cash: f64, commission: f64, slippage: f64) -> Self {
        Self {
            inner: BacktestEngine::new(initial_cash, commission, slippage),
        }
    }

    /// 运行回测
    pub fn run_backtest(
        &mut self,
        data: &PyMarketData,
        strategy_fn: PyObject,
    ) -> PyResult<PyDict> {
        Python::with_gil(|py| {
            // 转换数据
            let mut engine = self.inner.clone();
            engine.load_data(&data.inner).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
            })?;

            // 运行回测
            let results = engine
                .run()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // 转换结果为Python字典
            let dict = PyDict::new(py);
            dict.set_item("total_return", results.total_return)?;
            dict.set_item("sharpe_ratio", results.sharpe_ratio)?;
            dict.set_item("max_drawdown", results.max_drawdown)?;
            dict.set_item("total_trades", results.total_trades)?;

            Ok(dict.into())
        })
    }

    /// 获取性能指标
    pub fn get_metrics(&self) -> PyResult<PyDict> {
        Python::with_gil(|py| {
            let metrics = self.inner.get_metrics();
            let dict = PyDict::new(py);

            dict.set_item("total_equity", metrics.total_equity)?;
            dict.set_item("available_cash", metrics.available_cash)?;
            dict.set_item("total_profit", metrics.total_profit)?;

            Ok(dict.into())
        })
    }
}

/// Python包装的事件
#[pyclass(name = "Event")]
pub struct PyEvent {
    #[pyo3(get)]
    pub timestamp: i64,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub event_type: String,
}

/// Python包装的投资组合
#[pyclass(name = "Portfolio")]
pub struct PyPortfolio {
    inner: crate::portfolio::Portfolio,
}

#[pymethods]
impl PyPortfolio {
    #[getter]
    pub fn total_value(&self) -> f64 {
        self.inner.total_value()
    }

    #[getter]
    pub fn available_cash(&self) -> f64 {
        self.inner.available_cash()
    }

    pub fn positions(&self) -> PyResult<Vec<(String, u64, f64)>> {
        Ok(self
            .inner
            .get_positions()
            .iter()
            .map(|p| (p.symbol.clone(), p.quantity, p.avg_price))
            .collect())
    }
}

/// Python包装的订单方向
#[pyclass(name = "OrderSide")]
#[derive(Clone, Copy)]
pub enum PyOrderSide {
    Buy,
    Sell,
}

impl From<PyOrderSide> for OrderSide {
    fn from(side: PyOrderSide) -> Self {
        match side {
            PyOrderSide::Buy => OrderSide::Buy,
            PyOrderSide::Sell => OrderSide::Sell,
        }
    }
}

impl From<OrderSide> for PyOrderSide {
    fn from(side: OrderSide) -> Self {
        match side {
            OrderSide::Buy => PyOrderSide::Buy,
            OrderSide::Sell => PyOrderSide::Sell,
        }
    }
}

/// Python包装的订单类型
#[pyclass(name = "OrderType")]
#[derive(Clone, Copy)]
pub enum PyOrderType {
    Market,
    Limit,
}

impl From<PyOrderType> for OrderType {
    fn from(order_type: PyOrderType) -> Self {
        match order_type {
            PyOrderType::Market => OrderType::Market,
            PyOrderType::Limit => OrderType::Limit,
        }
    }
}

/// Python包装的订单状态
#[pyclass(name = "OrderStatus")]
#[derive(Clone, Copy)]
pub enum PyOrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Python包装的订单
#[pyclass(name = "Order")]
pub struct PyOrder {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: PyOrderSide,
    #[pyo3(get)]
    pub order_type: PyOrderType,
    #[pyo3(get)]
    pub quantity: i64,
    #[pyo3(get)]
    pub price: Option<f64>,
    #[pyo3(get)]
    pub filled_quantity: i64,
    #[pyo3(get)]
    pub avg_fill_price: Option<f64>,
    #[pyo3(get)]
    pub status: PyOrderStatus,
}

impl From<Order> for PyOrder {
    fn from(order: Order) -> Self {
        PyOrder {
            id: order.id,
            symbol: order.symbol,
            side: PyOrderSide::from(order.side),
            order_type: match order.order_type {
                OrderType::Market => PyOrderType::Market,
                OrderType::Limit => PyOrderType::Limit,
                _ => PyOrderType::Market, // 简化处理
            },
            quantity: order.quantity,
            price: order.price.map(|p| p.to_string().parse().unwrap_or(0.0)),
            filled_quantity: order.filled_quantity,
            avg_fill_price: order.avg_fill_price.map(|p| p.to_string().parse().unwrap_or(0.0)),
            status: match order.status {
                OrderStatus::Pending => PyOrderStatus::Pending,
                OrderStatus::PartiallyFilled => PyOrderStatus::PartiallyFilled,
                OrderStatus::Filled => PyOrderStatus::Filled,
                OrderStatus::Cancelled => PyOrderStatus::Cancelled,
                OrderStatus::Rejected => PyOrderStatus::Rejected,
                _ => PyOrderStatus::Pending,
            },
        }
    }
}

/// Python包装的订单管理器
#[pyclass(name = "OrderManager")]
pub struct PyOrderManager {
    inner: ThreadSafeOrderManager,
}

#[pymethods]
impl PyOrderManager {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ThreadSafeOrderManager::new(),
        }
    }

    /// 创建订单
    #[pyo3(signature = (symbol, side, order_type, quantity, price=None))]
    pub fn create_order(
        &self,
        symbol: String,
        side: PyOrderSide,
        order_type: PyOrderType,
        quantity: i64,
        price: Option<f64>,
    ) -> PyResult<String> {
        let price_decimal = price.map(|p| rust_decimal::Decimal::from_str(p.to_string().as_str()).unwrap());

        self.inner
            .create_order(symbol, side.into(), order_type.into(), quantity, price_decimal)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// 获取订单
    pub fn get_order(&self, order_id: &str) -> PyResult<Option<PyOrder>> {
        self.inner
            .get_order(order_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            .map(|opt| opt.map(PyOrder::from))
    }

    /// 取消订单
    pub fn cancel_order(&self, order_id: &str) -> PyResult<bool> {
        self.inner
            .cancel_order(order_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// 获取活跃订单
    pub fn get_active_orders(&self) -> PyResult<Vec<PyOrder>> {
        self.inner
            .get_active_orders()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
            .map(|orders| orders.into_iter().map(PyOrder::from).collect())
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> PyResult<PyDict> {
        Python::with_gil(|py| {
            let stats = self.inner.get_stats()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let dict = PyDict::new(py);
            dict.set_item("total_orders", stats.total_orders)?;
            dict.set_item("active_orders", stats.active_orders)?;
            dict.set_item("filled_orders", stats.filled_orders)?;
            dict.set_item("cancelled_orders", stats.cancelled_orders)?;
            dict.set_item("rejected_orders", stats.rejected_orders)?;

            Ok(dict.into())
        })
    }
}

/// Python包装的订单簿
#[pyclass(name = "OrderBook")]
pub struct PyOrderBook {
    inner: ThreadSafeOrderBook,
}

#[pymethods]
impl PyOrderBook {
    #[new]
    pub fn new(symbol: String) -> Self {
        Self {
            inner: ThreadSafeOrderBook::new(symbol),
        }
    }

    /// 获取深度数据
    #[pyo3(signature = (depth=5))]
    pub fn get_depth(&self, depth: usize) -> PyResult<PyDict> {
        Python::with_gil(|py| {
            let depth_data = self.inner.get_depth(depth)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let dict = PyDict::new(py);
            dict.set_item("symbol", &depth_data.symbol)?;

            // 转换买盘
            let bids: Vec<PyDict> = depth_data.bids.iter().map(|level| {
                let level_dict = PyDict::new(py);
                level_dict.set_item("price", level.price.to_string().parse::<f64>().unwrap_or(0.0)).unwrap();
                level_dict.set_item("total_quantity", level.total_quantity).unwrap();
                level_dict.set_item("order_count", level.order_count).unwrap();
                level_dict
            }).collect();
            dict.set_item("bids", bids)?;

            // 转换卖盘
            let asks: Vec<PyDict> = depth_data.asks.iter().map(|level| {
                let level_dict = PyDict::new(py);
                level_dict.set_item("price", level.price.to_string().parse::<f64>().unwrap_or(0.0)).unwrap();
                level_dict.set_item("total_quantity", level.total_quantity).unwrap();
                level_dict.set_item("order_count", level.order_count).unwrap();
                level_dict
            }).collect();
            dict.set_item("asks", asks)?;

            dict.set_item("last_price", depth_data.last_price.map(|p| p.to_string().parse::<f64>().unwrap()))?;
            dict.set_item("last_quantity", depth_data.last_quantity)?;

            Ok(dict.into())
        })
    }

    /// 获取最优买卖价
    pub fn get_best_bid_ask(&self) -> PyResult<(Option<f64>, Option<f64>)> {
        let (bid, ask) = self.inner.get_best_bid_ask()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let bid_f64 = bid.map(|p| p.to_string().parse::<f64>().unwrap_or(0.0));
        let ask_f64 = ask.map(|p| p.to_string().parse::<f64>().unwrap_or(0.0));

        Ok((bid_f64, ask_f64))
    }
}
