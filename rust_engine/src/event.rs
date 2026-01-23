/*!
事件系统
实现事件驱动架构的核心组件
*/

use crate::error::{EngineError, EngineResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

/// 事件类型枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventType {
    /// 市场数据事件
    Market,
    /// 信号事件
    Signal,
    /// 订单事件
    Order,
    /// 成交事件
    Fill,
    /// 自定义事件
    Custom(String),
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventType::Market => write!(f, "Market"),
            EventType::Signal => write!(f, "Signal"),
            EventType::Order => write!(f, "Order"),
            EventType::Fill => write!(f, "Fill"),
            EventType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// 事件trait
pub trait Event: fmt::Debug {
    /// 获取事件时间
    fn timestamp(&self) -> DateTime<Utc>;

    /// 获取事件类型
    fn event_type(&self) -> EventType;

    /// 获取事件关联的股票代码
    fn symbol(&self) -> &str;
}

/// 市场数据事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 股票代码
    pub symbol: String,
    /// 开盘价
    pub open: f64,
    /// 最高价
    pub high: f64,
    /// 最低价
    pub low: f64,
    /// 收盘价
    pub close: f64,
    /// 成交量
    pub volume: u64,
    /// 成交额
    pub amount: Option<f64>,
}

impl MarketEvent {
    /// 创建新的市场事件
    pub fn new(
        timestamp: DateTime<Utc>,
        symbol: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: u64,
    ) -> Self {
        Self {
            timestamp,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            amount: None,
        }
    }

    /// 验证数据有效性
    pub fn validate(&self) -> EngineResult<()> {
        if self.open < 0.0 || self.high < 0.0 || self.low < 0.0 || self.close < 0.0 {
            return Err(EngineError::DataError(
                "价格不能为负数".to_string(),
            ));
        }

        if self.high < self.low {
            return Err(EngineError::DataError(format!(
                "最高价({})不能低于最低价({})",
                self.high, self.low
            )));
        }

        if self.close > self.high || self.close < self.low {
            return Err(EngineError::DataError(
                "收盘价必须在最高价和最低价之间".to_string(),
            ));
        }

        Ok(())
    }
}

impl Event for MarketEvent {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> EventType {
        EventType::Market
    }

    fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// 信号类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SignalType {
    /// 买入信号
    Buy,
    /// 卖出信号
    Sell,
    /// 持有信号
    Hold,
}

/// 信号事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalEvent {
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 股票代码
    pub symbol: String,
    /// 信号类型
    pub signal_type: SignalType,
    /// 建议数量
    pub quantity: u64,
    /// 信号强度 (0.0 - 1.0)
    pub strength: f64,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
}

impl SignalEvent {
    /// 创建新的信号事件
    pub fn new(
        timestamp: DateTime<Utc>,
        symbol: String,
        signal_type: SignalType,
        quantity: u64,
    ) -> Self {
        Self {
            timestamp,
            symbol,
            signal_type,
            quantity,
            strength: 1.0,
            metadata: None,
        }
    }

    /// 设置信号强度
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// 设置元数据
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

impl Event for SignalEvent {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> EventType {
        EventType::Signal
    }

    fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// 订单方向
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderDirection {
    /// 买入
    Buy,
    /// 卖出
    Sell,
}

/// 订单类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    /// 市价单
    Market,
    /// 限价单
    Limit,
    /// 止损单
    StopLoss,
}

/// 订单事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderEvent {
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 订单ID
    pub order_id: String,
    /// 股票代码
    pub symbol: String,
    /// 订单方向
    pub direction: OrderDirection,
    /// 订单类型
    pub order_type: OrderType,
    /// 数量
    pub quantity: u64,
    /// 价格（限价单）
    pub limit_price: Option<f64>,
    /// 止损价格
    pub stop_price: Option<f64>,
}

impl OrderEvent {
    /// 创建新的订单事件
    pub fn new(
        timestamp: DateTime<Utc>,
        order_id: String,
        symbol: String,
        direction: OrderDirection,
        order_type: OrderType,
        quantity: u64,
    ) -> Self {
        Self {
            timestamp,
            order_id,
            symbol,
            direction,
            order_type,
            quantity,
            limit_price: None,
            stop_price: None,
        }
    }

    /// 设置限价
    pub fn with_limit_price(mut self, price: f64) -> Self {
        self.limit_price = Some(price);
        self
    }

    /// 设置止损价
    pub fn with_stop_price(mut self, price: f64) -> Self {
        self.stop_price = Some(price);
        self
    }

    /// 验证订单有效性
    pub fn validate(&self) -> EngineResult<()> {
        if self.quantity == 0 {
            return Err(EngineError::OrderError("订单数量不能为0".to_string()));
        }

        if let Some(limit_price) = self.limit_price {
            if limit_price <= 0.0 {
                return Err(EngineError::OrderError(
                    "限价必须大于0".to_string(),
                ));
            }
        }

        if let Some(stop_price) = self.stop_price {
            if stop_price <= 0.0 {
                return Err(EngineError::OrderError(
                    "止损价必须大于0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Event for OrderEvent {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> EventType {
        EventType::Order
    }

    fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// 成交事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    /// 时间戳
    pub timestamp: DateTime<Utc>,
    /// 订单ID
    pub order_id: String,
    /// 股票代码
    pub symbol: String,
    /// 成交方向
    pub direction: OrderDirection,
    /// 成交数量
    pub quantity: u64,
    /// 成交价格
    pub price: f64,
    /// 手续费
    pub commission: f64,
    /// 成交金额
    pub fill_amount: f64,
}

impl FillEvent {
    /// 创建新的成交事件
    pub fn new(
        timestamp: DateTime<Utc>,
        order_id: String,
        symbol: String,
        direction: OrderDirection,
        quantity: u64,
        price: f64,
        commission: f64,
    ) -> Self {
        let fill_amount = (quantity as f64) * price;
        Self {
            timestamp,
            order_id,
            symbol,
            direction,
            quantity,
            price,
            commission,
            fill_amount,
        }
    }

    /// 计算净成交额（扣除手续费）
    pub fn net_amount(&self) -> f64 {
        match self.direction {
            OrderDirection::Buy => self.fill_amount + self.commission,
            OrderDirection::Sell => self.fill_amount - self.commission,
        }
    }
}

impl Event for FillEvent {
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> EventType {
        EventType::Fill
    }

    fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// 事件队列
#[derive(Debug, Clone)]
pub struct EventQueue {
    events: Vec<Box<dyn Event + Send + Sync>>,
}

impl EventQueue {
    /// 创建新的事件队列
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
        }
    }

    /// 添加事件
    pub fn push(&mut self, event: Box<dyn Event + Send + Sync>) {
        self.events.push(event);
    }

    /// 获取下一个事件（按时间顺序）
    pub fn pop(&mut self) -> Option<Box<dyn Event + Send + Sync>> {
        if self.events.is_empty() {
            return None;
        }

        // 找到最早的事件
        let min_index = self
            .events
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.timestamp().cmp(&b.timestamp()))
            .map(|(i, _)| i)?;

        Some(self.events.remove(min_index))
    }

    /// 检查队列是否为空
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// 获取事件数量
    pub fn len(&self) -> usize {
        self.events.len()
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_event_validation() {
        let event = MarketEvent::new(
            Utc::now(),
            "600000.SH".to_string(),
            10.0,
            10.5,
            9.5,
            10.3,
            1000000,
        );

        assert!(event.validate().is_ok());
    }

    #[test]
    fn test_market_event_invalid_prices() {
        let event = MarketEvent::new(
            Utc::now(),
            "600000.SH".to_string(),
            10.0,
            9.5, // high < low
            9.5,
            10.3,
            1000000,
        );

        assert!(event.validate().is_err());
    }

    #[test]
    fn test_signal_event_strength_clamping() {
        let signal = SignalEvent::new(
            Utc::now(),
            "600000.SH".to_string(),
            SignalType::Buy,
            1000,
        )
        .with_strength(1.5); // 超过1.0

        assert_eq!(signal.strength, 1.0);
    }

    #[test]
    fn test_order_validation() {
        let order = OrderEvent::new(
            Utc::now(),
            "order_1".to_string(),
            "600000.SH".to_string(),
            OrderDirection::Buy,
            OrderType::Market,
            1000,
        );

        assert!(order.validate().is_ok());

        let invalid_order = OrderEvent::new(
            Utc::now(),
            "order_2".to_string(),
            "600000.SH".to_string(),
            OrderDirection::Buy,
            OrderType::Market,
            0, // 数量为0
        );

        assert!(invalid_order.validate().is_err());
    }

    #[test]
    fn test_event_queue() {
        let mut queue = EventQueue::new();

        let event1 = Box::new(MarketEvent::new(
            Utc::now(),
            "600000.SH".to_string(),
            10.0,
            10.5,
            9.5,
            10.3,
            1000000,
        )) as Box<dyn Event + Send + Sync>;

        let event2 = Box::new(SignalEvent::new(
            Utc::now(),
            "600000.SH".to_string(),
            SignalType::Buy,
            1000,
        )) as Box<dyn Event + Send + Sync>;

        queue.push(event1);
        queue.push(event2);

        assert_eq!(queue.len(), 2);
        assert!(!queue.is_empty());

        let _ = queue.pop();
        assert_eq!(queue.len(), 1);
    }
}
