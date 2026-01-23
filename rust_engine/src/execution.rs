/*!
订单执行模块

负责订单的执行、路由和管理
*/

use crate::error::{EngineError, EngineResult};
use crate::event::{OrderEvent, FillEvent};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::DateTime;
use rust_decimal::Decimal;

/// 订单状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// 待执行
    Pending,
    /// 部分成交
    PartiallyFilled,
    /// 全部成交
    Filled,
    /// 已取消
    Cancelled,
    /// 拒绝
    Rejected,
    /// 过期
    Expired,
}

/// 订单方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    /// 买入
    Buy,
    /// 卖出
    Sell,
}

/// 订单类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    /// 市价单
    Market,
    /// 限价单
    Limit,
    /// 止损单
    StopLoss,
    /// 止损限价单
    StopLimit,
}

/// 订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// 订单ID
    pub id: String,
    /// 股票代码
    pub symbol: String,
    /// 方向
    pub side: OrderSide,
    /// 类型
    pub order_type: OrderType,
    /// 数量
    pub quantity: i64,
    /// 价格（限价单）
    pub price: Option<Decimal>,
    /// 止损价
    pub stop_price: Option<Decimal>,
    /// 状态
    pub status: OrderStatus,
    /// 已成交数量
    pub filled_quantity: i64,
    /// 平均成交价
    pub avg_fill_price: Option<Decimal>,
    /// 创建时间
    pub created_at: DateTime<chrono::Utc>,
    /// 更新时间
    pub updated_at: DateTime<chrono::Utc>,
}

impl Order {
    /// 创建新订单
    pub fn new(
        id: String,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: i64,
        price: Option<Decimal>,
    ) -> Self {
        let now = chrono::Utc::now();
        Order {
            id,
            symbol,
            side,
            order_type,
            quantity,
            price,
            stop_price: None,
            status: OrderStatus::Pending,
            filled_quantity: 0,
            avg_fill_price: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// �剩余数量
    pub fn remaining_quantity(&self) -> i64 {
        self.quantity - self.filled_quantity
    }

    /// 是否已完成
    pub fn is_done(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected | OrderStatus::Expired
        )
    }
}

/// 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// 订单ID
    pub order_id: String,
    /// 成交数量
    pub filled_quantity: i64,
    /// 成交价格
    pub fill_price: Decimal,
    /// 手续费
    pub commission: Decimal,
    /// 成交时间
    pub timestamp: DateTime<chrono::Utc>,
}

/// 执行处理器trait
pub trait ExecutionHandler {
    /// 执行订单
    fn execute_order(&mut self, order: &Order) -> EngineResult<Vec<ExecutionResult>>;

    /// 取消订单
    fn cancel_order(&mut self, order_id: &str) -> EngineResult<bool>;

    /// 查询订单状态
    fn get_order_status(&self, order_id: &str) -> EngineResult<Option<OrderStatus>>;
}

/// 模拟执行器（用于回测）
pub struct SimulatedExecutionHandler {
    /// 佣金率
    commission_rate: Decimal,
    /// 最小手续费
    min_commission: Decimal,
    /// 活跃订单
    active_orders: HashMap<String, Order>,
}

impl SimulatedExecutionHandler {
    /// 创建模拟执行器
    pub fn new(commission_rate: Decimal, min_commission: Decimal) -> Self {
        SimulatedExecutionHandler {
            commission_rate,
            min_commission,
            active_orders: HashMap::new(),
        }
    }

    /// 计算手续费
    fn calculate_commission(&self, price: Decimal, quantity: i64) -> Decimal {
        let amount = price * Decimal::from(quantity);
        let commission = amount * self.commission_rate;
        commission.max(self.min_commission)
    }
}

impl ExecutionHandler for SimulatedExecutionHandler {
    fn execute_order(&mut self, order: &Order) -> EngineResult<Vec<ExecutionResult>> {
        // 市价单立即成交
        if order.order_type == OrderType::Market {
            let execution_price = order.price.unwrap_or(Decimal::ZERO); // 在回测中应该传入市价
            let commission = self.calculate_commission(execution_price, order.quantity);

            let result = ExecutionResult {
                order_id: order.id.clone(),
                filled_quantity: order.quantity,
                fill_price: execution_price,
                commission,
                timestamp: chrono::Utc::now(),
            };

            return Ok(vec![result]);
        }

        // 限价单需要记录到活跃订单中
        if order.order_type == OrderType::Limit {
            self.active_orders.insert(order.id.clone(), order.clone());
            return Ok(vec![]); // 限价单不会立即成交
        }

        Err(EngineError::OrderError("不支持的订单类型".to_string()))
    }

    fn cancel_order(&mut self, order_id: &str) -> EngineResult<bool> {
        Ok(self.active_orders.remove(order_id).is_some())
    }

    fn get_order_status(&self, order_id: &str) -> EngineResult<Option<OrderStatus>> {
        Ok(self.active_orders.get(order_id).map(|o| o.status))
    }
}

/// 风险检查
pub struct RiskChecker {
    /// 最大持仓比例
    max_position_ratio: Decimal,
    /// 最大单笔订单比例
    max_single_order_ratio: Decimal,
    /// 最大日亏损比例
    max_daily_loss_ratio: Decimal,
}

impl RiskChecker {
    /// 创建风险检查器
    pub fn new(
        max_position_ratio: Decimal,
        max_single_order_ratio: Decimal,
        max_daily_loss_ratio: Decimal,
    ) -> Self {
        RiskChecker {
            max_position_ratio,
            max_single_order_ratio,
            max_daily_loss_ratio,
        }
    }

    /// 检查订单是否通过风险控制
    pub fn check_order(&self, order: &Order, portfolio_value: Decimal) -> EngineResult<bool> {
        // 检查订单金额
        if let Some(price) = order.price {
            let order_value = price * Decimal::from(order.quantity);
            let order_ratio = order_value / portfolio_value;

            if order_ratio > self.max_single_order_ratio {
                return Err(EngineError::OrderError(format!(
                    "订单金额超过限制: {:.2}% > {:.2}%",
                    order_ratio * Decimal::from(100),
                    self.max_single_order_ratio * Decimal::from(100)
                )));
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            "order1".to_string(),
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            1000,
            Some(Decimal::from(10)),
        );

        assert_eq!(order.id, "order1");
        assert_eq!(order.quantity, 1000);
        assert_eq!(order.status, OrderStatus::Pending);
        assert_eq!(order.remaining_quantity(), 1000);
    }

    #[test]
    fn test_simulated_execution() {
        let mut handler = SimulatedExecutionHandler::new(
            Decimal::from_str("0.0003").unwrap(),
            Decimal::from(5),
        );

        let order = Order::new(
            "order1".to_string(),
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            1000,
            Some(Decimal::from(10)),
        );

        let results = handler.execute_order(&order).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].filled_quantity, 1000);
        assert_eq!(results[0].fill_price, Decimal::from(10));
    }

    #[test]
    fn test_risk_checker() {
        let checker = RiskChecker::new(
            Decimal::from_str("0.95").unwrap(),
            Decimal::from_str("0.20").unwrap(),
            Decimal::from_str("0.05").unwrap(),
        );

        let order = Order::new(
            "order1".to_string(),
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Market,
            1000,
            Some(Decimal::from(10)),
        );

        // 正常订单应该通过
        assert!(checker.check_order(&order, Decimal::from(1000000)).is_ok());

        // 超大订单应该被拒绝
        assert!(checker.check_order(&order, Decimal::from(40000)).is_err());
    }
}
