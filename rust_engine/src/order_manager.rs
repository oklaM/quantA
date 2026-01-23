/*!
订单管理模块

提供订单的创建、修改、取消、查询等功能
*/

use crate::error::{EngineError, EngineResult};
use crate::execution::{Order, OrderStatus, OrderSide, OrderType};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

/// 订单管理器
pub struct OrderManager {
    /// 订单存储
    orders: HashMap<String, Order>,
    /// 活跃订单（待成交或部分成交）
    active_orders: HashMap<String, Order>,
    /// 已完成订单
    completed_orders: Vec<Order>,
    /// 订单计数器
    order_counter: u64,
}

impl OrderManager {
    /// 创建订单管理器
    pub fn new() -> Self {
        OrderManager {
            orders: HashMap::new(),
            active_orders: HashMap::new(),
            completed_orders: Vec::new(),
            order_counter: 0,
        }
    }

    /// 生成订单ID
    fn generate_order_id(&mut self) -> String {
        self.order_counter += 1;
        format!("ORD_{:010}", self.order_counter)
    }

    /// 创建订单
    pub fn create_order(
        &mut self,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: i64,
        price: Option<Decimal>,
    ) -> EngineResult<String> {
        let order_id = self.generate_order_id();

        let order = Order::new(
            order_id.clone(),
            symbol,
            side,
            order_type,
            quantity,
            price,
        );

        // 检查订单参数
        if quantity <= 0 {
            return Err(EngineError::OrderError("订单数量必须大于0".to_string()));
        }

        if let Some(p) = price {
            if p <= Decimal::ZERO {
                return Err(EngineError::OrderError("订单价格必须大于0".to_string()));
            }
        }

        // 限价单必须有价格
        if order_type == OrderType::Limit && price.is_none() {
            return Err(EngineError::OrderError("限价单必须指定价格".to_string()));
        }

        // 存储订单
        self.orders.insert(order_id.clone(), order.clone());
        self.active_orders.insert(order_id.clone(), order);

        Ok(order_id)
    }

    /// 获取订单
    pub fn get_order(&self, order_id: &str) -> Option<&Order> {
        self.orders.get(order_id)
    }

    /// 获取可变订单
    pub fn get_order_mut(&mut self, order_id: &str) -> Option<&mut Order> {
        self.orders.get_mut(order_id)
    }

    /// 更新订单状态
    pub fn update_order_status(
        &mut self,
        order_id: &str,
        status: OrderStatus,
    ) -> EngineResult<()> {
        let order = self.orders.get_mut(order_id)
            .ok_or_else(|| EngineError::OrderError(format!("订单不存在: {}", order_id)))?;

        let old_status = order.status;
        order.status = status;
        order.updated_at = Utc::now();

        // 如果订单完成，从活跃订单移除
        if status != OrderStatus::Pending && status != OrderStatus::PartiallyFilled {
            if let Some(active_order) = self.active_orders.remove(order_id) {
                self.completed_orders.push(active_order);
            }
        } else if old_status == OrderStatus::Pending && status == OrderStatus::PartiallyFilled {
            // 从待成交转为部分成交，更新活跃订单
            if let Some(order) = self.orders.get(order_id) {
                self.active_orders.insert(order_id.to_string(), order.clone());
            }
        }

        Ok(())
    }

    /// 成交订单
    pub fn fill_order(
        &mut self,
        order_id: &str,
        filled_quantity: i64,
        fill_price: Decimal,
    ) -> EngineResult<()> {
        let order = self.orders.get_mut(order_id)
            .ok_or_else(|| EngineError::OrderError(format!("订单不存在: {}", order_id)))?;

        // 检查成交数量
        if filled_quantity <= 0 {
            return Err(EngineError::OrderError("成交数量必须大于0".to_string()));
        }

        if filled_quantity > order.remaining_quantity() {
            return Err(EngineError::OrderError(format!(
                "成交数量超过剩余数量: remaining={}, requested={}",
                order.remaining_quantity(),
                filled_quantity
            )));
        }

        // 更新成交信息
        order.filled_quantity += filled_quantity;

        // 计算平均成交价
        if let Some(avg_price) = order.avg_fill_price {
            let total_value = avg_price * Decimal::from(order.filled_quantity - filled_quantity)
                + fill_price * Decimal::from(filled_quantity);
            order.avg_fill_price = Some(total_value / Decimal::from(order.filled_quantity));
        } else {
            order.avg_fill_price = Some(fill_price);
        }

        // 更新状态
        if order.filled_quantity == order.quantity {
            self.update_order_status(order_id, OrderStatus::Filled)?;
        } else {
            self.update_order_status(order_id, OrderStatus::PartiallyFilled)?;
        }

        Ok(())
    }

    /// 取消订单
    pub fn cancel_order(&mut self, order_id: &str) -> EngineResult<bool> {
        let order = self.orders.get(order_id)
            .ok_or_else(|| EngineError::OrderError(format!("订单不存在: {}", order_id)))?;

        // 检查订单状态
        if order.is_done() {
            return Ok(false);
        }

        self.update_order_status(order_id, OrderStatus::Cancelled)?;
        Ok(true)
    }

    /// 批量取消订单
    pub fn cancel_orders(&mut self, order_ids: &[String]) -> EngineResult<Vec<String>> {
        let mut cancelled = Vec::new();

        for order_id in order_ids {
            if self.cancel_order(order_id)? {
                cancelled.push(order_id.clone());
            }
        }

        Ok(cancelled)
    }

    /// 取消所有活跃订单
    pub fn cancel_all_active(&mut self) -> EngineResult<Vec<String>> {
        let order_ids: Vec<String> = self.active_orders.keys().cloned().collect();
        self.cancel_orders(&order_ids)
    }

    /// 按股票取消订单
    pub fn cancel_by_symbol(&mut self, symbol: &str) -> EngineResult<Vec<String>> {
        let order_ids: Vec<String> = self.active_orders
            .values()
            .filter(|o| o.symbol == symbol)
            .map(|o| o.id.clone())
            .collect();

        self.cancel_orders(&order_ids)
    }

    /// 获取活跃订单
    pub fn get_active_orders(&self) -> Vec<&Order> {
        self.active_orders.values().collect()
    }

    /// 获取指定股票的活跃订单
    pub fn get_active_orders_by_symbol(&self, symbol: &str) -> Vec<&Order> {
        self.active_orders
            .values()
            .filter(|o| o.symbol == symbol)
            .collect()
    }

    /// 获取已完成订单
    pub fn get_completed_orders(&self) -> &[Order] {
        &self.completed_orders
    }

    /// 获取所有订单
    pub fn get_all_orders(&self) -> Vec<&Order> {
        self.orders.values().collect()
    }

    /// 统计信息
    pub fn get_stats(&self) -> OrderStats {
        let active_count = self.active_orders.len();
        let completed_count = self.completed_orders.len();

        let total_orders = self.orders.len();

        let mut filled_count = 0;
        let mut cancelled_count = 0;
        let mut rejected_count = 0;

        for order in self.completed_orders.iter() {
            match order.status {
                OrderStatus::Filled => filled_count += 1,
                OrderStatus::Cancelled => cancelled_count += 1,
                OrderStatus::Rejected => rejected_count += 1,
                _ => {}
            }
        }

        OrderStats {
            total_orders,
            active_orders: active_count,
            filled_orders: filled_count,
            cancelled_orders: cancelled_count,
            rejected_orders: rejected_count,
        }
    }
}

impl Default for OrderManager {
    fn default() -> Self {
        Self::new()
    }
}

/// 订单统计
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderStats {
    pub total_orders: usize,
    pub active_orders: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub rejected_orders: usize,
}

/// 线程安全的订单管理器
pub struct ThreadSafeOrderManager {
    inner: Arc<RwLock<OrderManager>>,
}

impl ThreadSafeOrderManager {
    /// 创建线程安全的订单管理器
    pub fn new() -> Self {
        ThreadSafeOrderManager {
            inner: Arc::new(RwLock::new(OrderManager::new())),
        }
    }

    /// 创建订单
    pub fn create_order(
        &self,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        quantity: i64,
        price: Option<Decimal>,
    ) -> EngineResult<String> {
        let mut manager = self.inner.write()
            .map_err(|e| EngineError::Other(format!("获取写锁失败: {}", e)))?;

        manager.create_order(symbol, side, order_type, quantity, price)
    }

    /// 获取订单
    pub fn get_order(&self, order_id: &str) -> EngineResult<Option<Order>> {
        let manager = self.inner.read()
            .map_err(|e| EngineError::Other(format!("获取读锁失败: {}", e)))?;

        Ok(manager.get_order(order_id).cloned())
    }

    /// 成交订单
    pub fn fill_order(
        &self,
        order_id: &str,
        filled_quantity: i64,
        fill_price: Decimal,
    ) -> EngineResult<()> {
        let mut manager = self.inner.write()
            .map_err(|e| EngineError::Other(format!("获取写锁失败: {}", e)))?;

        manager.fill_order(order_id, filled_quantity, fill_price)
    }

    /// 取消订单
    pub fn cancel_order(&self, order_id: &str) -> EngineResult<bool> {
        let mut manager = self.inner.write()
            .map_err(|e| EngineError::Other(format!("获取写锁失败: {}", e)))?;

        manager.cancel_order(order_id)
    }

    /// 获取活跃订单
    pub fn get_active_orders(&self) -> EngineResult<Vec<Order>> {
        let manager = self.inner.read()
            .map_err(|e| EngineError::Other(format!("获取读锁失败: {}", e)))?;

        Ok(manager.get_active_orders().into_iter().cloned().collect())
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> EngineResult<OrderStats> {
        let manager = self.inner.read()
            .map_err(|e| EngineError::Other(format!("获取读锁失败: {}", e)))?;

        Ok(manager.get_stats())
    }
}

impl Default for ThreadSafeOrderManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_order() {
        let mut manager = OrderManager::new();

        let order_id = manager.create_order(
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1000,
            Some(Decimal::from(10)),
        ).unwrap();

        let order = manager.get_order(&order_id).unwrap();
        assert_eq!(order.symbol, "000001.SZ");
        assert_eq!(order.quantity, 1000);
        assert_eq!(order.status, OrderStatus::Pending);
    }

    #[test]
    fn test_fill_order() {
        let mut manager = OrderManager::new();

        let order_id = manager.create_order(
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1000,
            Some(Decimal::from(10)),
        ).unwrap();

        // 部分成交
        manager.fill_order(&order_id, 500, Decimal::from(10)).unwrap();

        let order = manager.get_order(&order_id).unwrap();
        assert_eq!(order.filled_quantity, 500);
        assert_eq!(order.status, OrderStatus::PartiallyFilled);

        // 全部成交
        manager.fill_order(&order_id, 500, Decimal::from(11)).unwrap();

        let order = manager.get_order(&order_id).unwrap();
        assert_eq!(order.filled_quantity, 1000);
        assert_eq!(order.status, OrderStatus::Filled);
        assert_eq!(order.avg_fill_price, Some(Decimal::from(10) + Decimal::from(1)) / 2);
    }

    #[test]
    fn test_cancel_order() {
        let mut manager = OrderManager::new();

        let order_id = manager.create_order(
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1000,
            Some(Decimal::from(10)),
        ).unwrap();

        assert!(manager.cancel_order(&order_id).unwrap());

        let order = manager.get_order(&order_id).unwrap();
        assert_eq!(order.status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_order_stats() {
        let mut manager = OrderManager::new();

        // 创建多个订单
        let order_id1 = manager.create_order(
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1000,
            Some(Decimal::from(10)),
        ).unwrap();

        let order_id2 = manager.create_order(
            "000002.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1000,
            Some(Decimal::from(10)),
        ).unwrap();

        // 成交一个
        manager.fill_order(&order_id1, 1000, Decimal::from(10)).unwrap();

        // 取消一个
        manager.cancel_order(&order_id2).unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_orders, 2);
        assert_eq!(stats.filled_orders, 1);
        assert_eq!(stats.cancelled_orders, 1);
        assert_eq!(stats.active_orders, 0);
    }
}
