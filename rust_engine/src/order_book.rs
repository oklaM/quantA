/*!
订单簿模块

提供市场深度管理和价格匹配功能
*/

use crate::error::{EngineError, EngineResult};
use crate::execution::{Order, OrderSide, OrderType};
use crate::order_manager::OrderManager;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use chrono::Utc;

/// 价格档
#[derive(Debug, Clone)]
pub struct PriceLevel {
    /// 价格
    pub price: Decimal,
    /// 总数量
    pub total_quantity: i64,
    /// 订单数量
    pub order_count: usize,
    /// 订单列表
    pub orders: Vec<String>,
}

impl PriceLevel {
    /// 创建价格档
    pub fn new(price: Decimal) -> Self {
        PriceLevel {
            price,
            total_quantity: 0,
            order_count: 0,
            orders: Vec::new(),
        }
    }

    /// 添加订单
    pub fn add_order(&mut self, order_id: String, quantity: i64) {
        self.total_quantity += quantity;
        self.order_count += 1;
        self.orders.push(order_id);
    }

    /// 移除订单
    pub fn remove_order(&mut self, order_id: &str, quantity: i64) -> EngineResult<()> {
        if let Some(pos) = self.orders.iter().position(|id| id == order_id) {
            self.orders.remove(pos);
            self.total_quantity -= quantity;
            self.order_count -= 1;
            Ok(())
        } else {
            Err(EngineError::OrderError(format!("订单不存在: {}", order_id)))
        }
    }
}

/// 订单簿
pub struct OrderBook {
    /// 股票代码
    symbol: String,
    /// 买盘（价格从高到低）
    bids: BTreeMap<Decimal, PriceLevel>, // 使用反向排序
    /// 卖盘（价格从低到高）
    asks: BTreeMap<Decimal, PriceLevel>,
    /// 订单管理器
    order_manager: Arc<RwLock<OrderManager>>,
    /// 最新成交价
    last_price: Option<Decimal>,
    /// 最新成交量
    last_quantity: Option<i64>,
}

impl OrderBook {
    /// 创建订单簿
    pub fn new(symbol: String) -> Self {
        OrderBook {
            symbol,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            order_manager: Arc::new(RwLock::new(OrderManager::new())),
            last_price: None,
            last_quantity: None,
        }
    }

    /// 获取股票代码
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// 添加订单到订单簿
    pub fn add_order(&mut self, order: Order) -> EngineResult<()> {
        let order_id = order.id.clone();
        let price = order.price.unwrap_or(Decimal::ZERO);

        // 检查订单类型
        if order.order_type != OrderType::Limit {
            return Err(EngineError::OrderError("订单簿只支持限价单".to_string()));
        }

        let price_level = match order.side {
            OrderSide::Buy => {
                // 买盘，使用负价格实现反向排序
                let key = -price;
                self.bids.entry(key)
                    .or_insert_with(|| PriceLevel::new(price))
            }
            OrderSide::Sell => {
                self.asks.entry(price)
                    .or_insert_with(|| PriceLevel::new(price))
            }
        };

        price_level.add_order(order_id.clone(), order.quantity);

        Ok(())
    }

    /// 移除订单
    pub fn remove_order(&mut self, order_id: &str, price: Decimal, side: OrderSide, quantity: i64) -> EngineResult<()> {
        let price_level = match side {
            OrderSide::Buy => {
                let key = -price;
                self.bids.get_mut(&key)
            }
            OrderSide::Sell => {
                self.asks.get_mut(&price)
            }
        };

        if let Some(level) = price_level {
            level.remove_order(order_id, quantity)?;

            // 如果该价格档没有订单了，删除该档
            if level.order_count == 0 {
                match side {
                    OrderSide::Buy => {
                        self.bids.remove(&-price);
                    }
                    OrderSide::Sell => {
                        self.asks.remove(&price);
                    }
                }
            }
        }

        Ok(())
    }

    /// 获取最优买价
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.keys().next().map(|k| -k)
    }

    /// 获取最优卖价
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.keys().next().cloned()
    }

    /// 获取买卖价差
    pub fn spread(&self) -> Option<Decimal> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some(ask - bid)
        } else {
            None
        }
    }

    /// 获取中间价
    pub fn mid_price(&self) -> Option<Decimal> {
        if let (Some(bid), Some(ask)) = (self.best_bid(), self.best_ask()) {
            Some((bid + ask) / 2)
        } else if let Some(bid) = self.best_bid() {
            Some(bid)
        } else if let Some(ask) = self.best_ask() {
            Some(ask)
        } else {
            None
        }
    }

    /// 获取买盘深度（指定档位）
    pub fn get_bids(&self, depth: usize) -> Vec<&PriceLevel> {
        self.bids.values()
            .take(depth)
            .collect()
    }

    /// 获取卖盘深度（指定档位）
    pub fn get_asks(&self, depth: usize) -> Vec<&PriceLevel> {
        self.asks.values()
            .take(depth)
            .collect()
    }

    /// 获取完整的买卖盘深度
    pub fn get_depth(&self, depth: usize) -> OrderBookDepth {
        OrderBookDepth {
            symbol: self.symbol.clone(),
            bids: self.get_bids(depth).into_iter().cloned().collect(),
            asks: self.get_asks(depth).into_iter().cloned().collect(),
            last_price: self.last_price,
            last_quantity: self.last_quantity,
        }
    }

    /// 更新最新成交价
    pub fn update_last_trade(&mut self, price: Decimal, quantity: i64) {
        self.last_price = Some(price);
        self.last_quantity = Some(quantity);
    }

    /// 匹配订单（简化版）
    pub fn match_order(&mut self, order: &Order) -> EngineResult<Vec<Trade>> {
        let mut trades = Vec::new();

        match order.side {
            OrderSide::Buy => {
                // 买入订单，与卖盘匹配
                self.match_buy_order(order, &mut trades)?;
            }
            OrderSide::Sell => {
                // 卖出订单，与买盘匹配
                self.match_sell_order(order, &mut trades)?;
            }
        }

        Ok(trades)
    }

    /// 匹配买入订单
    fn match_buy_order(&mut self, order: &Order, trades: &mut Vec<Trade>) -> EngineResult<()> {
        let price = order.price.ok_or_else(|| {
            EngineError::OrderError("市价单需要特殊处理".to_string())
        })?;

        let mut remaining_quantity = order.quantity;

        // 遍历卖盘（价格从低到高）
        for (_, ask_level) in self.asks.iter() {
            if remaining_quantity <= 0 {
                break;
            }

            // 检查价格是否匹配
            if ask_level.price > price {
                break;
            }

            // 成交
            let trade_quantity = remaining_quantity.min(ask_level.total_quantity);
            trades.push(Trade {
                symbol: self.symbol.clone(),
                price: ask_level.price,
                quantity: trade_quantity,
                timestamp: Utc::now(),
            });

            remaining_quantity -= trade_quantity;
        }

        Ok(())
    }

    /// 匹配卖出订单
    fn match_sell_order(&mut self, order: &Order, trades: &mut Vec<Trade>) -> EngineResult<()> {
        let price = order.price.ok_or_else(|| {
            EngineError::OrderError("市价单需要特殊处理".to_string())
        })?;

        let mut remaining_quantity = order.quantity;

        // 遍历买盘（价格从高到低）
        for (_, bid_level) in self.bids.iter() {
            if remaining_quantity <= 0 {
                break;
            }

            // 检查价格是否匹配
            let bid_price = -bid_level.price; // 转换回正数
            if bid_price < price {
                break;
            }

            // 成交
            let trade_quantity = remaining_quantity.min(bid_level.total_quantity);
            trades.push(Trade {
                symbol: self.symbol.clone(),
                price: bid_price,
                quantity: trade_quantity,
                timestamp: Utc::now(),
            });

            remaining_quantity -= trade_quantity;
        }

        Ok(())
    }
}

/// 订单簿深度
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderBookDepth {
    pub symbol: String,
    pub bids: Vec<PriceLevel>,
    pub asks: Vec<PriceLevel>,
    pub last_price: Option<Decimal>,
    pub last_quantity: Option<i64>,
}

/// 成交记录
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub price: Decimal,
    pub quantity: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// 线程安全的订单簿
pub struct ThreadSafeOrderBook {
    inner: Arc<RwLock<OrderBook>>,
}

impl ThreadSafeOrderBook {
    /// 创建线程安全的订单簿
    pub fn new(symbol: String) -> Self {
        ThreadSafeOrderBook {
            inner: Arc::new(RwLock::new(OrderBook::new(symbol))),
        }
    }

    /// 获取深度
    pub fn get_depth(&self, depth: usize) -> EngineResult<OrderBookDepth> {
        let book = self.inner.read()
            .map_err(|e| EngineError::Other(format!("获取读锁失败: {}", e)))?;

        Ok(book.get_depth(depth))
    }

    /// 获取最优买卖价
    pub fn get_best_bid_ask(&self) -> EngineResult<(Option<Decimal>, Option<Decimal>)> {
        let book = self.inner.read()
            .map_err(|e| EngineError::Other(format!("获取读锁失败: {}", e)))?;

        Ok((book.best_bid(), book.best_ask()))
    }

    /// 更新最新成交
    pub fn update_last_trade(&self, price: Decimal, quantity: i64) -> EngineResult<()> {
        let mut book = self.inner.write()
            .map_err(|e| EngineError::Other(format!("获取写锁失败: {}", e)))?;

        book.update_last_trade(price, quantity);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book() {
        let mut order_book = OrderBook::new("000001.SZ".to_string());

        // 添加卖单
        let ask_order = Order::new(
            "ask1".to_string(),
            "000001.SZ".to_string(),
            OrderSide::Sell,
            OrderType::Limit,
            1000,
            Some(Decimal::from(10)),
        );
        order_book.add_order(ask_order).unwrap();

        // 添加买单
        let bid_order = Order::new(
            "bid1".to_string(),
            "000001.SZ".to_string(),
            OrderSide::Buy,
            OrderType::Limit,
            1000,
            Some(Decimal::from(9)),
        );
        order_book.add_order(bid_order).unwrap();

        // 检查最优价
        assert_eq!(order_book.best_bid(), Some(Decimal::from(9)));
        assert_eq!(order_book.best_ask(), Some(Decimal::from(10)));

        // 检查价差
        assert_eq!(order_book.spread(), Some(Decimal::from(1)));

        // 检查中间价
        assert_eq!(order_book.mid_price(), Some(Decimal::from(9) + Decimal::from(1)) / 2);
    }

    #[test]
    fn test_depth() {
        let mut order_book = OrderBook::new("000001.SZ".to_string());

        // 添加多个卖单
        for i in 1..=5 {
            let order = Order::new(
                format!("ask{}", i),
                "000001.SZ".to_string(),
                OrderSide::Sell,
                OrderType::Limit,
                1000 * i,
                Some(Decimal::from(10 + i)),
            );
            order_book.add_order(order).unwrap();
        }

        // 获取深度
        let depth = order_book.get_depth(3);
        assert_eq!(depth.asks.len(), 3);
        assert_eq!(depth.asks[0].price, Decimal::from(11));
        assert_eq!(depth.asks[2].price, Decimal::from(13));
    }
}
