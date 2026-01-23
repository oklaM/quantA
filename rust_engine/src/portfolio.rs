/*!
投资组合模块

负责持仓管理、资金计算等
*/

use crate::error::{EngineError, EngineResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::DateTime;
use rust_decimal::Decimal;

/// 持仓信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// 股票代码
    pub symbol: String,
    /// 数量
    pub quantity: i64,
    /// 可用数量（扣除冻结）
    pub available_quantity: i64,
    /// 平均成本
    pub avg_cost: Decimal,
    /// 当前市价
    pub current_price: Decimal,
    /// 市值
    pub market_value: Decimal,
    /// 浮动盈亏
    pub unrealized_pnl: Decimal,
    /// 开仓时间
    pub opened_at: DateTime<chrono::Utc>,
    /// 更新时间
    pub updated_at: DateTime<chrono::Utc>,
}

impl Position {
    /// 创建新持仓
    pub fn new(symbol: String, quantity: i64, price: Decimal) -> Self {
        let now = chrono::Utc::now();
        let market_value = Decimal::from(quantity) * price;

        Position {
            symbol,
            quantity,
            available_quantity: quantity,
            avg_cost: price,
            current_price: price,
            market_value,
            unrealized_pnl: Decimal::ZERO,
            opened_at: now,
            updated_at: now,
        }
    }

    /// 更新价格
    pub fn update_price(&mut self, new_price: Decimal) {
        self.current_price = new_price;
        self.market_value = Decimal::from(self.quantity) * new_price;
        self.unrealized_pnl = (new_price - self.avg_cost) * Decimal::from(self.quantity);
        self.updated_at = chrono::Utc::now();
    }

    /// 增加持仓
    pub fn increase(&mut self, quantity: i64, cost: Decimal) {
        let total_cost = self.avg_cost * Decimal::from(self.quantity) + cost * Decimal::from(quantity);
        self.quantity += quantity;
        self.available_quantity += quantity;
        self.avg_cost = total_cost / Decimal::from(self.quantity);
        self.updated_at = chrono::Utc::now();
    }

    /// 减少持仓
    pub fn decrease(&mut self, quantity: i64) -> EngineResult<()> {
        if quantity > self.available_quantity {
            return Err(EngineError::PositionError(format!(
                "持仓不足: available={}, requested={}",
                self.available_quantity, quantity
            )));
        }

        self.quantity -= quantity;
        self.available_quantity -= quantity;
        self.updated_at = chrono::Utc::now();

        Ok(())
    }

    /// 获取盈亏比例
    pub fn pnl_ratio(&self) -> Decimal {
        if self.avg_cost == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.current_price - self.avg_cost) / self.avg_cost
        }
    }
}

/// 投资组合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    /// 初始资金
    pub initial_cash: Decimal,
    /// 可用现金
    pub cash: Decimal,
    /// 持仓
    pub positions: HashMap<String, Position>,
    /// 冻结现金
    pub frozen_cash: Decimal,
    /// 创建时间
    pub created_at: DateTime<chrono::Utc>,
    /// 更新时间
    pub updated_at: DateTime<chrono::Utc>,
}

impl Portfolio {
    /// 创建新组合
    pub fn new(initial_cash: Decimal) -> Self {
        let now = chrono::Utc::now();
        Portfolio {
            initial_cash,
            cash: initial_cash,
            positions: HashMap::new(),
            frozen_cash: Decimal::ZERO,
            created_at: now,
            updated_at: now,
        }
    }

    /// 获取总资产
    pub fn total_value(&self) -> Decimal {
        let positions_value: Decimal = self.positions.values()
            .map(|p| p.market_value)
            .sum();

        self.cash + positions_value
    }

    /// 获取总盈亏
    pub fn total_pnl(&self) -> Decimal {
        self.total_value() - self.initial_cash
    }

    /// 获取盈亏比例
    pub fn pnl_ratio(&self) -> Decimal {
        if self.initial_cash == Decimal::ZERO {
            Decimal::ZERO
        } else {
            (self.total_value() - self.initial_cash) / self.initial_cash
        }
    }

    /// 获取持仓
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// 获取或创建持仓
    pub fn get_or_create_position(&mut self, symbol: &str) -> &mut Position {
        self.positions.entry(symbol.to_string())
            .or_insert_with(|| Position::new(symbol.to_string(), 0, Decimal::ZERO))
    }

    /// 更新持仓价格
    pub fn update_position_price(&mut self, symbol: &str, price: Decimal) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.update_price(price);
            self.updated_at = chrono::Utc::now();
        }
    }

    /// 买入
    pub fn buy(
        &mut self,
        symbol: &str,
        quantity: i64,
        price: Decimal,
        commission: Decimal,
    ) -> EngineResult<()> {
        let cost = Decimal::from(quantity) * price + commission;

        if cost > self.cash {
            return Err(EngineError::PositionError(format!(
                "资金不足: required={}, available={}",
                cost, self.cash
            )));
        }

        // 扣除资金
        self.cash -= cost;

        // 更新持仓
        let position = self.get_or_create_position(symbol);
        if position.quantity == 0 {
            *position = Position::new(symbol.to_string(), quantity, price);
        } else {
            position.increase(quantity, price);
        }

        self.updated_at = chrono::Utc::now();

        Ok(())
    }

    /// 卖出
    pub fn sell(
        &mut self,
        symbol: &str,
        quantity: i64,
        price: Decimal,
        commission: Decimal,
    ) -> EngineResult<Decimal> {
        let position = self.positions.get_mut(symbol)
            .ok_or_else(|| EngineError::PositionError(format!("持仓不存在: {}", symbol)))?;

        // 检查数量
        if quantity > position.available_quantity {
            return Err(EngineError::PositionError(format!(
                "持仓不足: available={}, requested={}",
                position.available_quantity, quantity
            )));
        }

        // 计算收入
        let proceeds = Decimal::from(quantity) * price - commission;

        // 更新持仓
        position.decrease(quantity)?;

        // 如果持仓为0，删除
        if position.quantity == 0 {
            self.positions.remove(symbol);
        }

        // 增加资金
        self.cash += proceeds;

        self.updated_at = chrono::Utc::now();

        Ok(proceeds)
    }

    /// 获取持仓市值
    pub fn get_position_value(&self, symbol: &str) -> Decimal {
        self.positions.get(symbol)
            .map(|p| p.market_value)
            .unwrap_or(Decimal::ZERO)
    }

    /// 获取持仓比例
    pub fn get_position_ratio(&self, symbol: &str) -> Decimal {
        let total = self.total_value();
        if total == Decimal::ZERO {
            Decimal::ZERO
        } else {
            self.get_position_value(symbol) / total
        }
    }

    /// 获取现金比例
    pub fn cash_ratio(&self) -> Decimal {
        let total = self.total_value();
        if total == Decimal::ZERO {
            Decimal::ZERO
        } else {
            self.cash / total
        }
    }

    /// 获取所有持仓列表
    pub fn get_all_positions(&self) -> Vec<&Position> {
        self.positions.values().collect()
    }

    /// 持仓数量
    pub fn position_count(&self) -> usize {
        self.positions.len()
    }
}

/// 投资组合指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    /// 总资产
    pub total_value: Decimal,
    /// 总盈亏
    pub total_pnl: Decimal,
    /// 盈亏比例
    pub pnl_ratio: Decimal,
    /// 现金比例
    pub cash_ratio: Decimal,
    /// 持仓数量
    pub position_count: usize,
    /// 单日盈亏
    pub daily_pnl: Decimal,
    /// 单日盈亏比例
    pub daily_pnl_ratio: Decimal,
    /// 最大回撤
    pub max_drawdown: Decimal,
    /// 夏普比率
    pub sharpe_ratio: Option<Decimal>,
}

impl Portfolio {
    /// 计算指标
    pub fn calculate_metrics(&self, prev_value: Option<Decimal>) -> PortfolioMetrics {
        let total_value = self.total_value();
        let total_pnl = self.total_pnl();
        let pnl_ratio = self.pnl_ratio();
        let cash_ratio = self.cash_ratio();
        let position_count = self.position_count();

        let (daily_pnl, daily_pnl_ratio) = if let Some(prev) = prev_value {
            let pnl = total_value - prev;
            let ratio = if prev > Decimal::ZERO {
                (total_value - prev) / prev
            } else {
                Decimal::ZERO
            };
            (pnl, ratio)
        } else {
            (Decimal::ZERO, Decimal::ZERO)
        };

        PortfolioMetrics {
            total_value,
            total_pnl,
            pnl_ratio,
            cash_ratio,
            position_count,
            daily_pnl,
            daily_pnl_ratio,
            max_drawdown: Decimal::ZERO, // 需要历史数据计算
            sharpe_ratio: None, // 需要历史数据计算
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_creation() {
        let portfolio = Portfolio::new(Decimal::from(1000000));

        assert_eq!(portfolio.initial_cash, Decimal::from(1000000));
        assert_eq!(portfolio.cash, Decimal::from(1000000));
        assert_eq!(portfolio.total_value(), Decimal::from(1000000));
        assert_eq!(portfolio.position_count(), 0);
    }

    #[test]
    fn test_buy_sell() {
        let mut portfolio = Portfolio::new(Decimal::from(1000000));

        // 买入
        portfolio.buy("000001.SZ", 1000, Decimal::from(10), Decimal::from(5)).unwrap();

        assert_eq!(portfolio.cash, Decimal::from(999995)); // 1000000 - 10000 - 5
        assert_eq!(portfolio.position_count(), 1);
        assert_eq!(portfolio.total_value(), Decimal::from(1000000)); // 资产价值不变

        // 更新价格
        portfolio.update_position_price("000001.SZ", Decimal::from(11));
        assert_eq!(portfolio.total_value(), Decimal::from(1010000)); // 999995 + 1000 * 11 = 1010995

        // 卖出
        portfolio.sell("000001.SZ", 1000, Decimal::from(11), Decimal::from(5)).unwrap();

        assert_eq!(portfolio.position_count(), 0); // 持仓清空
        assert!(portfolio.cash > Decimal::from(1000000)); // 应该有盈利
    }

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new("000001.SZ".to_string(), 1000, Decimal::from(10));

        assert_eq!(position.unrealized_pnl, Decimal::ZERO);

        position.update_price(Decimal::from(11));
        assert_eq!(position.unrealized_pnl, Decimal::from(1000)); // (11 - 10) * 1000

        position.update_price(Decimal::from(9));
        assert_eq!(position.unrealized_pnl, Decimal::from(-1000)); // (9 - 10) * 1000
    }
}
