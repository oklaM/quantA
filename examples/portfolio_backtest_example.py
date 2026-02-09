"""
组合回测示例
展示如何使用组合回测功能
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtest.engine.strategies import BuyAndHoldStrategy, MovingAverageCrossStrategy
from backtest.portfolio import Portfolio, PortfolioBacktestEngine, StrategyAllocation
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_sample_data(symbols: list, days: int = 500, start_price: float = 100.0):
    """
    生成多个股票的示例数据

    Args:
        symbols: 股票代码列表
        days: 天数
        start_price: 起始价格

    Returns:
        数据字典 {symbol: DataFrame}
    """
    np.random.seed(42)

    data_dict = {}

    for i, symbol in enumerate(symbols):
        # 每个股票有不同的收益率特征
        mean_return = 0.0005 + i * 0.0001
        volatility = 0.02 - i * 0.001

        returns = np.random.normal(mean_return, volatility, days)
        prices = start_price * (1 + returns).cumprod()

        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=days,
            freq='D'
        )

        data = []
        for date, close in zip(dates, prices):
            high = close * (1 + abs(np.random.normal(0, 0.015)))
            low = close * (1 - abs(np.random.normal(0, 0.015)))
            open_price = close * (1 + np.random.normal(0, 0.008))

            data.append({
                'datetime': date,
                'symbol': symbol,
                'open': open_price,
                'high': max(high, open_price, close),
                'low': min(low, open_price, close),
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })

        data_dict[symbol] = pd.DataFrame(data)

    return data_dict


def example_single_strategy_multi_asset():
    """示例1：单策略多资产组合回测"""
    print("\n" + "="*70)
    print("示例1：单策略多资产组合回测")
    print("="*70)

    # 生成数据
    print("\n生成数据...")
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']
    data_dict = generate_sample_data(symbols, days=500)

    # 创建策略配置
    print("创建策略配置...")
    strategy = BuyAndHoldStrategy(symbol='000001.SZ')  # symbol会被覆盖

    # 为每个股票分配相同权重
    strategy_allocations = []
    weight_per_symbol = 1.0 / len(symbols)

    for symbol in symbols:
        # 每个股票一个独立的策略实例
        strategy_allocations.append(
            StrategyAllocation(
                strategy=BuyAndHoldStrategy(symbol=symbol),
                symbols=[symbol],
                weight=weight_per_symbol,
            )
        )

    # 创建组合回测引擎
    print("创建组合回测引擎...")
    engine = PortfolioBacktestEngine(
        data_dict=data_dict,
        strategies=strategy_allocations,
        initial_cash=10000000.0,  # 1000万
        commission_rate=0.0003,
    )

    # 运行回测
    print("运行组合回测...")
    results = engine.run()

    # 打印结果
    print("\n" + "="*70)
    print("回测结果")
    print("="*70)
    print(f"总收益: {results['total_return']:.2%}")
    print(f"年化收益: {results['annual_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
    print(f"波动率: {results['volatility']:.2%}")

    # 各策略价值
    print("\n各策略最终价值:")
    for strategy_id, value in results['strategy_values'].items():
        strategy_name = strategy_allocations[strategy_id].strategy.__class__.__name__
        symbol = strategy_allocations[strategy_id].symbols[0]
        weight = strategy_allocations[strategy_id].weight
        print(f"  策略{strategy_id} ({strategy_name} - {symbol}): "
              f"¥{value:,.2f} (权重={weight:.1%})")

    return results


def example_multi_strategy_portfolio():
    """示例2：多策略组合回测"""
    print("\n" + "="*70)
    print("示例2：多策略组合回测")
    print("="*70)

    # 生成数据
    print("\n生成数据...")
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH', '000300.SH']
    data_dict = generate_sample_data(symbols, days=500)

    # 创建策略配置
    print("创建多策略配置...")

    # 策略1：买入持有（大盘股）
    strategy1_allocations = [
        StrategyAllocation(
            strategy=BuyAndHoldStrategy(symbol=symbol),
            symbols=[symbol],
            weight=0.2,  # 20%权重
        )
        for symbol in symbols[:2]
    ]

    # 策略2：均线交叉（中小盘股）
    strategy2_allocations = [
        StrategyAllocation(
            strategy=MovingAverageCrossStrategy(
                symbol=symbol,
                short_window=10,
                long_window=30,
            ),
            symbols=[symbol],
            weight=0.15,  # 15%权重
        )
        for symbol in symbols[2:]
    ]

    all_strategies = strategy1_allocations + strategy2_allocations

    # 创建组合回测引擎
    print("创建组合回测引擎...")
    engine = PortfolioBacktestEngine(
        data_dict=data_dict,
        strategies=all_strategies,
        initial_cash=10000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    print("运行多策略组合回测...")
    results = engine.run()

    # 打印结果
    print("\n" + "="*70)
    print("多策略组合回测结果")
    print("="*70)
    print(f"总收益: {results['total_return']:.2%}")
    print(f"年化收益: {results['annual_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")

    # 各策略表现
    print("\n各策略表现:")
    for strategy_id, value in results['strategy_values'].items():
        strategy_alloc = all_strategies[strategy_id]
        strategy_name = strategy_alloc.strategy.__class__.__name__
        symbol = strategy_alloc.symbols[0]
        initial_value = 10000000.0 * strategy_alloc.weight
        strategy_return = (value - initial_value) / initial_value

        print(f"  策略{strategy_id} ({strategy_name} - {symbol}):")
        print(f"    初始价值: ¥{initial_value:,.2f}")
        print(f"    最终价值: ¥{value:,.2f}")
        print(f"    收益率: {strategy_return:.2%}")
        print(f"    权重: {strategy_alloc.weight:.1%}")

    return results


def example_portfolio_rebalancing():
    """示例3：组合再平衡"""
    print("\n" + "="*70)
    print("示例3：组合再平衡")
    print("="*70)

    print("""
组合再平衡策略：

1. 定期再平衡
   每月/每季度调整组合权重，回到目标配置

   示例代码：
   ```python
   class RebalancingStrategy(Strategy):
       def __init__(self, symbols, target_weights, rebalance_freq='M'):
           self.symbols = symbols
           self.target_weights = target_weights
           self.rebalance_freq = rebalance_freq
           self.last_rebalance = None

       def on_bar(self, event):
           current_date = event.timestamp

           # 检查是否需要再平衡
           if self._should_rebalance(current_date):
               self._rebalance_portfolio(event)

       def _should_rebalance(self, current_date):
           if self.last_rebalance is None:
               return True

           if self.rebalance_freq == 'M':
               return current_date.month != self.last_rebalance.month
           elif self.rebalance_freq == 'Q':
               return current_date.quarter != self.last_rebalance.quarter
           else:
               return False

       def _rebalance_portfolio(self, event):
           # 计算当前权重
           current_weights = self._calculate_current_weights()

           # 计算需要调整的仓位
           for symbol in self.symbols:
               target_weight = self.target_weights[symbol]
               current_weight = current_weights.get(symbol, 0)

               # 如果偏离超过阈值，进行调整
               if abs(current_weight - target_weight) > 0.05:
                   self._create_rebalance_order(symbol, target_weight, current_weight)

           self.last_rebalance = event.timestamp
   ```

2. 波动率目标再平衡
   根据资产波动率调整权重

   ```python
   class VolatilityTargetStrategy(Strategy):
       def __init__(self, symbols, target_volatility=0.15):
           self.symbols = symbols
           self.target_volatility = target_volatility
           self.lookback = 60

       def _calculate_weights(self):
           # 计算各资产波动率
           volatilities = {}
           for symbol in self.symbols:
               returns = self._get_historical_returns(symbol, self.lookback)
               volatilities[symbol] = returns.std() * np.sqrt(252)

           # 反向波动率加权
           inv_vol = {s: 1/v for s, v in volatilities.items()}
           total_inv_vol = sum(inv_vol.values())

           weights = {s: v/total_inv_vol for s, v in inv_vol.items()}
           return weights
   ```

3. 动态权重调整
   根据市场状况动态调整权重

   ```python
   class DynamicWeightStrategy(Strategy):
       def on_bar(self, event):
           # 计算市场指标
           market_trend = self._calculate_market_trend()
           market_volatility = self._calculate_market_volatility()

           # 根据市场状况调整权重
           if market_trend > 0 and market_volatility < 0.2:
               # 牛市且低波动：增加风险资产权重
               self.adjust_weights(aggressive=True)
           elif market_trend < 0:
               # 熊市：减少风险资产权重
               self.adjust_weights(defensive=True)
   ```
""")


def example_portfolio_performance_attribution():
    """示例4：组合绩效归因"""
    print("\n" + "="*70)
    print("示例4：组合绩效归因")
    print("="*70)

    print("""
绩效归因分析：

1. 资产配置贡献
   计算各资产对组合收益的贡献

   ```python
   def calculate_allocation_contribution(portfolio, benchmark):
       contribution = {}

       for asset, weight in portfolio.weights.items():
           asset_return = portfolio.returns[asset]
           benchmark_return = benchmark.returns[asset]

           # 配置贡献 = weight * (asset_return - benchmark_return)
           contribution[asset] = weight * (asset_return - benchmark_return)

       return contribution
   ```

2. 策略贡献
   分析各策略对组合的贡献

   ```python
   def analyze_strategy_contribution(results):
       contributions = []

       for strategy_id, strategy_value in results['strategy_values'].items():
           initial_value = results['initial_value'] * strategy_weights[strategy_id]
           strategy_return = (strategy_value - initial_value) / initial_value

           contributions.append({
               'strategy_id': strategy_id,
               'return': strategy_return,
               'contribution': strategy_return * strategy_weights[strategy_id],
           })

       return pd.DataFrame(contributions)
   ```

3. 风险归因
   分析各资产对组合风险的贡献

   ```python
   def calculate_risk_contribution(portfolio):
       weights = np.array(list(portfolio.weights.values()))
       cov_matrix = portfolio.calculate_covariance_matrix()

       # 计算组合波动率
       portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

       # 计算各资产边际风险贡献
       marginal_contrib = cov_matrix @ weights / portfolio_vol

       # 计算各资产风险贡献
       risk_contrib = weights * marginal_contrib

       return dict(zip(portfolio.weights.keys(), risk_contrib))
   ```
""")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 组合回测示例")
    print("="*70)

    try:
        # 示例1：单策略多资产
        example_single_strategy_multi_asset()

        # 示例2：多策略组合
        # example_multi_strategy_portfolio()

        # 示例3-4：概念说明
        example_portfolio_rebalancing()
        example_portfolio_performance_attribution()

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        print(f"\n出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
