"""
高级策略示例：多因子组合策略
展示如何结合多个技术指标构建复杂的交易策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import Strategy
from backtest.engine.event_engine import BarEvent, OrderEvent
from backtest.engine.indicators import TechnicalIndicators
from utils import logger


class MultiFactorStrategy(Strategy):
    """
    多因子组合策略

    结合多个技术指标进行决策：
    1. 趋势因子：MA > 0表示上升趋势
    2. 动量因子：RSI在超买超卖区间
    3. 波动率因子：ATR衡量市场波动
    4. 成交量因子：成交量放大确认突破

    买入条件：
    - 短期MA > 长期MA (上升趋势)
    - RSI在30-70之间（不在极端区域）
    - 成交量大于均值

    卖出条件：
    - RSI > 70 (超买)
    - 或亏损超过止损线
    """

    def __init__(
        self,
        symbol: str,
        short_ma: int = 5,
        long_ma: int = 20,
        rsi_period: int = 14,
        atr_period: int = 14,
        volume_period: int = 20,
        quantity: int = 1000,
        stop_loss: float = 0.05,  # 5%止损
        take_profit: float = 0.15,  # 15%止盈
    ):
        super().__init__()
        self.symbol = symbol
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.volume_period = volume_period
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit

        # 价格历史
        self.price_history = []
        self.volume_history = []

        # 技术指标
        self.indicators = TechnicalIndicators()

        # 持仓信息
        self.entry_price = None
        self.entry_date = None

    def on_bar(self, event: BarEvent):
        """处理每个K线事件"""
        if event.symbol != self.symbol:
            return

        # 更新历史数据
        self.price_history.append(event.close)
        self.volume_history.append(event.volume)

        # 需要足够的历史数据
        min_period = max(self.long_ma, self.rsi_period, self.atr_period, self.volume_period)
        if len(self.price_history) < min_period + 1:
            return

        # 计算技术指标
        prices = pd.Series(self.price_history)
        volumes = pd.Series(self.volume_history)

        # 移动平均线
        ma_short = self.indicators.sma(prices, period=self.short_ma).iloc[-1]
        ma_long = self.indicators.sma(prices, period=self.long_ma).iloc[-1]

        # RSI
        rsi = self.indicators.rsi(prices, period=self.rsi_period).iloc[-1]

        # ATR（波动率）
        high_series = pd.Series([bar.high for bar in self.bars if bar.symbol == self.symbol][-len(prices):])
        low_series = pd.Series([bar.low for bar in self.bars if bar.symbol == self.symbol][-len(prices):])
        atr = self.indicators.atr(high_series, low_series, prices, period=self.atr_period).iloc[-1]

        # 成交量均值
        volume_ma = volumes.rolling(window=self.volume_period).mean().iloc[-1]

        # 获取当前持仓
        position = self.get_position(self.symbol)
        current_quantity = position["quantity"] if position else 0

        # === 买入逻辑 ===
        if current_quantity == 0:
            # 多个条件满足时买入
            if (
                ma_short > ma_long  # 上升趋势
                and 30 < rsi < 70  # RSI不在极端区域
                and event.volume > volume_ma  # 成交量放大
            ):
                self.buy(self.symbol, self.quantity, order_type="market")
                self.entry_price = event.close
                self.entry_date = event.datetime
                logger.info(
                    f"买入信号: 价格={event.close:.2f}, "
                    f"MA短={ma_short:.2f}, MA长={ma_long:.2f}, "
                    f"RSI={rsi:.2f}, 成交量={event.volume}"
                )

        # === 卖出逻辑 ===
        elif current_quantity > 0 and self.entry_price:
            # 计算收益率
            return_pct = (event.close - self.entry_price) / self.entry_price

            # 止损
            if return_pct < -self.stop_loss:
                self.sell(self.symbol, current_quantity, order_type="market")
                logger.info(
                    f"止损卖出: 价格={event.close:.2f}, "
                    f"收益率={return_pct*100:.2f}%"
                )
                self.entry_price = None
                self.entry_date = None

            # 止盈
            elif return_pct > self.take_profit:
                self.sell(self.symbol, current_quantity, order_type="market")
                logger.info(
                    f"止盈卖出: 价格={event.close:.2f}, "
                    f"收益率={return_pct*100:.2f}%"
                )
                self.entry_price = None
                self.entry_date = None

            # 超买卖出
            elif rsi > 70:
                self.sell(self.symbol, current_quantity, order_type="market")
                logger.info(
                    f"超买卖出: 价格={event.close:.2f}, RSI={rsi:.2f}"
                )
                self.entry_price = None
                self.entry_date = None

            # 趋势反转
            elif ma_short < ma_long:
                self.sell(self.symbol, current_quantity, order_type="market")
                logger.info(
                    f"趋势反转卖出: 价格={event.close:.2f}, "
                    f"MA短={ma_short:.2f}, MA长={ma_long:.2f}"
                )
                self.entry_price = None
                self.entry_date = None


def generate_sample_data(symbol: str = "600519.SH", days: int = 252):
    """生成示例数据"""
    np.random.seed(42)

    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq="D")

    # 生成带趋势的随机价格
    trend = np.linspace(100, 120, days)
    noise = np.random.randn(days) * 2
    prices = trend + noise

    # 确保价格为正
    prices = np.maximum(prices, 1)

    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, days)),
        "low": prices * (1 - np.random.uniform(0, 0.02, days)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, days)
    })

    # 确保high >= close >= low
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99

    # 过滤周末（模拟交易日）
    df = df[df["date"].dt.dayofweek < 5].reset_index(drop=True)

    return df


def example_1_multi_factor_strategy():
    """示例1：多因子策略完整回测"""
    logger.info("=" * 60)
    logger.info("示例1：多因子组合策略回测")
    logger.info("=" * 60)

    # 生成数据
    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=252)

    logger.info(f"生成数据: {len(data)}个交易日")
    logger.info(f"价格范围: {data['close'].min():.2f} - {data['close'].max():.2f}")

    # 创建策略
    strategy = MultiFactorStrategy(
        symbol=symbol,
        short_ma=5,
        long_ma=20,
        rsi_period=14,
        quantity=1000,
        stop_loss=0.05,
        take_profit=0.15
    )

    # 创建回测引擎
    engine = BacktestEngine(
        data={symbol: data},
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
    )

    # 运行回测
    logger.info("\n开始回测...")
    results = engine.run()

    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("回测结果")
    logger.info("=" * 60)
    logger.info(f"初始资金: ¥{1_000_000:,.2f}")
    logger.info(f"最终资金: ¥{results['final_value']:,.2f}")
    logger.info(f"总收益: ¥{results['total_return']:,.2f}")
    logger.info(f"收益率: {results['total_return_pct']:.2f}%")
    logger.info(f"总交易次数: {results['total_trades']}")
    logger.info(f"胜率: {results['win_rate']:.2f}%")
    logger.info(f"最大回撤: {results['max_drawdown']:.2f}%")
    logger.info(f"夏普比率: {results.get('sharpe_ratio', 'N/A')}")

    return results


def example_2_parameter_optimization():
    """示例2：参数优化"""
    logger.info("\n" + "=" * 60)
    logger.info("示例2：策略参数优化")
    logger.info("=" * 60)

    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=126)  # 使用半年数据优化

    # 定义参数网格
    param_grid = {
        'short_ma': [3, 5, 10],
        'long_ma': [15, 20, 30],
        'rsi_period': [10, 14, 20],
        'stop_loss': [0.03, 0.05, 0.10],
    }

    best_return = -float('inf')
    best_params = None
    total_combinations = (
        len(param_grid['short_ma']) *
        len(param_grid['long_ma']) *
        len(param_grid['rsi_period']) *
        len(param_grid['stop_loss'])
    )

    logger.info(f"测试 {total_combinations} 种参数组合...")

    count = 0
    for short_ma in param_grid['short_ma']:
        for long_ma in param_grid['long_ma']:
            if short_ma >= long_ma:
                continue  # 短期MA应该小于长期MA

            for rsi_period in param_grid['rsi_period']:
                for stop_loss in param_grid['stop_loss']:
                    count += 1

                    # 创建策略
                    strategy = MultiFactorStrategy(
                        symbol=symbol,
                        short_ma=short_ma,
                        long_ma=long_ma,
                        rsi_period=rsi_period,
                        quantity=1000,
                        stop_loss=stop_loss,
                    )

                    # 回测
                    engine = BacktestEngine(
                        data={symbol: data},
                        strategy=strategy,
                        initial_cash=1000000.0,
                        commission_rate=0.0003,
                    )
                    results = engine.run()

                    # 记录最佳参数
                    if results['total_return_pct'] > best_return:
                        best_return = results['total_return_pct']
                        best_params = {
                            'short_ma': short_ma,
                            'long_ma': long_ma,
                            'rsi_period': rsi_period,
                            'stop_loss': stop_loss,
                            'return': results['total_return_pct']
                        }

                    logger.info(
                        f"[{count}/{total_combinations}] "
                        f"MA({short_ma},{long_ma}) RSI({rsi_period}) "
                        f"止损={stop_loss*100:.0f}% -> 收益率={results['total_return_pct']:.2f}%"
                    )

    logger.info("\n" + "=" * 60)
    logger.info("最佳参数组合")
    logger.info("=" * 60)
    if best_params:
        logger.info(f"短期MA: {best_params['short_ma']}")
        logger.info(f"长期MA: {best_params['long_ma']}")
        logger.info(f"RSI周期: {best_params['rsi_period']}")
        logger.info(f"止损比例: {best_params['stop_loss']*100:.1f}%")
        logger.info(f"最佳收益率: {best_params['return']:.2f}%")

    return best_params


def example_3_walk_forward_analysis():
    """示例3：走向前分析（Walk Forward）"""
    logger.info("\n" + "=" * 60)
    logger.info("示例3：走向前分析")
    logger.info("=" * 60)

    symbol = "600519.SH"
    data = generate_sample_data(symbol, days=252)

    # 分割数据：训练期100天，测试期50天
    train_period = 100
    test_period = 50
    total_days = len(data)

    results_list = []

    for start in range(0, total_days - train_period - test_period, 50):
        # 训练集
        train_data = data.iloc[start:start + train_period]
        # 测试集
        test_data = data.iloc[start + train_period:start + train_period + test_period]

        logger.info(f"\n训练期: {train_data['date'].iloc[0].date()} 到 {train_data['date'].iloc[-1].date()}")
        logger.info(f"测试期: {test_data['date'].iloc[0].date()} 到 {test_data['date'].iloc[-1].date()}")

        # 使用训练数据优化参数
        best_params = None
        best_return = -float('inf')

        for short_ma in [5, 10]:
            for long_ma in [20, 30]:
                if short_ma >= long_ma:
                    continue

                strategy = MultiFactorStrategy(
                    symbol=symbol,
                    short_ma=short_ma,
                    long_ma=long_ma,
                    quantity=1000,
                )

                engine = BacktestEngine(
                    data={symbol: train_data},
                    strategy=strategy,
                    initial_cash=1000000.0,
                    commission_rate=0.0003,
                )
                results = engine.run()

                if results['total_return_pct'] > best_return:
                    best_return = results['total_return_pct']
                    best_params = {'short_ma': short_ma, 'long_ma': long_ma}

        # 在测试集上验证
        logger.info(f"最佳参数: MA({best_params['short_ma']}, {best_params['long_ma']})")

        strategy = MultiFactorStrategy(
            symbol=symbol,
            short_ma=best_params['short_ma'],
            long_ma=best_params['long_ma'],
            quantity=1000,
        )

        engine = BacktestEngine(
            data={symbol: test_data},
            strategy=strategy,
            initial_cash=1000000.0,
            commission_rate=0.0003,
        )
        test_results = engine.run()

        logger.info(f"测试期收益率: {test_results['total_return_pct']:.2f}%")

        results_list.append({
            'train_start': train_data['date'].iloc[0].date(),
            'test_start': test_data['date'].iloc[0].date(),
            'params': best_params,
            'train_return': best_return,
            'test_return': test_results['total_return_pct']
        })

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("走向前分析总结")
    logger.info("=" * 60)

    avg_test_return = np.mean([r['test_return'] for r in results_list])
    logger.info(f"平均测试期收益率: {avg_test_return:.2f}%")
    logger.info(f"测试期数量: {len(results_list)}")

    return results_list


if __name__ == "__main__":
    # 运行示例
    example_1_multi_factor_strategy()
    example_2_parameter_optimization()
    example_3_walk_forward_analysis()

    logger.info("\n" + "=" * 60)
    logger.info("所有示例运行完成！")
    logger.info("=" * 60)
