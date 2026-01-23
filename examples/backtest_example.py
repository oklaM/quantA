"""
回测系统示例
演示如何使用quantA回测系统
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest.engine.strategy import MovingAverageCrossStrategy, BuyAndHoldStrategy
from backtest.engine.backtest import run_backtest
from backtest.metrics.report import generate_report
from utils.logging import get_logger

logger = get_logger(__name__)


def generate_sample_data(
    symbol: str = "600519.SH",
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """
    生成模拟数据（用于演示）

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        模拟的OHLCV数据
    """
    # 生成日期范围
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    # 只保留交易日（排除周末）
    dates = [d for d in dates if d.weekday() < 5]

    # 生成随机价格数据（模拟股价走势）
    np.random.seed(42)
    n = len(dates)

    # 初始价格
    initial_price = 100.0

    # 生成收益率（布朗运动）
    returns = np.random.normal(0.0005, 0.02, n)
    prices = initial_price * (1 + returns).cumprod()

    # 生成OHLC
    data = []
    for i, date in enumerate(dates):
        close = prices[i]
        # 简单模拟：基于close生成其他价格
        noise = np.random.normal(0, 0.01, 3)
        high = close * (1 + abs(noise[0]))
        low = close * (1 - abs(noise[1]))
        open_price = close * (1 + noise[2])

        # 确保逻辑正确
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # 成交量
        volume = int(np.random.normal(1000000, 200000))

        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': max(0, volume),
        })

    return pd.DataFrame(data)


def run_ma_cross_strategy_example():
    """运行双均线策略示例"""
    logger.info("=" * 60)
    logger.info("双均线交叉策略回测示例")
    logger.info("=" * 60)

    # 1. 准备数据
    logger.info("步骤1: 准备数据")
    symbol = "600519.SH"
    data = {
        symbol: generate_sample_data(symbol)
    }
    logger.info(f"生成数据: {len(data[symbol])}条记录")

    # 2. 创建策略
    logger.info("\n步骤2: 创建策略")
    strategy = MovingAverageCrossStrategy(
        symbol=symbol,
        fast_period=5,
        slow_period=20,
        quantity=1000,  # 每次交易1000股
    )
    logger.info(f"策略参数: 快线={strategy.fast_period}, 慢线={strategy.slow_period}")

    # 3. 运行回测
    logger.info("\n步骤3: 运行回测")
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,  # 100万初始资金
        commission_rate=0.0003,  # 万分之三佣金
        slippage_rate=0.0001,  # 万分之一滑点
    )

    # 4. 输出结果
    logger.info("\n步骤4: 回测结果")
    _print_results(results, "双均线交叉策略")

    # 5. 生成报告
    logger.info("\n步骤5: 生成报告")
    report_path = project_root / "logs" / "backtest_report_ma_cross.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    generate_report(
        results=results,
        strategy_name="双均线交叉策略",
        output_path=str(report_path),
    )
    logger.info(f"报告已保存到: {report_path}")

    return results


def run_buy_and_hold_example():
    """运行买入持有策略示例"""
    logger.info("\n" + "=" * 60)
    logger.info("买入持有策略回测示例")
    logger.info("=" * 60)

    # 1. 准备数据
    symbol = "600519.SH"
    data = {
        symbol: generate_sample_data(symbol)
    }

    # 2. 创建策略
    strategy = BuyAndHoldStrategy(
        symbol=symbol,
        quantity=10000,  # 买入10000股
    )

    # 3. 运行回测
    results = run_backtest(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
    )

    # 4. 输出结果
    _print_results(results, "买入持有策略")

    return results


def _print_results(results: dict, strategy_name: str):
    """打印回测结果"""
    account = results.get('account', {})
    performance = results.get('performance', {})

    logger.info(f"\n【{strategy_name}回测结果】")
    logger.info(f"初始资金: ¥{account.get('initial_cash', 0):,.2f}")
    logger.info(f"最终资产: ¥{account.get('total_value', 0):,.2f}")
    logger.info(f"总收益率: {account.get('total_return_pct', 0):.2f}%")
    logger.info(f"夏普比率: {performance.get('sharpe_ratio', 0):.2f}")
    logger.info(f"最大回撤: {performance.get('max_drawdown', 0) * 100:.2f}%")
    logger.info(f"总交易次数: {account.get('total_trades', 0)}")

    # 打印持仓
    positions = results.get('positions', [])
    if positions:
        logger.info(f"\n当前持仓: {len(positions)}只股票")
        for pos in positions:
            logger.info(
                f"  {pos['symbol']}: {pos['quantity']}股, "
                f"盈亏 ¥{pos['pnl']:,.2f} ({pos['pnl_pct']:.2f}%)"
            )


def main():
    """主函数"""
    logger.info("quantA回测系统示例")
    logger.info(f"项目路径: {project_root}")

    # 运行示例
    ma_results = run_ma_cross_strategy_example()
    bh_results = run_buy_and_hold_example()

    # 对比
    logger.info("\n" + "=" * 60)
    logger.info("策略对比")
    logger.info("=" * 60)

    ma_return = ma_results['account']['total_return_pct']
    bh_return = bh_results['account']['total_return_pct']

    logger.info(f"双均线策略收益率: {ma_return:.2f}%")
    logger.info(f"买入持有收益率: {bh_return:.2f}%")
    logger.info(f"超额收益: {ma_return - bh_return:.2f}%")


if __name__ == "__main__":
    main()
