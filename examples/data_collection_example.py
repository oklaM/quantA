"""
数据采集和真实数据源使用示例

本示例展示如何使用quantA的数据采集系统获取真实市场数据
"""

import pandas as pd
from datetime import datetime, timedelta
from data.market.sources.tushare_provider import TushareProvider
from data.market.sources.akshare_provider import AKShareProvider
from data.market.collector import DataCollector, create_collector
from utils.logging import get_logger

logger = get_logger(__name__)


def example_akshare_daily_data():
    """示例1：使用AKShare获取日线数据（免费，无需token）"""
    print("\n" + "="*50)
    print("示例1：使用AKShare获取日线数据")
    print("="*50)

    # 创建AKShare数据源
    provider = AKShareProvider()
    provider.connect()

    # 获取平安银行(000001.SZ)的日线数据
    symbol = "000001.SZ"
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

    print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

    df = provider.get_daily_bar(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权
    )

    print(f"\n获取到 {len(df)} 条数据")
    print("\n数据预览:")
    print(df.head())
    print("\n数据统计:")
    print(df.describe())

    provider.disconnect()
    return df


def example_tushare_daily_data():
    """示例2：使用Tushare获取日线数据（需要token）"""
    print("\n" + "="*50)
    print("示例2：使用Tushare获取日线数据")
    print("="*50)

    try:
        # 从环境变量或配置文件读取token
        import os
        token = os.getenv('TUSHARE_TOKEN')

        if not token:
            print("未设置TUSHARE_TOKEN环境变量，跳过此示例")
            print("请在.env文件中设置：TUSHARE_TOKEN=your_token_here")
            return None

        # 创建Tushare数据源
        provider = TushareProvider(token=token)
        provider.connect()

        # 获取贵州茅台(600519.SH)的日线数据
        symbol = "600519.SH"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y%m%d")

        print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

        df = provider.get_daily_bar(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        print(f"\n获取到 {len(df)} 条数据")
        print("\n数据预览:")
        print(df.head())

        provider.disconnect()
        return df

    except Exception as e:
        print(f"Tushare示例失败: {e}")
        return None


def example_get_stock_list():
    """示例3：获取股票列表"""
    print("\n" + "="*50)
    print("示例3：获取股票列表")
    print("="*50)

    provider = AKShareProvider()
    provider.connect()

    # 获取股票列表
    df = provider.get_stock_list()

    print(f"\n共有 {len(df)} 只股票")
    print("\n按市场分布:")
    print(df['market'].value_counts())

    print("\n前10只股票:")
    print(df.head(10))

    provider.disconnect()
    return df


def example_realtime_quote():
    """示例4：获取实时行情"""
    print("\n" + "="*50)
    print("示例4：获取实时行情")
    print("="*50)

    provider = AKShareProvider()
    provider.connect()

    # 获取部分热门股票的实时行情
    symbols = ["000001", "000002", "600000", "600519", "600036"]

    print(f"获取 {len(symbols)} 只股票的实时行情...")
    df = provider.get_realtime_quote(symbols)

    print("\n实时行情:")
    print(df[['symbol', 'price', 'change_pct', 'volume', 'amount']])

    provider.disconnect()
    return df


def example_data_collector():
    """示例5：使用数据采集器批量采集数据"""
    print("\n" + "="*50)
    print("示例5：使用数据采集器批量采集数据")
    print("="*50)

    # 创建数据采集器
    collector = create_collector(provider="akshare")

    # 采集股票列表
    print("\n1. 采集股票列表...")
    count = collector.collect_stock_list()
    print(f"已采集 {count} 只股票")

    # 获取所有股票代码
    symbols = collector.get_all_symbols()
    print(f"数据库中共有 {len(symbols)} 只股票")

    # 采集部分股票的日线数据（最近30天）
    print("\n2. 采集最近30天的日线数据...")
    symbols_to_collect = symbols[:10]  # 只采集前10只股票作为示例

    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    count = collector.collect_daily_bar(
        symbols=symbols_to_collect,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    print(f"已采集 {count} 条日线数据")

    # 从数据库读取数据
    print("\n3. 从数据库读取数据...")
    symbol = symbols_to_collect[0]
    df = collector.get_daily_bar(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    print(f"\n{symbol} 的数据:")
    print(df.head())

    collector.close()
    return df


def example_update_daily_data():
    """示例6：增量更新日线数据"""
    print("\n" + "="*50)
    print("示例6：增量更新日线数据")
    print("="*50)

    # 创建数据采集器
    collector = create_collector(provider="akshare")

    # 采集股票列表（如果还没有）
    symbols = collector.get_all_symbols()
    if not symbols:
        print("先采集股票列表...")
        collector.collect_stock_list()
        symbols = collector.get_all_symbols()

    # 更新最近5天的数据
    symbols_to_update = symbols[:5]
    print(f"更新 {len(symbols_to_update)} 只股票的最近5天数据...")

    collector.update_daily_data(symbols=symbols_to_update, days=5)

    print("数据更新完成！")

    collector.close()


def example_backtest_with_real_data():
    """示例7：使用真实数据进行回测"""
    print("\n" + "="*50)
    print("示例7：使用真实数据进行回测")
    print("="*50)

    from backtest.engine.backtest import BacktestEngine
    from backtest.engine.strategy import BuyAndHoldStrategy

    # 使用AKShare获取真实数据
    provider = AKShareProvider()
    provider.connect()

    symbol = "000001.SZ"
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")  # 最近半年

    print(f"获取 {symbol} 的数据...")
    df = provider.get_daily_bar(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )

    provider.disconnect()

    # 转换数据格式以适配回测引擎
    df = df.rename(columns={'date': 'datetime'})

    # 准备回测数据
    data = {symbol: df}

    # 创建策略
    strategy = BuyAndHoldStrategy(symbol=symbol, quantity=1000)

    # 运行回测
    print(f"\n开始回测 {symbol}...")
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
    )

    results = engine.run()

    # 显示结果
    print("\n" + "="*50)
    print("回测结果")
    print("="*50)
    account = results['account']
    performance = results['performance']

    print(f"\n账户信息:")
    print(f"  初始资金: ¥{account['initial_cash']:,.2f}")
    print(f"  最终资产: ¥{account['total_value']:,.2f}")
    print(f"  总收益率: {account['total_return_pct']:.2f}%")
    print(f"  交易次数: {account['total_trades']}")

    print(f"\n绩效指标:")
    print(f"  夏普比率: {performance['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {performance['max_drawdown']:.2f}%")
    print(f"  波动率: {performance['volatility']:.2f}%")

    return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print("quantA 真实数据源使用示例")
    print("="*70)

    # 运行各个示例
    try:
        # 示例1：AKShare获取日线数据（最简单，免费）
        df1 = example_akshare_daily_data()

        # 示例2：Tushare获取日线数据（需要token）
        # df2 = example_tushare_daily_data()

        # 示例3：获取股票列表
        df3 = example_get_stock_list()

        # 示例4：获取实时行情
        df4 = example_realtime_quote()

        # 示例5：使用数据采集器
        # df5 = example_data_collector()  # 需要数据库，可选运行

        # 示例6：增量更新
        # example_update_daily_data()  # 需要数据库，可选运行

        # 示例7：使用真实数据回测
        # results = example_backtest_with_real_data()  # 可选运行

        print("\n" + "="*70)
        print("所有示例运行完成！")
        print("="*70)

    except Exception as e:
        logger.error(f"示例运行出错: {e}", exc_info=True)
        print(f"\n出错: {e}")
        print("请检查：")
        print("1. 是否安装了所需的依赖包（akshare, tushare等）")
        print("2. 网络连接是否正常")
        print("3. Tushare token是否正确设置（如果使用Tushare）")


if __name__ == "__main__":
    main()
