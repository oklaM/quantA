#!/usr/bin/env python3
"""
测试不同的股票代码格式
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/Users/rowan/Projects/quantA')

from data.market.sources.akshare_provider import AKShareProvider
from data.market.sources.tushare_provider import TushareProvider


def test_formats():
    """测试不同的股票代码格式"""

    # 测试股票
    test_symbols = [
        "000001.SZ",  # 标准格式
        "000001",     # 只有代码
        "sz000001",   # AKShare格式
        "sh000001",   # 错误格式
        "600000.SH",  # 上海股票
        "600000",     # 只有代码
        "sh600000",   # AKShare格式
    ]

    # 创建提供商
    ak_provider = AKShareProvider()
    ak_provider.connect()

    tushare_token = os.getenv('TUSHARE_TOKEN')
    if tushare_token:
        ts_provider = TushareProvider()
        ts_provider.connect()
    else:
        ts_provider = None

    # 测试日期
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    print("测试AKShare数据源:")
    print("-" * 50)

    for symbol in test_symbols:
        try:
            df = ak_provider.get_daily_bar(symbol, start_date, end_date)
            print(f"✓ {symbol}: {len(df)} 条数据")
            if not df.empty:
                print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
        except Exception as e:
            print(f"✗ {symbol}: {str(e)}")

    if ts_provider:
        print("\n测试Tushare数据源:")
        print("-" * 50)

        for symbol in test_symbols[:4]:  # 只测试前4个
            try:
                df = ts_provider.get_daily_bar(symbol, start_date, end_date)
                print(f"✓ {symbol}: {len(df)} 条数据")
                if not df.empty:
                    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
            except Exception as e:
                print(f"✗ {symbol}: {str(e)}")

    # 关闭连接
    ak_provider.disconnect()
    if ts_provider:
        ts_provider.disconnect()

if __name__ == "__main__":
    test_formats()