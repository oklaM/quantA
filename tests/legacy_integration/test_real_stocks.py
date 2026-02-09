#!/usr/bin/env python3
"""
测试真实的股票数据获取
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/Users/rowan/Projects/quantA')

from data.market.sources.akshare_provider import AKShareProvider


def test_with_real_stocks():
    """测试使用真实的股票代码"""

    # 创建提供商
    provider = AKShareProvider()
    provider.connect()

    # 从股票列表获取前几只股票
    print("获取股票列表...")
    stock_list = provider.get_stock_list()

    if stock_list.empty:
        print("无法获取股票列表")
        return

    # 取前5只股票进行测试
    test_stocks = stock_list.head(5)['symbol'].tolist()

    print(f"\n测试股票: {test_stocks}")

    # 测试日期
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

    for symbol in test_stocks:
        print(f"\n测试 {symbol}:")
        try:
            df = provider.get_daily_bar(symbol, start_date, end_date)
            if not df.empty:
                print(f"  ✓ 成功获取 {len(df)} 条数据")
                print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
                print(f"  价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
            else:
                print(f"  ✗ 数据为空")
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")

    # 关闭连接
    provider.disconnect()

if __name__ == "__main__":
    test_with_real_stocks()