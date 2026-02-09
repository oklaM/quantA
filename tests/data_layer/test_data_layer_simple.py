#!/usr/bin/env python3
"""
简化版数据层测试脚本

先测试基本功能，再测试复杂功能
"""

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/Users/rowan/Projects/quantA')

from data.market.data_manager import DataManager
from data.market.sources.akshare_provider import AKShareProvider
from data.market.sources.tushare_provider import TushareProvider
from utils.logging import get_logger

# 设置日志
logger = get_logger(__name__)

def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

def test_akshare_direct():
    """测试1: 直接使用AKShare获取数据"""
    print_section("测试1: 直接使用AKShare获取数据")

    try:
        # 创建AKShare提供商
        provider = AKShareProvider()
        provider.connect()

        # 获取平安银行数据
        symbol = "000001.SZ"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

        df = provider.get_daily_bar(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df.empty:
            print(f"未获取到数据: {symbol}")
            provider.disconnect()
            return None

        print(f"获取到 {len(df)} 条数据")
        print("\n数据预览:")
        print(df.head())

        provider.disconnect()
        return df

    except Exception as e:
        print(f"错误: {e}")
        try:
            provider.disconnect()
        except:
            pass
        return None

def test_tushare_direct():
    """测试2: 直接使用Tushare获取数据"""
    print_section("测试2: 直接使用Tushare获取数据")

    try:
        # 检查token
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            print("未设置TUSHARE_TOKEN环境变量")
            return None

        # 创建Tushare提供商
        provider = TushareProvider()
        provider.connect()

        # 获取平安银行数据
        symbol = "000001.SZ"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")

        print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")

        df = provider.get_daily_bar(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df.empty:
            print(f"未获取到数据: {symbol}")
            provider.disconnect()
            return None

        print(f"获取到 {len(df)} 条数据")
        print("\n数据预览:")
        print(df.head())

        provider.disconnect()
        return df

    except Exception as e:
        print(f"错误: {e}")
        try:
            provider.disconnect()
        except:
            pass
        return None

def test_data_manager_fixed():
    """测试3: 使用修复后的DataManager"""
    print_section("测试3: 使用修复后的DataManager")

    try:
        # 创建数据管理器
        data_manager = DataManager(provider="akshare")

        # 测试获取单个股票数据
        symbol = "000001.SZ"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

        print(f"通过数据管理器获取 {symbol} 数据...")
        df = data_manager.get_stock_data(symbol, start_date, end_date)

        if df.empty:
            print("未获取到数据")
            return False

        print(f"获取到 {len(df)} 条数据")
        print("\n数据预览:")
        print(df.head())

        # 检查缓存功能
        print("测试缓存功能...")
        df_cached = data_manager.get_stock_data(symbol, start_date, end_date)

        if len(df_cached) != len(df):
            print("缓存数据不一致")
            return False

        # 检查缓存列表
        cached_symbols = data_manager.get_cached_symbols()
        if symbol not in cached_symbols:
            print("缓存列表不包含该股票")
            return False

        print(f"缓存符号: {cached_symbols}")

        # 测试批量获取
        print("测试批量获取...")
        symbols = ["000001.SZ", "000002.SZ"]
        batch_data = data_manager.get_multiple_stocks(symbols, start_date, end_date)

        if len(batch_data) < 1:
            print("批量获取失败")
            return False

        print(f"批量获取结果: {len(batch_data)} 个股票")

        # 清理缓存
        data_manager.clear_cache()

        return True

    except Exception as e:
        print(f"错误: {e}")
        try:
            if 'data_manager' in locals():
                data_manager.clear_cache()
        except:
            pass
        return False

def test_stock_list():
    """测试4: 获取股票列表"""
    print_section("测试4: 获取股票列表")

    try:
        # 使用AKShare获取股票列表
        provider = AKShareProvider()
        provider.connect()

        df = provider.get_stock_list()

        if df.empty:
            print("未获取到股票列表")
            return None

        print(f"获取到 {len(df)} 只股票")
        print("\n按市场分布:")
        print(df['market'].value_counts())

        print("\n前10只股票:")
        print(df.head(10))

        provider.disconnect()
        return df

    except Exception as e:
        print(f"错误: {e}")
        try:
            provider.disconnect()
        except:
            pass
        return None

def main():
    """主函数"""
    print_section("quantA 数据层基础功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行测试
    results = []

    # 测试1: AKShare
    akshare_result = test_akshare_direct()
    results.append(("AKShare数据获取", akshare_result is not None))

    # 测试2: Tushare
    tushare_result = test_tushare_direct()
    results.append(("Tushare数据获取", tushare_result is not None))

    # 测试3: DataManager
    dm_result = test_data_manager_fixed()
    results.append(("数据管理器", dm_result))

    # 测试4: 股票列表
    stock_list = test_stock_list()
    results.append(("股票列表", stock_list is not None))

    # 总结
    print_section("测试结果总结")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    success_rate = passed_count / total_count if total_count > 0 else 0

    print(f"总测试数: {total_count}")
    print(f"通过数: {passed_count}")
    print(f"失败数: {total_count - passed_count}")
    print(f"成功率: {success_rate:.1%}")

    print("\n详细结果:")
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")

    if success_rate >= 0.75:
        print(f"\n总体评价: 优秀 (成功率 {success_rate:.1%})")
    elif success_rate >= 0.5:
        print(f"\n总体评价: 良好 (成功率 {success_rate:.1%})")
    else:
        print(f"\n总体评价: 需要改进 (成功率 {success_rate:.1%})")

    return success_rate >= 0.5

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)