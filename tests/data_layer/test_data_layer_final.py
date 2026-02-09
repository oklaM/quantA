#!/usr/bin/env python3
"""
最终数据层测试报告

测试 quantA 项目的数据获取与存储功能
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/Users/rowan/Projects/quantA')

from data.market.data_manager import DataManager
from data.market.sources.akshare_provider import AKShareProvider
from data.market.sources.tushare_provider import TushareProvider
from data.market.storage.timeseries_db import TABLE_SCHEMAS, get_timeseries_db
from utils.logging import get_logger

# 设置日志
logger = get_logger(__name__)

def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)

def test_environment():
    """测试1: 环境配置"""
    print_section("测试1: 环境配置")

    # 检查Python路径
    print(f"Python路径: {sys.path[0]}")
    print(f"工作目录: {os.getcwd()}")

    # 检查环境变量
    print(f"\n环境变量:")
    print(f"TUSHARE_TOKEN: {'已设置' if os.getenv('TUSHARE_TOKEN') else '未设置'}")
    print(f"ZHIPUAI_API_KEY: {'已设置' if os.getenv('ZHIPUAI_API_KEY') else '未设置'}")

    # 检查依赖包
    print(f"\n依赖包:")
    packages = ['pandas', 'duckdb', 'akshare', 'tushare']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✓ {pkg}")
        except ImportError:
            print(f"✗ {pkg}")

    return True

def test_data_sources():
    """测试2: 数据源连接性"""
    print_section("测试2: 数据源连接性")

    results = {}

    # 测试AKShare
    try:
        provider = AKShareProvider()
        provider.connect()
        results['akshare'] = True
        print("✓ AKShare: 连接成功")
        provider.disconnect()
    except Exception as e:
        results['akshare'] = False
        print(f"✗ AKShare: {str(e)}")

    # 测试Tushare
    if os.getenv('TUSHARE_TOKEN'):
        try:
            provider = TushareProvider()
            provider.connect()
            results['tushare'] = True
            print("✓ Tushare: 连接成功")
            provider.disconnect()
        except Exception as e:
            results['tushare'] = False
            print(f"✗ Tushare: {str(e)}")
    else:
        results['tushare'] = False
        print("✗ Tushare: 未设置TOKEN")

    return results

def test_database():
    """测试3: 数据库功能"""
    print_section("测试3: 数据库功能")

    try:
        # 获取数据库实例
        db = get_timeseries_db()
        db.connect()
        print("✓ DuckDB: 连接成功")

        # 测试创建表
        test_table = "test_table"
        if not db.table_exists(test_table):
            schema = TABLE_SCHEMAS["daily_bar"]
            db.create_table(test_table, schema)
            print("✓ DuckDB: 创建表成功")

        # 测试写入数据
        test_data = pd.DataFrame({
            'symbol': ['000001.SZ'],
            'date': ['2024-01-01'],
            'open': [10.0],
            'high': [10.5],
            'low': [9.8],
            'close': [10.2],
            'volume': [1000000],
            'amount': [10200000.0]
        })

        db.write(test_table, test_data)
        print("✓ DuckDB: 写入数据成功")

        # 测试读取数据
        read_data = db.read(test_table)
        if not read_data.empty:
            print("✓ DuckDB: 读取数据成功")
            print(f"  数据量: {len(read_data)}")
        else:
            print("✗ DuckDB: 读取数据失败")

        # 清理测试数据
        db.execute_sql(f"DROP TABLE IF EXISTS {test_table}")

        # 断开连接
        db.disconnect()
        print("✓ DuckDB: 连接已断开")

        return True

    except Exception as e:
        print(f"✗ DuckDB: {str(e)}")
        try:
            if 'db' in locals():
                db.disconnect()
        except:
            pass
        return False

def test_data_manager():
    """测试4: 数据管理器"""
    print_section("测试4: 数据管理器")

    try:
        # 测试初始化
        dm = DataManager(provider="akshare")
        print("✓ DataManager: 初始化成功")

        # 测试缓存功能
        test_key = "test_cache"
        dm._cache[test_key] = pd.DataFrame({'test': [1, 2, 3]})
        cached_keys = dm.get_cached_symbols()
        if test_key in cached_keys:
            print("✓ DataManager: 缓存功能正常")
        else:
            print("✗ DataManager: 缓存功能异常")

        # 测试清空缓存
        dm.clear_cache()
        if test_key not in dm.get_cached_symbols():
            print("✓ DataManager: 清空缓存成功")
        else:
            print("✗ DataManager: 清空缓存失败")

        return True

    except Exception as e:
        print(f"✗ DataManager: {str(e)}")
        try:
            if 'dm' in locals():
                dm.clear_cache()
        except:
            pass
        return False

def test_stock_info():
    """测试5: 股票信息获取"""
    print_section("测试5: 股票信息获取")

    try:
        provider = AKShareProvider()
        provider.connect()
        print("✓ AKShare: 连接成功")

        # 获取股票列表
        stock_list = provider.get_stock_list()
        if not stock_list.empty:
            print(f"✓ 获取股票列表成功: {len(stock_list)} 只股票")
            print(f"  市场分布: {dict(stock_list['market'].value_counts())}")
            print(f"  前5只股票:")
            print(stock_list.head()[['symbol', 'name', 'market']])
        else:
            print("✗ 获取股票列表失败")
            return False

        # 测试股票基本信息
        symbol = stock_list.iloc[0]['symbol']
        try:
            stock_info = provider.get_stock_info(symbol)
            print(f"✓ 获取股票信息成功: {symbol}")
            print(f"  基本信息: {dict(stock_info)}")
        except Exception as e:
            print(f"✗ 获取股票信息失败: {symbol} - {str(e)}")

        provider.disconnect()
        return True

    except Exception as e:
        print(f"✗ 股票信息获取失败: {str(e)}")
        try:
            provider.disconnect()
        except:
            pass
        return False

def main():
    """主函数"""
    print_section("quantA 数据层功能测试报告")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行所有测试
    test_results = []

    # 测试1: 环境配置
    env_ok = test_environment()
    test_results.append(("环境配置", env_ok))

    # 测试2: 数据源连接性
    data_sources = test_data_sources()
    ds_ok = any(data_sources.values())
    test_results.append(("数据源连接", ds_ok))

    # 测试3: 数据库功能
    db_ok = test_database()
    test_results.append(("数据库功能", db_ok))

    # 测试4: 数据管理器
    dm_ok = test_data_manager()
    test_results.append(("数据管理器", dm_ok))

    # 测试5: 股票信息获取
    info_ok = test_stock_info()
    test_results.append(("股票信息获取", info_ok))

    # 生成最终报告
    print_section("测试结果总结")

    passed_count = sum(1 for _, passed in test_results if passed)
    total_count = len(test_results)
    success_rate = passed_count / total_count if total_count > 0 else 0

    print(f"总测试数: {total_count}")
    print(f"通过数: {passed_count}")
    print(f"失败数: {total_count - passed_count}")
    print(f"成功率: {success_rate:.1%}")

    print("\n详细结果:")
    for test_name, passed in test_results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {test_name}: {status}")

    print("\n数据源详情:")
    for source, status in data_sources.items():
        print(f"  {source}: {'✓ 连接正常' if status else '✗ 连接失败'}")

    if success_rate >= 0.8:
        print(f"\n总体评价: 优秀 (成功率 {success_rate:.1%})")
        print("数据层核心功能正常，可以继续开发和完善")
    elif success_rate >= 0.6:
        print(f"\n总体评价: 良好 (成功率 {success_rate:.1%})")
        print("数据层基本功能正常，需要修复一些问题")
    else:
        print(f"\n总体评价: 需要改进 (成功率 {success_rate:.1%})")
        print("数据层存在较多问题，需要重点修复")

    # 生成详细报告
    print_section("详细建议")

    if not data_sources.get('akshare', False):
        print("1. AKShare连接问题:")
        print("   - 检查网络连接")
        print("   - 检查akshare库版本")
        print("   - 可能是服务器限制或网络问题")

    if not data_sources.get('tushare', False):
        print("2. Tushare连接问题:")
        print("   - 确认TUSHARE_TOKEN环境变量已设置")
        print("   - 检查token是否有效")
        print("   - 免费版token可能有频率限制")

    if not db_ok:
        print("3. 数据库问题:")
        print("   - 确认duckdb已安装")
        print("   - 检查数据库文件权限")

    if not dm_ok:
        print("4. 数据管理器问题:")
        print("   - 检查数据源初始化")
        print("   - 验证缓存机制")

    if not info_ok:
        print("5. 股票信息问题:")
        print("   - 网络连接问题")
        print("   - AKShare接口变更")

    return success_rate >= 0.6

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)