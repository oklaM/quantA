#!/usr/bin/env python3
"""
数据层测试脚本

测试 quantA 项目的数据获取与存储功能，包括：
1. Tushare 数据源获取
2. DuckDB 数据存储
3. 数据管理器的读取和查询功能
4. 数据质量和完整性验证
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# 添加项目根目录到 Python 路径
sys.path.insert(0, '/Users/rowan/Projects/quantA')

from data.market.collector import DataCollector, create_collector
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

def print_test_result(test_name, success, details=""):
    """打印测试结果"""
    status = "✓ 通过" if success else "✗ 失败"
    print(f"{test_name}: {status}")
    if details:
        print(f"  详情: {details}")

def test_tushare_data_source():
    """测试1: Tushare 数据源获取"""
    print_section("测试1: Tushare 数据源获取")

    start_time = time.time()

    try:
        # 检查token
        token = os.getenv('TUSHARE_TOKEN')
        if not token:
            print_test_result("Tushare Token检查", False, "未设置TUSHARE_TOKEN环境变量")
            return False

        # 创建Tushare提供商
        provider = TushareProvider()
        provider.connect()

        # 测试获取平安银行数据
        symbol = "000001.SZ"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

        print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
        df = provider.get_daily_bar(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        # 验证数据
        if df.empty:
            print_test_result(f"获取 {symbol} 数据", False, "返回空DataFrame")
            return False

        # 检查必需的列
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print_test_result(f"数据列检查", False, f"缺少列: {missing_columns}")
            return False

        # 检查数据范围
        if len(df) < 1:
            print_test_result(f"数据量检查", False, f"数据量不足: {len(df)}")
            return False

        # 检查数据完整性
        if df['symbol'].iloc[0] != symbol:
            print_test_result(f"数据一致性", False, f"股票代码不匹配")
            return False

        # 检查日期范围
        actual_dates = pd.to_datetime(df['date']).min().strftime("%Y%m%d"), pd.to_datetime(df['date']).max().strftime("%Y%m%d")
        print(f"实际日期范围: {actual_dates[0]} 到 {actual_dates[1]}")

        end_time = time.time()
        duration = end_time - start_time

        print(f"获取到 {len(df)} 条数据")
        print(f"耗时: {duration:.2f}秒")
        print("\n数据预览:")
        print(df.head())

        print_test_result(f"Tushare数据获取", True, f"成功获取 {len(df)} 条数据，耗时 {duration:.2f}秒")

        # 断开连接
        provider.disconnect()

        return df

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print_test_result(f"Tushare数据获取", False, f"错误: {str(e)}，耗时 {duration:.2f}秒")
        return None

def test_akshare_data_source():
    """测试2: AKShare 数据源获取"""
    print_section("测试2: AKShare 数据源获取")

    start_time = time.time()

    try:
        # 创建AKShare提供商
        provider = AKShareProvider()
        provider.connect()

        # 测试获取平安银行数据
        symbol = "000001.SZ"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

        print(f"获取 {symbol} 从 {start_date} 到 {end_date} 的数据...")
        df = provider.get_daily_bar(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        # 验证数据
        if df.empty:
            print_test_result(f"获取 {symbol} 数据", False, "返回空DataFrame")
            return False

        # 检查必需的列
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print_test_result(f"数据列检查", False, f"缺少列: {missing_columns}")
            return False

        # 检查数据范围
        if len(df) < 1:
            print_test_result(f"数据量检查", False, f"数据量不足: {len(df)}")
            return False

        end_time = time.time()
        duration = end_time - start_time

        print(f"获取到 {len(df)} 条数据")
        print(f"耗时: {duration:.2f}秒")
        print("\n数据预览:")
        print(df.head())

        print_test_result(f"AKShare数据获取", True, f"成功获取 {len(df)} 条数据，耗时 {duration:.2f}秒")

        # 断开连接
        provider.disconnect()

        return df

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print_test_result(f"AKShare数据获取", False, f"错误: {str(e)}，耗时 {duration:.2f}秒")
        return None

def test_duckdb_storage():
    """测试3: DuckDB 数据存储"""
    print_section("测试3: DuckDB 数据存储")

    start_time = time.time()

    try:
        # 获取数据库实例
        db = get_timeseries_db()
        db.connect()

        # 检查表是否存在
        table_name = "test_daily_bar"
        if not db.table_exists(table_name):
            # 创建测试表
            schema = TABLE_SCHEMAS["daily_bar"]
            db.create_table(table_name, schema)
            print(f"创建测试表: {table_name}")

        # 准备测试数据
        test_data = pd.DataFrame({
            'symbol': ['000001.SZ', '000002.SZ'],
            'date': ['2024-01-08', '2024-01-08'],
            'open': [10.0, 20.0],
            'high': [10.5, 21.0],
            'low': [9.8, 19.5],
            'close': [10.2, 20.5],
            'volume': [1000000, 2000000],
            'amount': [10200000.0, 41000000.0]
        })

        # 写入数据
        print("写入测试数据...")
        db.write(table_name, test_data)

        # 读取数据验证
        print("读取数据验证...")
        read_data = db.read(table_name)

        if read_data.empty:
            print_test_result(f"数据读取", False, "读取数据为空")
            return False

        if len(read_data) != len(test_data):
            print_test_result(f"数据一致性", False, f"写入 {len(test_data)} 条，读取 {len(read_data)} 条")
            return False

        # 检查数据内容
        if not read_data['symbol'].tolist() == test_data['symbol'].tolist():
            print_test_result(f"数据内容", False, "数据内容不匹配")
            return False

        end_time = time.time()
        duration = end_time - start_time

        print(f"成功写入 {len(test_data)} 条数据")
        print(f"成功读取 {len(read_data)} 条数据")
        print(f"耗时: {duration:.2f}秒")
        print("\n写入的数据:")
        print(test_data)
        print("\n读取的数据:")
        print(read_data)

        print_test_result(f"DuckDB数据存储", True, f"成功存储和读取 {len(test_data)} 条数据，耗时 {duration:.2f}秒")

        # 清理测试数据
        db.execute_sql(f"DELETE FROM {table_name}")

        # 断开连接
        db.disconnect()

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print_test_result(f"DuckDB数据存储", False, f"错误: {str(e)}，耗时 {duration:.2f}秒")

        # 尝试断开连接
        try:
            if 'db' in locals():
                db.disconnect()
        except:
            pass

        return False

def test_data_manager():
    """测试4: 数据管理器"""
    print_section("测试4: 数据管理器")

    start_time = time.time()

    try:
        # 创建数据管理器
        data_manager = DataManager(provider="tushare")

        # 测试获取单个股票数据
        symbol = "000001.SZ"
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")

        print(f"通过数据管理器获取 {symbol} 数据...")
        df = data_manager.get_stock_data(symbol, start_date, end_date)

        if df.empty:
            print_test_result(f"数据管理器获取", False, "返回空DataFrame")
            return False

        # 检查缓存功能
        print("测试缓存功能...")
        df_cached = data_manager.get_stock_data(symbol, start_date, end_date)

        if len(df_cached) != len(df):
            print_test_result(f"缓存功能", False, "缓存数据不一致")
            return False

        # 检查缓存列表
        cached_symbols = data_manager.get_cached_symbols()
        if symbol not in cached_symbols:
            print_test_result(f"缓存列表", False, "缓存列表不包含该股票")
            return False

        # 测试批量获取
        print("测试批量获取...")
        symbols = ["000001.SZ", "000002.SZ"]
        batch_data = data_manager.get_multiple_stocks(symbols, start_date, end_date)

        if len(batch_data) < 1:
            print_test_result(f"批量获取", False, "批量获取失败")
            return False

        end_time = time.time()
        duration = end_time - start_time

        print(f"成功获取单个股票数据: {len(df)} 条")
        print(f"成功获取批量数据: {len(batch_data)} 个股票")
        print(f"缓存符号: {cached_symbols}")
        print(f"耗时: {duration:.2f}秒")
        print("\n数据预览:")
        print(df.head())

        print_test_result(f"数据管理器", True, f"成功获取和管理数据，耗时 {duration:.2f}秒")

        # 清理缓存
        data_manager.clear_cache()

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print_test_result(f"数据管理器", False, f"错误: {str(e)}，耗时 {duration:.2f}秒")

        # 尝试清理缓存
        try:
            if 'data_manager' in locals():
                data_manager.clear_cache()
        except:
            pass

        return False

def test_data_collector():
    """测试5: 数据采集器"""
    print_section("测试5: 数据采集器")

    start_time = time.time()

    try:
        # 创建数据采集器（使用AKShare避免token限制）
        collector = create_collector(provider="akshare")

        # 测试获取股票列表
        print("获取股票列表...")
        stock_count = collector.collect_stock_list()

        if stock_count < 1:
            print_test_result(f"获取股票列表", False, "未获取到股票列表")
            return False

        # 获取所有股票代码
        all_symbols = collector.get_all_symbols()

        if not all_symbols:
            print_test_result(f"获取股票代码", False, "未获取到股票代码")
            return False

        print(f"获取到 {stock_count} 只股票")
        print(f"前5只股票: {all_symbols[:5]}")

        # 测试采集少量股票的日线数据
        symbols_to_collect = all_symbols[:3]  # 只取前3只股票
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")

        print(f"\n采集 {len(symbols_to_collect)} 只股票的日线数据...")
        record_count = collector.collect_daily_bar(
            symbols=symbols_to_collect,
            start_date=start_date,
            end_date=end_date
        )

        if record_count < 1:
            print_test_result(f"采集日线数据", False, "未采集到数据")
            return False

        # 测试从数据库读取数据
        print(f"\n从数据库读取数据...")
        symbol = symbols_to_collect[0]
        df = collector.get_daily_bar(symbol, start_date, end_date)

        if df.empty:
            print_test_result(f"读取数据库数据", False, "读取数据为空")
            return False

        end_time = time.time()
        duration = end_time - start_time

        print(f"股票列表采集完成: {stock_count} 只")
        print(f"日线数据采集完成: {record_count} 条")
        print(f"数据读取完成: {len(df)} 条")
        print(f"耗时: {duration:.2f}秒")
        print("\n采集的数据预览:")
        print(df.head())

        print_test_result(f"数据采集器", True, f"成功采集 {record_count} 条数据，耗时 {duration:.2f}秒")

        # 关闭连接
        collector.close()

        return True

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print_test_result(f"数据采集器", False, f"错误: {str(e)}，耗时 {duration:.2f}秒")

        # 尝试关闭连接
        try:
            if 'collector' in locals():
                collector.close()
        except:
            pass

        return False

def test_data_quality():
    """测试6: 数据质量验证"""
    print_section("测试6: 数据质量验证")

    try:
        # 创建数据管理器
        data_manager = DataManager(provider="tushare")

        # 测试数据
        symbols = ["000001.SZ", "000002.SZ"]
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

        print("获取数据质量验证样本...")
        all_data = {}
        for symbol in symbols:
            df = data_manager.get_stock_data(symbol, start_date, end_date)
            if not df.empty:
                all_data[symbol] = df

        if not all_data:
            print_test_result(f"数据质量验证", False, "未获取到验证数据")
            return False

        quality_results = []

        for symbol, df in all_data.items():
            # 检查1: 数据完整性
            missing_values = df.isnull().sum()
            has_missing = missing_values.any()

            # 检查2: 价格合理性
            price_cols = ['open', 'high', 'low', 'close']
            negative_prices = (df[price_cols] < 0).any().any()

            # 检查3: 价格逻辑
            price_logic_error = False
            for _, row in df.iterrows():
                if row['high'] < row['low']:
                    price_logic_error = True
                    break
                if not (row['low'] <= row['open'] <= row['high']):
                    price_logic_error = True
                    break
                if not (row['low'] <= row['close'] <= row['high']):
                    price_logic_error = True
                    break

            # 检查4: 交易量合理性
            negative_volume = (df['volume'] < 0).any()

            # 检查5: 交易金额合理性
            negative_amount = (df['amount'] < 0).any()

            # 检查6: 连续性
            date_diff = pd.to_datetime(df['date']).diff().dropna()
            has_gap = (date_diff > timedelta(days=1)).any()

            quality_check = {
                'symbol': symbol,
                'data_count': len(df),
                'missing_values': not has_missing,
                'reasonable_prices': not negative_prices,
                'price_logic': not price_logic_error,
                'reasonable_volume': not negative_volume,
                'reasonable_amount': not negative_amount,
                'continuous_dates': not has_gap
            }

            quality_results.append(quality_check)

            print(f"\n{symbol} 质量检查结果:")
            print(f"  数据条数: {quality_check['data_count']}")
            print(f"  缺失值: {'✓' if quality_check['missing_values'] else '✗'}")
            print(f"  价格合理性: {'✓' if quality_check['reasonable_prices'] else '✗'}")
            print(f"  价格逻辑: {'✓' if quality_check['price_logic'] else '✗'}")
            print(f"  交易量合理性: {'✓' if quality_check['reasonable_volume'] else '✗'}")
            print(f"  交易金额合理性: {'✓' if quality_check['reasonable_amount'] else '✗'}")
            print(f"  日期连续性: {'✓' if quality_check['continuous_dates'] else '✗'}")

        # 综合质量评估
        passed_checks = sum(1 for r in quality_results for k, v in r.items() if k != 'symbol' and k != 'data_count' and v)
        total_checks = sum(1 for r in quality_results for k in r.keys() if k not in ['symbol', 'data_count'])

        if total_checks > 0:
            quality_score = passed_checks / total_checks
            quality_grade = 'A' if quality_score >= 0.9 else 'B' if quality_score >= 0.8 else 'C' if quality_score >= 0.6 else 'D'
        else:
            quality_grade = 'N/A'

        print(f"\n数据质量总体评分: {quality_grade} ({quality_score:.1%})")

        # 清理缓存
        data_manager.clear_cache()

        if quality_score >= 0.8:
            print_test_result(f"数据质量验证", True, f"质量评分: {quality_grade} ({quality_score:.1%})")
            return True
        else:
            print_test_result(f"数据质量验证", False, f"质量评分: {quality_grade} ({quality_score:.1%})")
            return False

    except Exception as e:
        print_test_result(f"数据质量验证", False, f"错误: {str(e)}")
        return False

def main():
    """主函数"""
    print_section("quantA 数据层功能测试")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path[0]}")

    # 检查环境变量
    print(f"\n环境变量检查:")
    print(f"TUSHARE_TOKEN: {'已设置' if os.getenv('TUSHARE_TOKEN') else '未设置'}")
    print(f"ZHIPUAI_API_KEY: {'已设置' if os.getenv('ZHIPUAI_API_KEY') else '未设置'}")

    # 运行测试
    test_results = []

    # 测试1: Tushare数据源
    tushare_result = test_tushare_data_source()
    test_results.append(("Tushare数据源", tushare_result is not None))

    # 测试2: AKShare数据源
    akshare_result = test_akshare_data_source()
    test_results.append(("AKShare数据源", akshare_result is not None))

    # 测试3: DuckDB存储
    duckdb_result = test_duckdb_storage()
    test_results.append(("DuckDB存储", duckdb_result))

    # 测试4: 数据管理器
    dm_result = test_data_manager()
    test_results.append(("数据管理器", dm_result))

    # 测试5: 数据采集器
    collector_result = test_data_collector()
    test_results.append(("数据采集器", collector_result))

    # 测试6: 数据质量
    quality_result = test_data_quality()
    test_results.append(("数据质量验证", quality_result))

    # 总结
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

    if success_rate >= 0.8:
        print(f"\n总体评价: 优秀 (成功率 {success_rate:.1%})")
    elif success_rate >= 0.6:
        print(f"\n总体评价: 良好 (成功率 {success_rate:.1%})")
    else:
        print(f"\n总体评价: 需要改进 (成功率 {success_rate:.1%})")

    return success_rate >= 0.6

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n测试执行失败: {e}")
        sys.exit(1)