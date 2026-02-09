"""
数据管道集成测试
测试数据获取、处理和存储的完整流程
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.market.data_manager import DataManager
from data.market.sources.akshare_provider import AKShareProvider
from data.market.sources.base_provider import BaseDataProvider
from data.market.sources.tushare_provider import TushareProvider
from utils.logging import get_logger

logger = get_logger(__name__)


@pytest.fixture
def sample_raw_data():
    """生成模拟原始数据"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "date": dates,
            "open": np.random.uniform(10, 20, len(dates)),
            "high": np.random.uniform(15, 25, len(dates)),
            "low": np.random.uniform(8, 15, len(dates)),
            "close": np.random.uniform(10, 20, len(dates)),
            "volume": np.random.randint(1000000, 10000000, len(dates)),
            "amount": np.random.uniform(10000000, 100000000, len(dates)),
        }
    )

    # 添加一些缺失值以测试数据清洗
    data.loc[10:15, "volume"] = np.nan
    data.loc[50:52, "close"] = np.nan

    return data


@pytest.fixture
def multi_stock_data():
    """生成多股票数据"""
    dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
    symbols = ["600000.SH", "000001.SZ", "600036.SH"]

    data_dict = {}
    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)
        data_dict[symbol] = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(10, 20, len(dates)),
                "high": np.random.uniform(15, 25, len(dates)),
                "low": np.random.uniform(8, 15, len(dates)),
                "close": np.random.uniform(10, 20, len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
                "amount": np.random.uniform(10000000, 100000000, len(dates)),
            }
        )

    return data_dict


@pytest.mark.integration
class TestDataAcquisitionIntegration:
    """测试数据获取流程"""

    def test_data_manager_initialization(self):
        """测试数据管理器初始化"""
        # 测试默认provider (AKShare)
        manager = DataManager()
        assert manager.provider is not None
        assert isinstance(manager.provider, AKShareProvider)

        # 测试Tushare provider
        manager_tushare = DataManager(provider="tushare")
        assert isinstance(manager_tushare.provider, TushareProvider)

    def test_data_provider_fallback(self):
        """测试数据源切换"""
        # 测试不同provider的创建
        manager_akshare = DataManager(provider="akshare")
        assert isinstance(manager_akshare.provider, AKShareProvider)

        manager_tushare = DataManager(provider="tushare")
        assert isinstance(manager_tushare.provider, TushareProvider)

        # 验证provider可以正常访问
        assert hasattr(manager_akshare.provider, 'get_daily_bar')
        assert hasattr(manager_tushare.provider, 'get_daily_bar')

    def test_data_cache_mechanism(self):
        """测试数据缓存机制"""
        manager = DataManager()

        # 生成测试数据并缓存
        cache_key = "test_600000.SH_None_None"
        test_data = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
        manager._cache[cache_key] = test_data

        # 验证缓存工作
        assert cache_key in manager._cache
        assert len(manager.get_cached_symbols()) > 0

        # 清空缓存
        manager.clear_cache()
        assert len(manager._cache) == 0


@pytest.mark.integration
@pytest.mark.requires_data
class TestDataProcessingIntegration:
    """测试数据处理流程"""

    def test_data_cleaning_workflow(self, sample_raw_data):
        """测试数据清洗完整流程"""
        from utils.data_utils import preprocess_data

        # 1. 原始数据有缺失值
        assert sample_raw_data.isnull().sum().sum() > 0

        # 2. 清洗数据
        cleaned_data = preprocess_data(
            sample_raw_data,
            fill_method="ffill",
            drop_na=True,
        )

        # 3. 验证清洗结果
        assert cleaned_data.isnull().sum().sum() == 0
        assert len(cleaned_data) <= len(sample_raw_data)

    def test_data_normalization(self, sample_raw_data):
        """测试数据标准化"""
        from utils.data_utils import normalize_data

        # 先清洗数据
        from utils.data_utils import preprocess_data

        cleaned_data = preprocess_data(sample_raw_data, fill_method="ffill")

        # 标准化
        normalized_data = normalize_data(cleaned_data, method="zscore")

        # 验证标准化结果
        assert normalized_data["close"].mean() < 0.1  # 均值接近0
        assert abs(normalized_data["close"].std() - 1.0) < 0.1  # 标准差接近1

    def test_feature_engineering_pipeline(self, sample_raw_data):
        """测试特征工程完整流程"""
        from backtest.engine.indicators import TechnicalIndicators
        from utils.data_utils import preprocess_data

        # 1. 清洗数据
        cleaned_data = preprocess_data(sample_raw_data, fill_method="ffill")

        # 2. 添加技术指标
        indicators = TechnicalIndicators()
        featured_data = cleaned_data.copy()

        # 添加一些常用指标
        if "close" in featured_data.columns:
            featured_data["sma_5"] = indicators.sma(featured_data["close"], 5)
            featured_data["sma_20"] = indicators.sma(featured_data["close"], 20)
            featured_data["rsi_14"] = indicators.rsi(featured_data["close"], 14)

        # 3. 验证特征添加
        original_cols = set(cleaned_data.columns)
        new_cols = set(featured_data.columns)

        assert len(new_cols) > len(original_cols)
        assert "sma_5" in new_cols or "sma_20" in new_cols

    def test_multi_stock_feature_consistency(self, multi_stock_data):
        """测试多股票特征一致性"""
        from backtest.engine.indicators import TechnicalIndicators

        indicators = TechnicalIndicators()

        # 对每只股票添加特征
        featured_data = {}
        for symbol, data in multi_stock_data.items():
            df = data.copy()
            if "close" in df.columns:
                df["sma_5"] = indicators.sma(df["close"], 5)
                df["sma_20"] = indicators.sma(df["close"], 20)
                df["rsi_14"] = indicators.rsi(df["close"], 14)
            featured_data[symbol] = df

        # 验证所有股票都有相同的特征列
        first_cols = set(featured_data["600000.SH"].columns)
        for symbol, data in featured_data.items():
            assert set(data.columns) == first_cols, f"{symbol} 特征不一致"


@pytest.mark.integration
class TestDataStorageIntegration:
    """测试数据存储流程"""

    def test_dataframe_serialization(self, sample_raw_data):
        """测试DataFrame序列化"""
        from utils.data_utils import serialize_dataframe, deserialize_dataframe

        # 序列化
        serialized = serialize_dataframe(sample_raw_data)
        assert isinstance(serialized, dict)
        assert "data" in serialized
        assert "columns" in serialized

        # 反序列化
        deserialized = deserialize_dataframe(serialized)
        pd.testing.assert_frame_equal(deserialized, sample_raw_data)

    def test_data_export_to_csv(self, sample_raw_data, tmp_path):
        """测试导出到CSV"""
        from utils.data_utils import export_to_csv

        # 导出
        file_path = tmp_path / "test_data.csv"
        export_to_csv(sample_raw_data, file_path)

        # 验证文件存在
        assert file_path.exists()

        # 读取并验证
        loaded_data = pd.read_csv(file_path)
        assert len(loaded_data) == len(sample_raw_data)

    def test_data_export_to_parquet(self, sample_raw_data, tmp_path):
        """测试导出到Parquet"""
        from utils.data_utils import export_to_parquet

        # 导出
        file_path = tmp_path / "test_data.parquet"
        export_to_parquet(sample_raw_data, file_path)

        # 验证文件存在
        assert file_path.exists()

        # 读取并验证
        loaded_data = pd.read_parquet(file_path)
        assert len(loaded_data) == len(sample_raw_data)


@pytest.mark.integration
class TestDataPipelineIntegration:
    """测试完整数据管道"""

    def test_complete_pipeline_single_stock(self):
        """测试单股票完整管道"""
        from backtest.engine.indicators import TechnicalIndicators
        from utils.data_utils import export_to_parquet, preprocess_data

        # 1. 生成原始数据
        dates = pd.date_range(start="2023-01-01", end="2023-03-31", freq="D")
        raw_data = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(10, 20, len(dates)),
                "high": np.random.uniform(15, 25, len(dates)),
                "low": np.random.uniform(8, 15, len(dates)),
                "close": np.random.uniform(10, 20, len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            }
        )

        # 2. 数据清洗
        cleaned_data = preprocess_data(raw_data, fill_method="ffill")

        # 3. 特征工程
        indicators = TechnicalIndicators()
        featured_data = indicators.add_all_indicators(cleaned_data)

        # 4. 导出数据
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            export_to_parquet(featured_data, f.name)
            assert Path(f.name).exists()
            Path(f.name).unlink()

        # 验证每一步都成功
        assert len(cleaned_data) > 0
        assert len(featured_data) > 0
        assert len(featured_data.columns) > len(raw_data.columns)

    def test_complete_pipeline_multi_stock(self, multi_stock_data):
        """测试多股票完整管道"""
        from backtest.engine.indicators import TechnicalIndicators
        from utils.data_utils import preprocess_data

        processed_data = {}

        # 对每只股票执行完整流程
        for symbol, data in multi_stock_data.items():
            # 清洗
            cleaned = preprocess_data(data, fill_method="ffill")

            # 特征工程
            indicators = TechnicalIndicators()
            featured = indicators.add_all_indicators(cleaned)

            processed_data[symbol] = featured

        # 验证所有股票都处理成功
        assert len(processed_data) == len(multi_stock_data)
        assert all(len(data) > 0 for data in processed_data.values())

    def test_data_pipeline_with_splits(self, sample_raw_data):
        """测试包含数据分割的管道"""
        from backtest.engine.indicators import TechnicalIndicators
        from utils.data_utils import preprocess_data, split_data

        # 1. 清洗数据
        cleaned_data = preprocess_data(sample_raw_data, fill_method="ffill")

        # 2. 特征工程
        indicators = TechnicalIndicators()
        featured_data = indicators.add_all_indicators(cleaned_data)

        # 3. 数据分割
        train_data, test_data = split_data(
            featured_data,
            train_ratio=0.8,
            method="time",
        )

        # 验证分割结果
        total_len = len(featured_data)
        assert len(train_data) + len(test_data) == total_len
        assert abs(len(train_data) / total_len - 0.8) < 0.05

        # 验证时间顺序
        if len(train_data) > 0 and len(test_data) > 0:
            assert train_data.iloc[-1]["date"] <= test_data.iloc[0]["date"]


@pytest.mark.integration
@pytest.mark.slow
class TestDataQualityIntegration:
    """测试数据质量检查"""

    def test_data_validation_checks(self, sample_raw_data):
        """测试数据验证检查"""
        from utils.data_utils import validate_data

        # 验证原始数据（有缺失值）
        issues = validate_data(sample_raw_data)
        assert len(issues) > 0  # 应该发现缺失值问题

    def test_outlier_detection(self, sample_raw_data):
        """测试异常值检测"""
        from utils.data_utils import detect_outliers

        # 添加一些异常值
        data_with_outliers = sample_raw_data.copy()
        data_with_outliers.loc[0, "close"] = 1000.0  # 明显异常
        data_with_outliers.loc[10, "volume"] = 1e15  # 明显异常

        outliers = detect_outliers(data_with_outliers, threshold=3.0)

        # 应该检测到异常值
        assert len(outliers) > 0

    def test_data_consistency_check(self, multi_stock_data):
        """测试数据一致性检查"""
        from utils.data_utils import check_data_consistency

        # 检查多股票数据一致性
        issues = check_data_consistency(multi_stock_data)

        # 验证检查结果
        assert isinstance(issues, dict)
        # 可能没有问题，也可能有，取决于数据生成器


@pytest.mark.integration
class TestDataPerformanceIntegration:
    """测试数据处理性能"""

    def test_large_data_processing(self):
        """测试大数据集处理性能"""
        import time

        from backtest.engine.indicators import TechnicalIndicators
        from utils.data_utils import preprocess_data

        # 生成大数据集（1年数据）
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="H")
        large_data = pd.DataFrame(
            {
                "date": dates,
                "open": np.random.uniform(10, 20, len(dates)),
                "high": np.random.uniform(15, 25, len(dates)),
                "low": np.random.uniform(8, 15, len(dates)),
                "close": np.random.uniform(10, 20, len(dates)),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            }
        )

        # 测试处理时间
        start_time = time.time()

        cleaned_data = preprocess_data(large_data, fill_method="ffill")
        indicators = TechnicalIndicators()
        featured_data = indicators.add_all_indicators(cleaned_data)

        elapsed_time = time.time() - start_time

        # 验证性能（应该在合理时间内完成）
        assert elapsed_time < 30.0  # 30秒内完成
        assert len(featured_data) == len(large_data)

    def test_batch_processing_performance(self, multi_stock_data):
        """测试批量处理性能"""
        import time

        from backtest.engine.indicators import TechnicalIndicators
        from utils.data_utils import preprocess_data

        start_time = time.time()

        # 批量处理多股票
        results = {}
        for symbol, data in multi_stock_data.items():
            cleaned = preprocess_data(data, fill_method="ffill")
            indicators = TechnicalIndicators()
            featured = indicators.add_all_indicators(cleaned)
            results[symbol] = featured

        elapsed_time = time.time() - start_time

        # 验证批量处理性能
        assert elapsed_time < 10.0  # 10秒内完成
        assert len(results) == len(multi_stock_data)
