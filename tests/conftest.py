"""
pytest配置和共享fixtures
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """项目根目录路径"""
    return project_root


@pytest.fixture(scope="function")
def sample_price_data():
    """
    生成示例价格数据用于测试

    Returns:
        pd.DataFrame: 包含OHLCV数据的DataFrame
    """
    np.random.seed(42)
    n = 100

    # 生成随机价格数据
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n)
    prices = base_price * (1 + returns).cumprod()

    dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq="D")

    data = pd.DataFrame(
        {
            "datetime": dates,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, n),
        }
    )

    # 确保high >= close >= low
    data["high"] = data[["open", "close", "high"]].max(axis=1)
    data["low"] = data[["open", "close", "low"]].min(axis=1)

    return data


@pytest.fixture(scope="function")
def sample_multi_symbol_data():
    """
    生成多股票示例数据

    Returns:
        dict: {symbol: DataFrame}
    """
    symbols = ["000001.SZ", "600000.SH", "300001.SZ"]
    data = {}

    for symbol in symbols:
        np.random.seed(hash(symbol) % 1000)
        n = 100
        base_price = np.random.uniform(10, 100)
        returns = np.random.normal(0, 0.02, n)
        prices = base_price * (1 + returns).cumprod()

        dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq="D")

        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, n)),
                "high": prices * (1 + np.random.uniform(0, 0.02, n)),
                "low": prices * (1 - np.random.uniform(0, 0.02, n)),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, n),
            }
        )

        df["high"] = df[["open", "close", "high"]].max(axis=1)
        df["low"] = df[["open", "close", "low"]].min(axis=1)
        data[symbol] = df

    return data


@pytest.fixture(scope="function")
def sample_signals():
    """
    生成示例交易信号

    Returns:
        pd.DataFrame: 包含交易信号的DataFrame
    """
    dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=10, freq="D")

    signals = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": ["000001.SZ"] * 10,
            "signal": [0, 1, 1, 0, 0, -1, 0, 1, 0, -1],
            "price": np.random.uniform(90, 110, 10),
        }
    )

    return signals


@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    """临时目录fixture"""
    return tmp_path


@pytest.fixture(scope="session")
def test_config():
    """
    测试配置

    Returns:
        dict: 测试配置字典
    """
    return {
        "symbols": ["000001.SZ", "600000.SH"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 1000000,
        "commission": 0.0003,
        "slippage": 0.0001,
    }
