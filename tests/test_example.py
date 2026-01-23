"""
示例测试文件 - 验证测试框架配置
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.unit
def test_sample_fixture(sample_price_data):
    """测试sample_price_data fixture是否正常工作"""
    assert isinstance(sample_price_data, pd.DataFrame)
    assert len(sample_price_data) == 100
    assert "datetime" in sample_price_data.columns
    assert "close" in sample_price_data.columns
    assert "volume" in sample_price_data.columns


@pytest.mark.unit
def test_numpy_operations():
    """测试基本的numpy操作"""
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15


@pytest.mark.unit
def test_pandas_operations():
    """测试基本的pandas操作"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert len(df) == 3
    assert df["a"].sum() == 6
    assert df["b"].mean() == 5.0


@pytest.mark.unit
def test_project_imports():
    """测试项目模块是否可以正常导入"""
    try:
        import sys
        from pathlib import Path

        # 添加项目根目录到路径
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        # 尝试导入主要模块
        from config import settings
        from utils import logger

        assert True
    except ImportError as e:
        pytest.fail(f"导入模块失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
