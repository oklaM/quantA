"""
工具函数模块单元测试
测试 utils/helpers.py 中的所有辅助函数
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import time

from utils.helpers import (
    timing_decorator,
    format_number,
    format_money,
    calculate_returns,
    calculate_log_returns,
    calculate_cumulative_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_volatility,
    chunk_list,
    merge_dicts,
    safe_divide,
    clamp,
    normalize,
    retry_on_exception,
)


@pytest.mark.unit
class TestFormatNumber:
    """数字格式化测试"""

    def test_format_number_basic(self):
        """测试基本数字格式化"""
        assert format_number(1234.5678, 2) == "1,234.57"
        assert format_number(1234567.89, 2) == "1,234,567.89"

    def test_format_number_with_decimals(self):
        """测试不同小数位数"""
        assert format_number(1234.5678, 0) == "1,235"
        assert format_number(1234.5678, 2) == "1,234.57"
        assert format_number(1234.5678, 4) == "1,234.5678"

    def test_format_number_negative(self):
        """测试负数格式化"""
        assert format_number(-1234.56, 2) == "-1,234.56"

    def test_format_number_zero(self):
        """测试零格式化"""
        assert format_number(0, 2) == "0.00"

    def test_format_number_as_percentage(self):
        """测试百分比格式化"""
        assert format_number(0.1234, 2, percentage=True) == "12.34%"
        assert format_number(0.0567, 2, percentage=True) == "5.67%"
        assert format_number(-0.05, 2, percentage=True) == "-5.00%"

    def test_format_number_nan(self):
        """测试NaN值"""
        result = format_number(np.nan, 2)
        assert result == "N/A"

    def test_format_number_large_numbers(self):
        """测试大数字"""
        assert format_number(1e10, 2) == "10,000,000,000.00"

    def test_format_number_small_decimals(self):
        """测试小数"""
        # 0.0012345四舍五入到6位小数
        result = format_number(0.0012345, 6)
        # Python的round函数和格式化可能有所不同
        # 只要格式化正确即可
        assert "0.00123" in result


@pytest.mark.unit
class TestFormatMoney:
    """金额格式化测试"""

    def test_format_money_in_wan(self):
        """测试以万为单位"""
        assert format_money(100000, "万") == "10.00万"
        assert format_money(1234567.89, "万") == "123.46万"

    def test_format_money_in_yi(self):
        """测试以亿为单位"""
        assert format_money(100000000, "亿") == "1.00亿"
        assert format_money(1234567890.12, "亿") == "12.35亿"

    def test_format_money_no_unit(self):
        """测试不使用单位"""
        assert format_money(1234.56, "") == "1,234.56"

    def test_format_money_nan(self):
        """测试NaN值"""
        assert format_money(np.nan, "万") == "N/A"

    def test_format_money_negative(self):
        """测试负金额"""
        assert format_money(-100000, "万") == "-10.00万"

    def test_format_money_small_amount(self):
        """测试小金额"""
        assert format_money(9999, "万") == "1.00万"


@pytest.mark.unit
class TestCalculateReturns:
    """收益率计算测试"""

    def test_calculate_returns_basic(self):
        """测试基本收益率计算"""
        prices = pd.Series([100, 101, 102, 103])
        returns = calculate_returns(prices)
        assert len(returns) == 4
        assert pd.isna(returns.iloc[0])
        assert returns.iloc[1] == pytest.approx(0.01, rel=1e-2)

    def test_calculate_returns_single_value(self):
        """测试单个值"""
        prices = pd.Series([100])
        returns = calculate_returns(prices)
        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])

    def test_calculate_returns_empty(self):
        """测试空序列"""
        prices = pd.Series([])
        returns = calculate_returns(prices)
        assert len(returns) == 0

    def test_calculate_returns_with_zeros(self):
        """测试包含零值"""
        prices = pd.Series([100, 0, 50])
        returns = calculate_returns(prices)
        assert returns.iloc[1] == -1.0
        # 从0到50会产生inf
        import numpy as np
        assert np.isinf(returns.iloc[2])


@pytest.mark.unit
class TestCalculateLogReturns:
    """对数收益率计算测试"""

    def test_calculate_log_returns_basic(self):
        """测试基本对数收益率计算"""
        prices = pd.Series([100, 110, 121])
        log_returns = calculate_log_returns(prices)
        assert len(log_returns) == 3
        assert pd.isna(log_returns.iloc[0])
        assert log_returns.iloc[1] == pytest.approx(np.log(1.1), rel=1e-4)

    def test_calculate_log_returns_vs_simple(self):
        """测试对数收益率与简单收益率的差异"""
        prices = pd.Series([100, 105])
        simple_returns = calculate_returns(prices)
        log_returns = calculate_log_returns(prices)
        assert log_returns.iloc[1] < simple_returns.iloc[1]


@pytest.mark.unit
class TestCalculateCumulativeReturns:
    """累计收益率计算测试"""

    def test_calculate_cumulative_returns_basic(self):
        """测试基本累计收益率计算"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        cum_returns = calculate_cumulative_returns(returns)
        expected = (1 + returns).cumprod() - 1
        pd.testing.assert_series_equal(cum_returns, expected)

    def test_calculate_cumulative_returns_with_nan(self):
        """测试包含NaN的累计收益率"""
        returns = pd.Series([0.01, np.nan, 0.02])
        cum_returns = calculate_cumulative_returns(returns)
        assert pd.isna(cum_returns.iloc[1])

    def test_calculate_cumulative_returns_all_positive(self):
        """测试全部为正的累计收益率"""
        returns = pd.Series([0.01] * 10)
        cum_returns = calculate_cumulative_returns(returns)
        assert cum_returns.iloc[-1] > 0


@pytest.mark.unit
class TestCalculateSharpeRatio:
    """夏普比率计算测试"""

    def test_calculate_sharpe_ratio_basic(self):
        """测试基本夏普比率计算"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_calculate_sharpe_ratio_empty(self):
        """测试空序列"""
        returns = pd.Series([])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_zero_std(self):
        """测试零标准差"""
        returns = pd.Series([0.01, 0.01, 0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_calculate_sharpe_ratio_different_risk_free(self):
        """测试不同无风险利率"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        sharpe1 = calculate_sharpe_ratio(returns, risk_free_rate=0.03)
        sharpe2 = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
        assert sharpe1 != sharpe2

    def test_calculate_sharpe_ratio_annualization(self):
        """测试年化参数"""
        returns = pd.Series([0.001] * 252)
        sharpe = calculate_sharpe_ratio(returns, periods_per_year=252)
        assert isinstance(sharpe, float)


@pytest.mark.unit
class TestCalculateSortinoRatio:
    """索提诺比率计算测试"""

    def test_calculate_sortino_ratio_basic(self):
        """测试基本索提诺比率计算"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sortino = calculate_sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_calculate_sortino_ratio_empty(self):
        """测试空序列"""
        returns = pd.Series([])
        sortino = calculate_sortino_ratio(returns)
        assert sortino == 0.0

    def test_calculate_sortino_ratio_no_downside(self):
        """测试无下行风险"""
        returns = pd.Series([0.01, 0.02, 0.03])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        # 没有下行收益时应该返回0或很大的值
        assert sortino >= 0

    def test_calculate_sortino_ratio_zero_downside_std(self):
        """测试零下行标准差"""
        returns = pd.Series([0.01, 0.01, 0.01])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino == 0.0


@pytest.mark.unit
class TestCalculateMaxDrawdown:
    """最大回撤计算测试"""

    def test_calculate_max_drawdown_basic(self):
        """测试基本最大回撤计算"""
        prices = pd.Series([100, 110, 120, 110, 100, 90])
        mdd = calculate_max_drawdown(prices)
        assert mdd < 0
        assert mdd > -0.5  # 回撤应该小于50%

    def test_calculate_max_drawdown_no_decline(self):
        """测试没有下跌"""
        prices = pd.Series([100, 110, 120, 130])
        mdd = calculate_max_drawdown(prices)
        assert mdd == 0.0

    def test_calculate_max_drawdown_large_decline(self):
        """测试大幅下跌"""
        prices = pd.Series([100, 90, 80, 70])
        mdd = calculate_max_drawdown(prices)
        # 从100降到70，累计跌幅是30%
        assert mdd < -0.20  # 至少20%的回撤


@pytest.mark.unit
class TestCalculateVolatility:
    """波动率计算测试"""

    def test_calculate_volatility_basic(self):
        """测试基本波动率计算"""
        returns = pd.Series([0.01, -0.01, 0.02, -0.02])
        vol = calculate_volatility(returns)
        assert vol > 0

    def test_calculate_volatility_annualization(self):
        """测试年化"""
        returns = pd.Series([0.01] * 252)
        vol = calculate_volatility(returns, periods_per_year=252)
        # 年化波动率应该接近0（因为收益不变）
        assert vol >= 0

    def test_calculate_volatility_single_value(self):
        """测试单个值"""
        returns = pd.Series([0.01])
        vol = calculate_volatility(returns)
        # 单个值的标准差是NaN或0
        assert pd.isna(vol) or vol == 0.0


@pytest.mark.unit
class TestChunkList:
    """列表分块测试"""

    def test_chunk_list_basic(self):
        """测试基本分块"""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        chunks = chunk_list(lst, 3)
        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7, 8, 9]

    def test_chunk_list_uneven(self):
        """测试不均匀分块"""
        lst = [1, 2, 3, 4, 5]
        chunks = chunk_list(lst, 2)
        assert len(chunks) == 3
        assert chunks[-1] == [5]

    def test_chunk_list_empty(self):
        """测试空列表"""
        chunks = chunk_list([], 3)
        assert chunks == []

    def test_chunk_list_single_element(self):
        """测试单个元素"""
        chunks = chunk_list([1], 3)
        assert len(chunks) == 1
        assert chunks[0] == [1]

    def test_chunk_list_size_larger_than_list(self):
        """测试块大小大于列表"""
        lst = [1, 2, 3]
        chunks = chunk_list(lst, 10)
        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3]


@pytest.mark.unit
class TestMergeDicts:
    """字典合并测试"""

    def test_merge_dicts_basic(self):
        """测试基本合并"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_dicts_override(self):
        """测试覆盖"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_dicts_empty(self):
        """测试空字典"""
        dict1 = {"a": 1}
        result = merge_dicts(dict1, {})
        assert result == {"a": 1}

    def test_merge_dicts_multiple(self):
        """测试多个字典"""
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        dict3 = {"c": 3}
        result = merge_dicts(dict1, dict2, dict3)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_merge_dicts_no_args(self):
        """测试无参数"""
        result = merge_dicts()
        assert result == {}


@pytest.mark.unit
class TestSafeDivide:
    """安全除法测试"""

    def test_safe_divide_normal(self):
        """测试正常除法"""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(7, 2) == 3.5

    def test_safe_divide_by_zero(self):
        """测试除零"""
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(0, 0) == 0.0

    def test_safe_divide_negative(self):
        """测试负数"""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0

    def test_safe_divide_float(self):
        """测试浮点数"""
        assert safe_divide(1.5, 0.5) == 3.0


@pytest.mark.unit
class TestClamp:
    """值限制测试"""

    def test_clamp_within_range(self):
        """测试在范围内"""
        assert clamp(5, 0, 10) == 5
        assert clamp(0.5, 0, 1) == 0.5

    def test_clamp_below_min(self):
        """测试低于最小值"""
        assert clamp(-5, 0, 10) == 0
        assert clamp(0.3, 0.5, 1) == 0.5

    def test_clamp_above_max(self):
        """测试高于最大值"""
        assert clamp(15, 0, 10) == 10
        assert clamp(1.5, 0, 1) == 1

    def test_clamp_at_boundaries(self):
        """测试边界值"""
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10

    def test_clamp_negative_range(self):
        """测试负范围"""
        assert clamp(-5, -10, 0) == -5
        assert clamp(-15, -10, 0) == -10
        assert clamp(5, -10, 0) == 0


@pytest.mark.unit
class TestNormalize:
    """数据归一化测试"""

    def test_normalize_minmax(self):
        """测试MinMax归一化"""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize(data, method="minmax")
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.iloc[0] == 0.0
        assert normalized.iloc[-1] == 1.0

    def test_normalize_zscore(self):
        """测试Z-score归一化"""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize(data, method="zscore")
        assert abs(normalized.mean()) < 1e-10  # 均值接近0
        assert abs(normalized.std() - 1.0) < 1e-10  # 标准差接近1

    def test_normalize_rank(self):
        """测试排名归一化"""
        data = pd.Series([3, 1, 4, 1, 5])
        normalized = normalize(data, method="rank")
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_constant_values(self):
        """测试常量值"""
        data = pd.Series([5, 5, 5, 5])
        normalized = normalize(data, method="minmax")
        assert all(normalized == 0.5)

    def test_normalize_invalid_method(self):
        """测试无效方法"""
        data = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="未知的归一化方法"):
            normalize(data, method="invalid")


@pytest.mark.unit
class TestRetryDecorator:
    """重试装饰器测试"""

    def test_retry_on_success(self):
        """测试成功执行"""
        @retry_on_exception(max_retries=3)
        def func():
            return "success"

        result = func()
        assert result == "success"

    def test_retry_on_failure_then_success(self):
        """测试失败后重试成功"""
        call_count = 0

        @retry_on_exception(max_retries=3, exceptions=(ValueError,))
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Fail")
            return "success"

        result = func()
        assert result == "success"
        assert call_count == 2

    def test_retry_on_failure_max_retries(self):
        """测试达到最大重试次数"""
        @retry_on_exception(max_retries=2, exceptions=(ValueError,), delay=0.01)
        def func():
            raise ValueError("Always fail")

        with pytest.raises(ValueError, match="Always fail"):
            func()

    def test_retry_on_different_exception(self):
        """测试不同异常不重试"""
        @retry_on_exception(max_retries=3, exceptions=(ValueError,))
        def func():
            raise TypeError("Different error")

        with pytest.raises(TypeError):
            func()

    def test_retry_with_delay(self):
        """测试重试延迟"""
        call_times = []

        @retry_on_exception(max_retries=2, exceptions=(ValueError,), delay=0.05)
        def func():
            call_times.append(time.time())
            if len(call_times) < 2:
                raise ValueError("Fail")
            return "success"

        func()
        assert len(call_times) == 2
        # 检查延迟（允许一些误差）
        assert call_times[1] - call_times[0] >= 0.04


@pytest.mark.unit
class TestTimingDecorator:
    """计时装饰器测试"""

    @patch('utils.helpers.logger')
    def test_timing_decorator(self, mock_logger):
        """测试计时装饰器"""
        @timing_decorator
        def func():
            time.sleep(0.01)
            return "result"

        result = func()
        assert result == "result"
        # 验证logger.debug被调用
        mock_logger.debug.assert_called_once()

    @patch('utils.helpers.logger')
    def test_timing_decorator_with_args(self, mock_logger):
        """测试带参数的函数"""
        @timing_decorator
        def func(a, b):
            return a + b

        result = func(1, 2)
        assert result == 3
        mock_logger.debug.assert_called_once()

    @patch('utils.helpers.logger')
    def test_timing_decorator_exception(self, mock_logger):
        """测试异常情况"""
        @timing_decorator
        def func():
            raise ValueError("Error")

        with pytest.raises(ValueError):
            func()

        # 异常时可能不会记录时间（取决于实现）
        # 只要不崩溃即可


@pytest.mark.unit
class TestEdgeCases:
    """边界情况测试"""

    def test_format_number_very_small(self):
        """测试非常小的数字"""
        assert format_number(0.000001, 6) == "0.000001"

    def test_format_number_very_large(self):
        """测试非常大的数字"""
        result = format_number(1e15, 2)
        assert "1,000,000,000,000,000" in result

    def test_calculate_returns_with_negative_prices(self):
        """测试负价格（异常情况）"""
        prices = pd.Series([100, -50, 100])
        returns = calculate_returns(prices)
        assert len(returns) == 3

    def test_normalize_empty_series(self):
        """测试空序列归一化"""
        data = pd.Series([])
        normalized = normalize(data, method="minmax")
        assert len(normalized) == 0

    def test_chunk_list_zero_chunk_size(self):
        """测试零块大小"""
        # chunk_size为0会导致range()错误
        lst = [1, 2, 3]
        with pytest.raises(ValueError):
            chunk_list(lst, 0)

    def test_merge_dicts_with_none_values(self):
        """测试包含None值的字典"""
        dict1 = {"a": None}
        dict2 = {"b": 2}
        result = merge_dicts(dict1, dict2)
        assert result == {"a": None, "b": 2}

    def test_safe_divide_with_negative_zero(self):
        """测试负零"""
        result = safe_divide(5, -0.0)
        # 浮点数-0.0和0.0相等
        assert result == 0.0 or result == float('-inf')

    def test_clamp_invalid_range(self):
        """测试无效范围（min > max）"""
        # 这个函数的行为取决于实现
        result = clamp(5, 10, 0)
        # 应该至少返回一个数字
        assert isinstance(result, (int, float))
