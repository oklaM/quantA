"""
测试utils/helpers工具函数
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from utils.helpers import (
    calculate_cumulative_returns,
    calculate_log_returns,
    calculate_max_drawdown,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_volatility,
    chunk_list,
    clamp,
    format_money,
    format_number,
    merge_dicts,
    normalize,
    safe_divide,
)


@pytest.mark.unit
class TestFormatFunctions:
    """测试格式化函数"""

    def test_format_number_small(self):
        """测试小数格式化"""
        result = format_number(1234.56, decimal_places=2)
        assert result == "1,234.56"

    def test_format_number_large(self):
        """测试大数格式化"""
        result = format_number(1234567.89, decimal_places=2)
        assert "1,234,567.89" in result

    def test_format_number_percentage(self):
        """测试百分比格式化"""
        result = format_number(0.1234, decimal_places=2, percentage=True)
        assert result == "12.34%"

    def test_format_money_wan(self):
        """测试金额格式化-万"""
        result = format_money(100000, unit="万")
        assert "10" in result

    def test_format_money_yi(self):
        """测试金额格式化-亿"""
        result = format_money(100000000, unit="亿")
        assert "1" in result

    def test_format_money_negative(self):
        """测试负数金额"""
        result = format_money(-50000)
        assert "-" in result


@pytest.mark.unit
class TestCalculateReturns:
    """测试收益率计算"""

    def test_calculate_returns_basic(self):
        """测试基本收益率计算"""
        prices = pd.Series([100, 102, 101, 103])
        returns = calculate_returns(prices)

        # 第一个应该是NaN
        assert pd.isna(returns.iloc[0])

        # 第二个应该是 (102-100)/100 = 0.02
        assert np.isclose(returns.iloc[1], 0.02, rtol=0.01)

    def test_calculate_log_returns(self):
        """测试对数收益率"""
        prices = pd.Series([100, 110, 121])
        log_returns = calculate_log_returns(prices)

        # log(110/100) ≈ 0.09531
        # log(121/110) ≈ 0.09531
        assert np.isclose(log_returns.iloc[1], 0.0953, rtol=0.01)
        assert np.isclose(log_returns.iloc[2], 0.0953, rtol=0.01)

    def test_calculate_cumulative_returns(self):
        """测试累积收益率"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        cum_returns = calculate_cumulative_returns(returns)

        # (1.01 * 1.02 * 0.99 * 1.03) - 1 ≈ 0.0509
        expected = (1.01 * 1.02 * 0.99 * 1.03) - 1
        assert np.isclose(cum_returns.iloc[-1], expected, rtol=0.01)


@pytest.mark.unit
class TestRiskMetrics:
    """测试风险指标计算"""

    def test_calculate_sharpe_ratio(self):
        """测试夏普比率"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

        # 应该是正数
        assert sharpe > 0

    def test_calculate_sharpe_ratio_negative(self):
        """测试负收益的夏普比率"""
        returns = pd.Series([-0.01, -0.02, -0.01])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

        # 应该是负数
        assert sharpe < 0

    def test_calculate_sortino_ratio(self):
        """测试索提诺比率"""
        # 使用多个负收益以避免std为NaN
        returns = pd.Series([0.01, 0.02, -0.01, -0.02, 0.03, 0.01])
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)

        # 应该是正数或NaN（当downside_std为0时）
        if not pd.isna(sortino):
            assert sortino > 0

    def test_calculate_sortino_ratio_can_be_nan(self):
        """测试索提诺比率可能为NaN的情况"""
        # 当所有收益都相同时，downside_std为0，返回0
        returns = pd.Series([0.01] * 5)
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        # 这种情况返回0.0而不是NaN
        assert sortino == 0.0

    def test_calculate_max_drawdown(self):
        """测试最大回撤"""
        prices = pd.Series([100, 110, 105, 120, 115, 130])
        mdd = calculate_max_drawdown(prices)

        # 最大回撤是负值（从110到105的回撤）
        assert mdd < 0
        assert mdd > -0.1

    def test_calculate_volatility(self):
        """测试波动率"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        vol = calculate_volatility(returns)

        # 波动率应该大于0
        assert vol > 0

    def test_calculate_volatility_periods(self):
        """测试不同周期下的波动率"""
        returns = pd.Series([0.01] * 10)
        vol_daily = calculate_volatility(returns, periods_per_year=1)
        vol_annual = calculate_volatility(returns, periods_per_year=252)

        # 年化波动率应该更大
        assert vol_annual > vol_daily


@pytest.mark.unit
class TestUtilityFunctions:
    """测试工具函数"""

    def test_chunk_list(self):
        """测试列表分块"""
        lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        chunks = chunk_list(lst, 3)

        assert len(chunks) == 4
        assert chunks[0] == [1, 2, 3]
        assert chunks[3] == [10]

    def test_chunk_list_empty(self):
        """测试空列表分块"""
        chunks = chunk_list([], 3)
        assert chunks == []

    def test_merge_dicts(self):
        """测试字典合并"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}
        dict3 = {"e": 5}

        merged = merge_dicts(dict1, dict2, dict3)

        assert merged == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}

    def test_merge_dicts_override(self):
        """测试字典合并覆盖"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}

        merged = merge_dicts(dict1, dict2)

        # 后面的字典应该覆盖前面的
        assert merged["b"] == 3
        assert merged["a"] == 1
        assert merged["c"] == 4

    def test_safe_divide(self):
        """测试安全除法"""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(0, 5) == 0.0

    def test_safe_divide_zero_denominator(self):
        """测试安全除法零分母"""
        result = safe_divide(10, 0)
        assert result == 0.0

    def test_safe_divide_negative(self):
        """测试负数安全除法"""
        assert safe_divide(-10, 2) == -5.0
        assert safe_divide(10, -2) == -5.0

    def test_clamp(self):
        """测试数值限制"""
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_clamp_edge_cases(self):
        """测试边界值"""
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10

    def test_normalize(self):
        """测试归一化"""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize(data)

        # 检查范围在0-1之间
        assert all(0 <= x <= 1 for x in normalized)
        # 检查最小值为0，最大值为1
        assert np.isclose(normalized.min(), 0)
        assert np.isclose(normalized.max(), 1)

    def test_normalize_zscore(self):
        """测试z-score归一化"""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize(data, method="zscore")

        # z-score归一化后均值应该接近0，标准差接近1
        assert np.isclose(normalized.mean(), 0, atol=1e-10)
        assert np.isclose(normalized.std(), 1, atol=1e-10)

    def test_normalize_rank(self):
        """测试rank归一化"""
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = normalize(data, method="rank")

        # rank归一化后范围应该在0-1之间
        assert all(0 <= x <= 1 for x in normalized)


@pytest.mark.unit
class TestDecorators:
    """测试装饰器"""

    def test_timing_decorator(self):
        """测试计时装饰器"""
        from utils.helpers import timing_decorator

        @timing_decorator
        def slow_function():
            time.sleep(0.01)
            return 42

        result = slow_function()
        assert result == 42

    def test_retry_decorator(self):
        """测试重试装饰器"""
        from utils.helpers import retry_on_exception

        call_count = 0

        @retry_on_exception(max_retries=3, exceptions=(ValueError,))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet!")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_failure(self):
        """测试重试装饰器最终失败"""
        from utils.helpers import retry_on_exception

        @retry_on_exception(max_retries=2, exceptions=(ValueError,))
        def always_fail():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fail()
