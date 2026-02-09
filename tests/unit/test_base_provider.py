"""
数据源基类模块单元测试
测试 data/market/sources/base_provider.py 中的所有类和方法
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock
import pandas as pd

from data.market.sources.base_provider import (
    BaseDataProvider,
    DataProviderError,
    ConnectionError,
    DataNotFoundError,
    RateLimitError,
)


# 创建一个具体的实现用于测试
class ConcreteDataProvider(BaseDataProvider):
    """具体的数据提供者实现，用于测试"""

    def __init__(self, name="test_provider"):
        super().__init__(name)
        self._connected = False

    def connect(self) -> bool:
        self._is_connected = True
        return True

    def disconnect(self):
        self._is_connected = False

    def get_daily_bar(self, symbol, start_date=None, end_date=None, adjust="qfq"):
        # 返回一个空的DataFrame用于测试
        return pd.DataFrame()

    def get_minute_bar(self, symbol, trade_date, period=1):
        return pd.DataFrame()

    def get_realtime_quote(self, symbols):
        return pd.DataFrame()

    def get_stock_list(self, market=None):
        return pd.DataFrame()

    def get_index_list(self):
        return pd.DataFrame()

    def get_stock_info(self, symbol):
        return {}

    def get_financial(self, symbol, start_date=None, end_date=None):
        return pd.DataFrame()


@pytest.mark.unit
class TestBaseDataProvider:
    """数据源基类测试"""

    def test_provider_creation(self):
        """测试数据提供者创建"""
        provider = ConcreteDataProvider("test_provider")
        assert provider.name == "test_provider"
        assert provider.is_connected is False

    def test_provider_connect(self):
        """测试连接"""
        provider = ConcreteDataProvider("test_provider")
        result = provider.connect()
        assert result is True
        assert provider.is_connected is True

    def test_provider_disconnect(self):
        """测试断开连接"""
        provider = ConcreteDataProvider("test_provider")
        provider.connect()
        provider.disconnect()
        assert provider.is_connected is False

    def test_provider_is_connected_property(self):
        """测试is_connected属性"""
        provider = ConcreteDataProvider("test_provider")
        assert provider.is_connected is False

        provider.connect()
        assert provider.is_connected is True

        provider.disconnect()
        assert provider.is_connected is False


@pytest.mark.unit
class TestNormalizeSymbol:
    """股票代码标准化测试"""

    def test_normalize_symbol_with_suffix(self):
        """测试已有后缀的代码"""
        provider = ConcreteDataProvider()
        assert provider.normalize_symbol("600519.SH") == "600519.SH"
        assert provider.normalize_symbol("000858.SZ") == "000858.SZ"

    def test_normalize_symbol_shanghai(self):
        """测试上海市场代码"""
        provider = ConcreteDataProvider()
        assert provider.normalize_symbol("600519") == "600519.SH"
        assert provider.normalize_symbol("688001") == "688001.SH"  # 科创板

    def test_normalize_symbol_shenzhen(self):
        """测试深圳市场代码"""
        provider = ConcreteDataProvider()
        assert provider.normalize_symbol("000858") == "000858.SZ"
        assert provider.normalize_symbol("300750") == "300750.SZ"  # 创业板

    def test_normalize_symbol_with_spaces(self):
        """测试带空格的代码"""
        provider = ConcreteDataProvider()
        assert provider.normalize_symbol(" 600519 ") == "600519.SH"
        assert provider.normalize_symbol(" 000858.SZ ") == "000858.SZ"

    def test_normalize_symbol_lowercase(self):
        """测试小写代码"""
        provider = ConcreteDataProvider()
        assert provider.normalize_symbol("600519.sh") == "600519.SH"
        assert provider.normalize_symbol("000858.sz") == "000858.SZ"

    def test_normalize_symbol_unknown_prefix(self):
        """测试未知前缀"""
        provider = ConcreteDataProvider()
        # 不是6、0、3开头的代码
        assert provider.normalize_symbol("888888") == "888888"
        assert provider.normalize_symbol("123456") == "123456"

    def test_normalize_symbol_empty(self):
        """测试空字符串"""
        provider = ConcreteDataProvider()
        assert provider.normalize_symbol("") == ""

    def test_normalize_symbol_multiple_dots(self):
        """测试多个点"""
        provider = ConcreteDataProvider()
        # 已有后缀的应该保持不变
        assert provider.normalize_symbol("600519.SH.") == "600519.SH."


@pytest.mark.unit
class TestValidateDateRange:
    """日期范围验证测试"""

    def test_validate_date_range_both_none(self):
        """测试两个参数都为None"""
        provider = ConcreteDataProvider()
        start, end = provider.validate_date_range(None, None)

        assert end is not None
        assert start is not None
        # 默认应该是最近一年
        assert len(start) == 8  # YYYYMMDD格式
        assert len(end) == 8

    def test_validate_date_range_with_start(self):
        """测试只有开始日期"""
        provider = ConcreteDataProvider()
        start, end = provider.validate_date_range("20240101", None)

        assert start == "20240101"
        assert end is not None

    def test_validate_date_range_with_end(self):
        """测试只有结束日期"""
        provider = ConcreteDataProvider()
        start, end = provider.validate_date_range(None, "20241231")

        assert start is not None
        assert end == "20241231"

    def test_validate_date_range_both_provided(self):
        """测试两个参数都提供"""
        provider = ConcreteDataProvider()
        start, end = provider.validate_date_range("20240101", "20241231")

        assert start == "20240101"
        assert end == "20241231"

    def test_validate_date_range_default_format(self):
        """测试默认日期格式"""
        provider = ConcreteDataProvider()
        start, end = provider.validate_date_range(None, None)

        # 应该是YYYYMMDD格式
        assert len(start) == 8
        assert len(end) == 8
        assert start.isdigit()
        assert end.isdigit()


@pytest.mark.unit
class TestDataProviderExceptions:
    """数据提供者异常测试"""

    def test_data_provider_error(self):
        """测试基础异常"""
        error = DataProviderError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_connection_error(self):
        """测试连接异常"""
        error = ConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, DataProviderError)

    def test_data_not_found_error(self):
        """测试数据不存在异常"""
        error = DataNotFoundError("Data not found")
        assert str(error) == "Data not found"
        assert isinstance(error, DataProviderError)

    def test_rate_limit_error(self):
        """测试频率限制异常"""
        error = RateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, DataProviderError)

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        assert issubclass(ConnectionError, DataProviderError)
        assert issubclass(DataNotFoundError, DataProviderError)
        assert issubclass(RateLimitError, DataProviderError)

    def test_exception_raising(self):
        """测试异常抛出"""
        with pytest.raises(DataProviderError, match="Test error"):
            raise DataProviderError("Test error")

        with pytest.raises(ConnectionError):
            raise ConnectionError()

        with pytest.raises(DataNotFoundError):
            raise DataNotFoundError()

        with pytest.raises(RateLimitError):
            raise RateLimitError()


@pytest.mark.unit
class TestAbstractMethods:
    """抽象方法测试"""

    def test_abstract_methods_must_be_implemented(self):
        """测试抽象方法必须被实现"""
        # 尝试创建不实现所有抽象方法的类应该失败
        with pytest.raises(TypeError):
            # 这个类没有实现所有抽象方法
            class IncompleteProvider(BaseDataProvider):
                def __init__(self):
                    super().__init__("incomplete")

                # 只实现了部分方法
                def connect(self):
                    pass

            # 尝试实例化应该失败
            IncompleteProvider()

    def test_concrete_provider_has_all_methods(self):
        """测试具体提供者有所有方法"""
        provider = ConcreteDataProvider()

        # 检查所有必需的方法都存在
        assert hasattr(provider, 'connect')
        assert hasattr(provider, 'disconnect')
        assert hasattr(provider, 'get_daily_bar')
        assert hasattr(provider, 'get_minute_bar')
        assert hasattr(provider, 'get_realtime_quote')
        assert hasattr(provider, 'get_stock_list')
        assert hasattr(provider, 'get_index_list')
        assert hasattr(provider, 'get_stock_info')
        assert hasattr(provider, 'get_financial')
        assert hasattr(provider, 'normalize_symbol')
        assert hasattr(provider, 'validate_date_range')

    def test_abstract_methods_are_callable(self):
        """测试抽象方法可调用"""
        provider = ConcreteDataProvider()

        assert callable(provider.connect)
        assert callable(provider.disconnect)
        assert callable(provider.get_daily_bar)
        assert callable(provider.get_minute_bar)
        assert callable(provider.get_realtime_quote)
        assert callable(provider.get_stock_list)
        assert callable(provider.get_index_list)
        assert callable(provider.get_stock_info)
        assert callable(provider.get_financial)


@pytest.mark.unit
class TestDataProviderInterface:
    """数据提供者接口测试"""

    def test_get_daily_bar_signature(self):
        """测试日线数据接口签名"""
        provider = ConcreteDataProvider()
        # 应该接受这些参数
        result = provider.get_daily_bar(
            symbol="600519.SH",
            start_date="20240101",
            end_date="20241231",
            adjust="qfq"
        )
        assert isinstance(result, pd.DataFrame)

    def test_get_daily_bar_default_adjust(self):
        """测试默认复权类型"""
        provider = ConcreteDataProvider()
        result = provider.get_daily_bar("600519.SH")
        assert isinstance(result, pd.DataFrame)

    def test_get_minute_bar_signature(self):
        """测试分钟线数据接口签名"""
        provider = ConcreteDataProvider()
        result = provider.get_minute_bar(
            symbol="600519.SH",
            trade_date="20240101",
            period=5
        )
        assert isinstance(result, pd.DataFrame)

    def test_get_minute_bar_default_period(self):
        """测试默认周期"""
        provider = ConcreteDataProvider()
        result = provider.get_minute_bar("600519.SH", "20240101")
        assert isinstance(result, pd.DataFrame)

    def test_get_realtime_quote_signature(self):
        """测试实时行情接口签名"""
        provider = ConcreteDataProvider()
        symbols = ["600519.SH", "000858.SZ"]
        result = provider.get_realtime_quote(symbols)
        assert isinstance(result, pd.DataFrame)

    def test_get_stock_list_signature(self):
        """测试股票列表接口签名"""
        provider = ConcreteDataProvider()
        result = provider.get_stock_list(market="SH")
        assert isinstance(result, pd.DataFrame)

    def test_get_stock_list_default_market(self):
        """测试默认市场参数"""
        provider = ConcreteDataProvider()
        result = provider.get_stock_list()
        assert isinstance(result, pd.DataFrame)

    def test_get_stock_info_signature(self):
        """测试股票信息接口签名"""
        provider = ConcreteDataProvider()
        result = provider.get_stock_info("600519.SH")
        assert isinstance(result, dict)

    def test_get_financial_signature(self):
        """测试财务数据接口签名"""
        provider = ConcreteDataProvider()
        result = provider.get_financial(
            symbol="600519.SH",
            start_date="20240101",
            end_date="20241231"
        )
        assert isinstance(result, pd.DataFrame)


@pytest.mark.unit
class TestDataProviderEdgeCases:
    """数据提供者边界情况测试"""

    def test_normalize_symbol_very_long(self):
        """测试很长的代码"""
        provider = ConcreteDataProvider()
        long_symbol = "A" * 100
        result = provider.normalize_symbol(long_symbol)
        # 应该不会崩溃
        assert isinstance(result, str)

    def test_normalize_symbol_special_chars(self):
        """测试特殊字符"""
        provider = ConcreteDataProvider()
        # 包含特殊字符的代码
        result = provider.normalize_symbol("600519-SH")
        # 应该保持不变（因为不是以.开头）
        assert "600519-SH" in result

    def test_validate_date_range_invalid_format(self):
        """测试无效日期格式"""
        provider = ConcreteDataProvider()
        # 即使格式不正确，也应该返回一些值
        start, end = provider.validate_date_range("invalid", None)
        # 实际行为取决于实现
        assert isinstance(start, str)
        assert isinstance(end, str)

    def test_multiple_connect_disconnect(self):
        """测试多次连接断开"""
        provider = ConcreteDataProvider()

        # 连接
        provider.connect()
        assert provider.is_connected

        # 断开
        provider.disconnect()
        assert not provider.is_connected

        # 再次连接
        provider.connect()
        assert provider.is_connected

        # 再次断开
        provider.disconnect()
        assert not provider.is_connected

    def test_disconnect_without_connect(self):
        """测试未连接就断开"""
        provider = ConcreteDataProvider()
        # 未连接就断开应该不会出错
        provider.disconnect()
        assert not provider.is_connected

    def test_connect_already_connected(self):
        """测试已经连接再连接"""
        provider = ConcreteDataProvider()
        provider.connect()
        assert provider.is_connected

        # 再次连接
        provider.connect()
        assert provider.is_connected

    def test_provider_name(self):
        """测试提供者名称"""
        provider1 = ConcreteDataProvider("provider1")
        provider2 = ConcreteDataProvider("provider2")

        assert provider1.name == "provider1"
        assert provider2.name == "provider2"
        assert provider1.name != provider2.name


@pytest.mark.unit
class TestDataProviderUtilityMethods:
    """数据提供者工具方法测试"""

    def test_normalize_symbol_returns_string(self):
        """测试标准化返回字符串"""
        provider = ConcreteDataProvider()
        result = provider.normalize_symbol("600519")
        assert isinstance(result, str)

    def test_normalize_symbol_case_conversion(self):
        """测试大小写转换"""
        provider = ConcreteDataProvider()
        # 应该转换为大写
        assert provider.normalize_symbol("600519.sh") == "600519.SH"
        assert provider.normalize_symbol("600519.Sh") == "600519.SH"
        assert provider.normalize_symbol("600519.sH") == "600519.SH"

    def test_normalize_symbol_whitespace_removal(self):
        """测试空白字符删除"""
        provider = ConcreteDataProvider()
        # 应该删除前后空白
        assert provider.normalize_symbol("\t600519\n") == "600519.SH"
        assert provider.normalize_symbol("\r\n000858.SZ\r\n") == "000858.SZ"

    def test_validate_date_range_future_date(self):
        """测试未来日期"""
        provider = ConcreteDataProvider()
        # 未来日期应该也能处理
        future_date = "20991231"
        start, end = provider.validate_date_range("20240101", future_date)
        assert start == "20240101"
        assert end == future_date

    def test_validate_date_range_past_date(self):
        """测试过去日期"""
        provider = ConcreteDataProvider()
        # 很久以前的日期
        past_date = "19900101"
        start, end = provider.validate_date_range(past_date, "20240101")
        assert start == past_date
        assert end == "20240101"


@pytest.mark.unit
class TestDataProviderStateManagement:
    """数据提供者状态管理测试"""

    def test_connection_state_changes(self):
        """测试连接状态变化"""
        provider = ConcreteDataProvider()

        # 初始状态
        assert not provider.is_connected

        # 连接后
        provider.connect()
        assert provider.is_connected

        # 断开后
        provider.disconnect()
        assert not provider.is_connected

    def test_multiple_instances_independent(self):
        """测试多个实例独立"""
        provider1 = ConcreteDataProvider("provider1")
        provider2 = ConcreteDataProvider("provider2")

        # 连接provider1
        provider1.connect()
        assert provider1.is_connected
        assert not provider2.is_connected

        # 连接provider2
        provider2.connect()
        assert provider1.is_connected
        assert provider2.is_connected

        # 断开provider1
        provider1.disconnect()
        assert not provider1.is_connected
        assert provider2.is_connected
