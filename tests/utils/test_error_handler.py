"""
测试统一错误处理机制
"""

import asyncio
import pytest

from utils.error_handler import (
    AgentError,
    BacktestError,
    DataError,
    ErrorHandler,
    QuantAError,
    RLError,
    create_error,
    format_error_for_api,
    handle_errors,
    log_and_raise,
)


class TestQuantAError:
    """测试基础异常类"""

    def test_basic_error(self):
        """测试基本异常"""
        error = QuantAError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.details == {}

    def test_error_with_details(self):
        """测试带详情的异常"""
        error = QuantAError("Test error", details={"key": "value"})
        assert "key=value" in str(error)
        assert error.details == {"key": "value"}

    def test_error_to_dict(self):
        """测试转换为字典"""
        error = QuantAError("Test error", details={"key": "value"})
        error_dict = error.to_dict()
        assert error_dict["error_type"] == "QuantAError"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}


class TestDataError:
    """测试数据异常"""

    def test_data_error_basic(self):
        """测试基本数据异常"""
        error = DataError("Data fetch failed")
        assert isinstance(error, QuantAError)
        assert "Data fetch failed" in str(error)

    def test_data_error_with_details(self):
        """测试带详情的数据异常"""
        error = DataError(
            "Data fetch failed",
            data_source="tushare",
            symbol="600000",
            details={"retry_count": 3},
        )
        assert error.details["data_source"] == "tushare"
        assert error.details["symbol"] == "600000"
        assert error.details["retry_count"] == 3


class TestAgentError:
    """测试Agent异常"""

    def test_agent_error_basic(self):
        """测试基本Agent异常"""
        error = AgentError("Agent execution failed")
        assert isinstance(error, QuantAError)

    def test_agent_error_with_details(self):
        """测试带详情的Agent异常"""
        error = AgentError(
            "Agent execution failed",
            agent_name="market_data_agent",
            agent_type="market_data",
        )
        assert error.details["agent_name"] == "market_data_agent"
        assert error.details["agent_type"] == "market_data"


class TestBacktestError:
    """测试回测异常"""

    def test_backtest_error_basic(self):
        """测试基本回测异常"""
        error = BacktestError("Backtest failed")
        assert isinstance(error, QuantAError)

    def test_backtest_error_with_details(self):
        """测试带详情的回测异常"""
        error = BacktestError(
            "Backtest failed",
            symbol="600000",
            timestamp="2024-01-01 10:00:00",
        )
        assert error.details["symbol"] == "600000"
        assert error.details["timestamp"] == "2024-01-01 10:00:00"


class TestRLError:
    """测试RL异常"""

    def test_rl_error_basic(self):
        """测试基本RL异常"""
        error = RLError("RL training failed")
        assert isinstance(error, QuantAError)

    def test_rl_error_with_details(self):
        """测试带详情的RL异常"""
        error = RLError(
            "RL training failed",
            algorithm="ppo",
            env_id="ASharesTrading-v0",
        )
        assert error.details["algorithm"] == "ppo"
        assert error.details["env_id"] == "ASharesTrading-v0"


class TestHandleErrorsDecorator:
    """测试错误处理装饰器"""

    def test_sync_function_no_error(self):
        """测试同步函数无错误情况"""
        @handle_errors(error_type=QuantAError, default_return=None)
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_sync_function_with_error_no_reraise(self):
        """测试同步函数有错误但不重新抛出"""
        @handle_errors(error_type=QuantAError, default_return="default")
        def failing_function():
            raise QuantAError("Test error")

        result = failing_function()
        assert result == "default"

    def test_sync_function_with_error_reraise(self):
        """测试同步函数有错误并重新抛出"""
        @handle_errors(error_type=QuantAError, reraise=True)
        def failing_function():
            raise QuantAError("Test error")

        with pytest.raises(QuantAError) as exc_info:
            failing_function()
        assert str(exc_info.value) == "Test error"

    def test_async_function_no_error(self):
        """测试异步函数无错误情况"""
        @handle_errors(error_type=QuantAError, default_return=None)
        async def successful_async_function():
            return "async_success"

        result = asyncio.run(successful_async_function())
        assert result == "async_success"

    def test_async_function_with_error_no_reraise(self):
        """测试异步函数有错误但不重新抛出"""
        @handle_errors(error_type=QuantAError, default_return="default")
        async def failing_async_function():
            raise QuantAError("Test error")

        result = asyncio.run(failing_async_function())
        assert result == "default"

    def test_async_function_with_error_reraise(self):
        """测试异步函数有错误并重新抛出"""
        @handle_errors(error_type=QuantAError, reraise=True)
        async def failing_async_function():
            raise QuantAError("Test error")

        with pytest.raises(QuantAError):
            asyncio.run(failing_async_function())

    def test_specific_error_type(self):
        """测试特定错误类型捕获"""
        @handle_errors(error_type=DataError, default_return="data_error")
        def mixed_error_function(error_type: str):
            if error_type == "data":
                raise DataError("Data error")
            elif error_type == "agent":
                raise AgentError("Agent error")
            return "success"

        # DataError会被捕获
        result = mixed_error_function("data")
        assert result == "data_error"

        # AgentError不会被捕获（因为指定了DataError）
        with pytest.raises(AgentError):
            mixed_error_function("agent")

    def test_decorator_with_context(self):
        """测试带上下文信息的装饰器"""
        @handle_errors(
            error_type=QuantAError,
            default_return="default",
            context={"module": "test", "version": "1.0"},
        )
        def failing_function():
            raise QuantAError("Test error")

        result = failing_function()
        assert result == "default"


class TestErrorHandlerContextManager:
    """测试错误处理上下文管理器"""

    def test_context_manager_no_error(self):
        """测试无错误的上下文"""
        with ErrorHandler() as handler:
            result = 1 + 1

        assert not handler.has_error()
        assert handler.get_error() is None

    def test_context_manager_with_error_no_reraise(self):
        """测试有错误但不重新抛出"""
        with ErrorHandler(reraise=False) as handler:
            raise QuantAError("Test error")

        assert handler.has_error()
        assert isinstance(handler.get_error(), QuantAError)
        assert str(handler.get_error()) == "Test error"

    def test_context_manager_with_error_reraise(self):
        """测试有错误并重新抛出"""
        with pytest.raises(QuantAError):
            with ErrorHandler(reraise=True) as handler:
                raise QuantAError("Test error")

    def test_context_manager_with_specific_error_type(self):
        """测试特定错误类型"""
        with ErrorHandler(error_type=DataError, reraise=False) as handler:
            raise DataError("Data error")

        assert handler.has_error()
        assert isinstance(handler.get_error(), DataError)


class TestHelperFunctions:
    """测试辅助函数"""

    def test_create_error(self):
        """测试创建错误"""
        error = create_error(
            DataError,
            "Data fetch failed",
            symbol="600000",
            data_source="tushare",
        )
        assert isinstance(error, DataError)
        assert error.details["symbol"] == "600000"
        assert error.details["data_source"] == "tushare"

    def test_format_error_for_api(self):
        """测试API错误格式化"""
        error = QuantAError("Test error", details={"key": "value"})
        api_response = format_error_for_api(error)

        assert api_response["success"] is False
        assert api_response["error"]["error_type"] == "QuantAError"
        assert api_response["error"]["message"] == "Test error"
        assert api_response["error"]["details"] == {"key": "value"}

    def test_log_and_raise(self):
        """测试记录并抛出"""
        with pytest.raises(DataError) as exc_info:
            log_and_raise(
                DataError,
                "Data fetch failed",
                symbol="600000",
                log_level="ERROR",
            )

        # 错误消息包含详情
        assert "Data fetch failed" in str(exc_info.value)
        assert "symbol=600000" in str(exc_info.value)
        assert exc_info.value.details["symbol"] == "600000"


class TestErrorInheritance:
    """测试错误继承关系"""

    def test_all_errors_inherit_from_quanta_error(self):
        """测试所有错误都继承自QuantAError"""
        errors = [
            DataError("test"),
            AgentError("test"),
            BacktestError("test"),
            RLError("test"),
        ]
        for error in errors:
            assert isinstance(error, QuantAError)
            assert isinstance(error, Exception)

    def test_catch_base_error(self):
        """测试可以用基类捕获所有子类错误"""
        caught_errors = []

        def catch_all(error):
            try:
                raise error
            except QuantAError as e:
                caught_errors.append(type(e).__name__)

        catch_all(DataError("data error"))
        catch_all(AgentError("agent error"))
        catch_all(BacktestError("backtest error"))
        catch_all(RLError("rl error"))

        assert "DataError" in caught_errors
        assert "AgentError" in caught_errors
        assert "BacktestError" in caught_errors
        assert "RLError" in caught_errors


class TestErrorDetails:
    """测试错误详情处理"""

    def test_error_details_preservation(self):
        """测试错误详情保持"""
        error = DataError(
            "Data error",
            data_source="tushare",
            symbol="600000",
        )
        error_dict = error.to_dict()
        assert error_dict["details"]["data_source"] == "tushare"
        assert error_dict["details"]["symbol"] == "600000"

    def test_multiple_details(self):
        """测试多个详情字段"""
        error = BacktestError(
            "Backtest error",
            symbol="600000",
            timestamp="2024-01-01",
            details={"additional": "info"},
        )
        assert error.details["symbol"] == "600000"
        assert error.details["timestamp"] == "2024-01-01"
        assert error.details["additional"] == "info"


class TestDecoratorPreservesFunctionMetadata:
    """测试装饰器保留函数元数据"""

    def test_function_name_preserved(self):
        """测试函数名保留"""
        @handle_errors()
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_function_docstring_preserved(self):
        """测试函数文档字符串保留"""
        @handle_errors()
        def my_function():
            """This is my function"""
            pass

        assert my_function.__doc__ == "This is my function"

    def test_async_function_detection(self):
        """测试异步函数检测"""
        @handle_errors()
        async def async_function():
            pass

        import inspect
        assert inspect.iscoroutinefunction(async_function)
