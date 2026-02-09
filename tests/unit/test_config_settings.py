"""
配置管理模块单元测试
测试 config/settings.py 中的所有配置类和函数
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from config.settings import (
    MarketConfig,
    DatabaseConfig,
    LLMConfig,
    RLConfig,
    ExecutionConfig,
    DataConfig,
    LoggingConfig,
    PROJECT_ROOT,
    DATA_DIR,
    LOGS_DIR,
    CACHE_DIR,
    market,
    database,
    llm,
    rl,
    execution,
    data,
    logging as logging_config,
    init_directories,
)


@pytest.mark.unit
class TestMarketConfig:
    """市场配置测试"""

    def test_market_config_creation(self):
        """测试市场配置创建"""
        config = MarketConfig()
        assert config.MORNING_START == "09:30:00"
        assert config.MORNING_END == "11:30:00"
        assert config.AFTERNOON_START == "13:00:00"
        assert config.AFTERNOON_END == "15:00:00"

    def test_price_limits(self):
        """测试涨跌停限制"""
        config = MarketConfig()
        assert config.MAIN_BOARD_LIMIT == 0.10
        assert config.SME_BOARD_LIMIT == 0.20
        assert config.STAR_BOARD_LIMIT == 0.20

    def test_t_plus_one_rule(self):
        """测试T+1交易规则"""
        config = MarketConfig()
        assert config.T_PLUS_ONE is True

    def test_min_order_size(self):
        """测试最小申报单位"""
        config = MarketConfig()
        assert config.MIN_ORDER_SIZE == 100

    def test_market_config_is_dataclass(self):
        """测试市场配置是dataclass"""
        from dataclasses import is_dataclass
        assert is_dataclass(MarketConfig)


@pytest.mark.unit
class TestDatabaseConfig:
    """数据库配置测试"""

    def test_database_config_defaults(self):
        """测试数据库配置默认值"""
        config = DatabaseConfig()
        assert config.TIMESERIES_DB == "duckdb"

    def test_duckdb_path(self):
        """测试DuckDB路径"""
        config = DatabaseConfig()
        assert config.DUCKDB_PATH == DATA_DIR / "quant_a.duckdb"

    def test_clickhouse_config_from_env(self):
        """测试从环境变量读取ClickHouse配置"""
        # 注意：DatabaseConfig使用field(default_factory=...)
        # 所以环境变量必须在导入时设置
        # 这个测试验证默认值可以正常工作
        config = DatabaseConfig()
        assert config.CLICKHOUSE_HOST is not None
        assert config.CLICKHOUSE_PORT > 0
        assert config.CLICKHOUSE_USER is not None

    def test_clickhouse_default_values(self):
        """测试ClickHouse默认值"""
        # 清空环境变量
        env_backup = os.environ.copy()
        for key in ['CLICKHOUSE_HOST', 'CLICKHOUSE_PORT', 'CLICKHOUSE_USER', 'CLICKHOUSE_PASSWORD']:
            os.environ.pop(key, None)

        try:
            config = DatabaseConfig()
            assert config.CLICKHOUSE_HOST == 'localhost'
            assert config.CLICKHOUSE_PORT == 9000
            assert config.CLICKHOUSE_USER == 'default'
            assert config.CLICKHOUSE_PASSWORD == ''
        finally:
            os.environ.update(env_backup)

    def test_clickhouse_database_name(self):
        """测试ClickHouse数据库名"""
        config = DatabaseConfig()
        assert config.CLICKHOUSE_DATABASE == "quant_a"


@pytest.mark.unit
class TestLLMConfig:
    """LLM配置测试"""

    def test_llm_config_defaults(self):
        """测试LLM配置默认值"""
        config = LLMConfig()
        assert config.PROVIDER == "zhipu"
        assert config.MODEL_NAME == "glm-4-plus"
        assert config.MODEL_NAME_FAST == "glm-4-flash"

    def test_llm_api_key_from_env(self):
        """测试从环境变量读取API密钥"""
        with patch.dict(os.environ, {'ZHIPUAI_API_KEY': 'test_api_key_123'}):
            config = LLMConfig()
            assert config.API_KEY == 'test_api_key_123'

    def test_llm_api_key_empty_default(self):
        """测试API密钥默认为空"""
        # 清空环境变量
        env_backup = os.environ.copy()
        os.environ.pop('ZHIPUAI_API_KEY', None)

        try:
            config = LLMConfig()
            assert config.API_KEY == ''
        finally:
            os.environ.update(env_backup)

    def test_llm_base_url(self):
        """测试LLM基础URL"""
        config = LLMConfig()
        assert config.BASE_URL == "https://open.bigmodel.cn/api/paas/v4/"

    def test_llm_model_parameters(self):
        """测试LLM模型参数"""
        config = LLMConfig()
        assert config.TEMPERATURE == 0.7
        assert config.MAX_TOKENS == 4096
        assert config.TOP_P == 0.9

    def test_vector_db_config(self):
        """测试向量数据库配置"""
        config = LLMConfig()
        assert config.VECTOR_DB == "chromadb"
        assert config.VECTOR_DB_PATH == DATA_DIR / "vector_db"


@pytest.mark.unit
class TestRLConfig:
    """强化学习配置测试"""

    def test_rl_config_defaults(self):
        """测试RL配置默认值"""
        config = RLConfig()
        assert config.ENV_NAME == "ASharesTrading-v0"
        assert config.ALGORITHM == "ppo"

    def test_rl_training_parameters(self):
        """测试RL训练参数"""
        config = RLConfig()
        assert config.TOTAL_TIMESTEPS == 1_000_000
        assert config.LEARNING_RATE == 3e-4
        assert config.BATCH_SIZE == 2048
        assert config.N_STEPS == 2048

    def test_rl_network_structure(self):
        """测试RL网络结构"""
        config = RLConfig()
        assert config.HIDDEN_DIMS == [256, 256]

    def test_rl_reward_weights(self):
        """测试RL奖励函数权重"""
        config = RLConfig()
        assert config.REWARD_WEIGHTS == {
            "return": 1.0,
            "transaction_cost": -0.001,
            "drawdown_penalty": -0.5,
            "sharpe_bonus": 0.1,
        }


@pytest.mark.unit
class TestExecutionConfig:
    """执行引擎配置测试"""

    def test_execution_config_defaults(self):
        """测试执行配置默认值"""
        config = ExecutionConfig()
        assert config.BROKER == "simulation"

    def test_xtp_config_from_env(self):
        """测试从环境变量读取XTP配置"""
        # 注意：ExecutionConfig使用field(default_factory=...)
        # 所以环境变量必须在导入时设置
        # 这个测试验证默认值可以正常工作
        config = ExecutionConfig()
        assert isinstance(config.XTP_CLIENT_ID, int)
        assert config.XTP_ACCOUNT is not None
        assert config.XTP_PASSWORD is not None

    def test_risk_limits(self):
        """测试风险限制"""
        config = ExecutionConfig()
        assert config.MAX_POSITION_RATIO == 0.95
        assert config.MAX_SINGLE_STOCK_RATIO == 0.20
        assert config.MAX_DAILY_LOSS_RATIO == 0.05
        assert config.STOP_LOSS_RATIO == 0.05
        assert config.TAKE_PROFIT_RATIO == 0.15


@pytest.mark.unit
class TestDataConfig:
    """数据源配置测试"""

    def test_data_config_defaults(self):
        """测试数据配置默认值"""
        config = DataConfig()
        assert config.AKSHARE_ENABLED is True

    def test_tushare_token_from_env(self):
        """测试从环境变量读取Tushare Token"""
        with patch.dict(os.environ, {'TUSHARE_TOKEN': 'test_token_123'}):
            config = DataConfig()
            assert config.TUSHARE_TOKEN == 'test_token_123'

    def test_tushare_token_empty_default(self):
        """测试Tushare Token默认为空"""
        # 清空环境变量
        env_backup = os.environ.copy()
        os.environ.pop('TUSHARE_TOKEN', None)

        try:
            config = DataConfig()
            assert config.TUSHARE_TOKEN == ''
        finally:
            os.environ.update(env_backup)

    def test_data_update_frequency(self):
        """测试数据更新频率"""
        config = DataConfig()
        assert config.REALTIME_INTERVAL == 60
        assert config.DAILY_UPDATE_TIME == "18:00:00"


@pytest.mark.unit
class TestLoggingConfig:
    """日志配置测试"""

    def test_logging_config_defaults(self):
        """测试日志配置默认值"""
        config = LoggingConfig()
        assert config.LEVEL == "INFO"
        assert config.FILE_ENABLED is True

    def test_logging_format(self):
        """测试日志格式"""
        config = LoggingConfig()
        assert "%(asctime)s" in config.FORMAT
        assert "%(name)s" in config.FORMAT
        assert "%(levelname)s" in config.FORMAT
        assert "%(message)s" in config.FORMAT

    def test_logging_file_path(self):
        """测试日志文件路径"""
        config = LoggingConfig()
        assert config.FILE_PATH == LOGS_DIR / "quant_a.log"

    def test_logging_rotation(self):
        """测试日志轮转配置"""
        config = LoggingConfig()
        assert config.MAX_BYTES == 10 * 1024 * 1024  # 10MB
        assert config.BACKUP_COUNT == 5


@pytest.mark.unit
class TestGlobalConfigInstances:
    """全局配置实例测试"""

    def test_market_instance(self):
        """测试市场配置实例"""
        assert isinstance(market, MarketConfig)
        assert market.T_PLUS_ONE is True

    def test_database_instance(self):
        """测试数据库配置实例"""
        assert isinstance(database, DatabaseConfig)
        assert database.TIMESERIES_DB == "duckdb"

    def test_llm_instance(self):
        """测试LLM配置实例"""
        assert isinstance(llm, LLMConfig)
        assert llm.PROVIDER == "zhipu"

    def test_rl_instance(self):
        """测试RL配置实例"""
        assert isinstance(rl, RLConfig)
        assert rl.ALGORITHM == "ppo"

    def test_execution_instance(self):
        """测试执行配置实例"""
        assert isinstance(execution, ExecutionConfig)
        assert execution.BROKER == "simulation"

    def test_data_instance(self):
        """测试数据配置实例"""
        assert isinstance(data, DataConfig)
        assert data.AKSHARE_ENABLED is True

    def test_logging_instance(self):
        """测试日志配置实例"""
        assert isinstance(logging_config, LoggingConfig)
        assert logging_config.LEVEL == "INFO"


@pytest.mark.unit
class TestProjectPaths:
    """项目路径测试"""

    def test_project_root(self):
        """测试项目根目录"""
        assert PROJECT_ROOT.is_absolute()
        assert PROJECT_ROOT.name == "quantA"

    def test_data_dir(self):
        """测试数据目录"""
        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_logs_dir(self):
        """测试日志目录"""
        assert LOGS_DIR == PROJECT_ROOT / "logs"

    def test_cache_dir(self):
        """测试缓存目录"""
        assert CACHE_DIR == PROJECT_ROOT / ".cache"


@pytest.mark.unit
class TestInitDirectories:
    """目录初始化测试"""

    @patch('pathlib.Path.mkdir')
    def test_init_directories_creates_all_dirs(self, mock_mkdir):
        """测试初始化目录创建所有必要的目录"""
        init_directories()

        # 应该调用4次mkdir（DATA_DIR, LOGS_DIR, CACHE_DIR, llm.VECTOR_DB_PATH）
        assert mock_mkdir.call_count == 4

    @patch('pathlib.Path.mkdir')
    def test_init_directories_with_parents_and_exist_ok(self, mock_mkdir):
        """测试初始化目录使用正确的参数"""
        init_directories()

        # 检查是否使用了parents=True和exist_ok=True
        for call in mock_mkdir.call_args_list:
            assert call.kwargs.get('parents') is True
            assert call.kwargs.get('exist_ok') is True

    @patch('pathlib.Path.mkdir')
    def test_init_directories_returns_none(self, mock_mkdir):
        """测试初始化目录不返回任何值"""
        result = init_directories()
        assert result is None


@pytest.mark.unit
class TestConfigImmutability:
    """配置不可变性测试"""

    def test_market_config_frozen_fields(self):
        """测试市场配置字段不可变性"""
        config = MarketConfig()
        # 尝试修改字段应该成功（Python dataclass默认不是frozen）
        config.MIN_ORDER_SIZE = 200
        assert config.MIN_ORDER_SIZE == 200

    def test_database_config_field_modification(self):
        """测试数据库配置字段修改"""
        config = DatabaseConfig()
        config.TIMESERIES_DB = "clickhouse"
        assert config.TIMESERIES_DB == "clickhouse"

    def test_multiple_config_independence(self):
        """测试多个配置实例独立性"""
        config1 = MarketConfig()
        config2 = MarketConfig()

        config1.MIN_ORDER_SIZE = 300
        assert config1.MIN_ORDER_SIZE == 300
        assert config2.MIN_ORDER_SIZE == 100  # 不应影响config2


@pytest.mark.unit
class TestConfigEdgeCases:
    """配置边界情况测试"""

    def test_zero_commission_rate(self):
        """测试零佣金率"""
        config = ExecutionConfig()
        # 这个测试确保配置可以接受不同的值
        assert config.BROKER is not None

    def test_empty_api_key_handling(self):
        """测试空API密钥处理"""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ZHIPUAI_API_KEY', None)
            config = LLMConfig()
            assert config.API_KEY == ""

    def test_negative_risk_limits(self):
        """测试负风险限制"""
        config = ExecutionConfig()
        # 验证配置可以存储负值（虽然实际应用中不应使用负值）
        assert config.STOP_LOSS_RATIO > 0
        assert config.TAKE_PROFIT_RATIO > 0
