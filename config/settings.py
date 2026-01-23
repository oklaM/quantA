"""
quantA 全局配置文件
A股量化AI交易系统配置
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / ".cache"


@dataclass
class MarketConfig:
    """市场配置"""

    # 交易时间
    MORNING_START = "09:30:00"
    MORNING_END = "11:30:00"
    AFTERNOON_START = "13:00:00"
    AFTERNOON_END = "15:00:00"

    # 涨跌停限制
    MAIN_BOARD_LIMIT = 0.10  # 主板 ±10%
    SME_BOARD_LIMIT = 0.20  # 创业板/科创板 ±20%
    STAR_BOARD_LIMIT = 0.20  # 科创板 ±20%

    # T+1 交易规则
    T_PLUS_ONE = True  # 当天买入次日才能卖出

    # 最小申报单位
    MIN_ORDER_SIZE = 100  # 1手 = 100股


@dataclass
class DatabaseConfig:
    """数据库配置"""

    # 时序数据库选择: "clickhouse" 或 "duckdb"
    TIMESERIES_DB = "duckdb"

    # DuckDB 配置
    DUCKDB_PATH = DATA_DIR / "quant_a.duckdb"

    # ClickHouse 配置（可选）
    CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
    CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "9000"))
    CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
    CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
    CLICKHOUSE_DATABASE = "quant_a"


@dataclass
class LLMConfig:
    """LLM模型配置"""

    # 模型选择
    PROVIDER = "zhipu"  # zhipu, openai, anthropic
    MODEL_NAME = "glm-4-plus"  # GLM-4-Plus
    MODEL_NAME_FAST = "glm-4-flash"  # 快速模型

    # API配置
    API_KEY: str = field(default_factory=lambda: os.getenv("ZHIPUAI_API_KEY", ""))
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

    # 模型参数
    TEMPERATURE = 0.7
    MAX_TOKENS = 4096
    TOP_P = 0.9

    # 向量数据库
    VECTOR_DB = "chromadb"  # chromadb, qdrant
    VECTOR_DB_PATH = DATA_DIR / "vector_db"


@dataclass
class RLConfig:
    """强化学习配置"""

    # 环境
    ENV_NAME = "ASharesTrading-v0"

    # 算法选择: "ppo", "dqn", "a2c"
    ALGORITHM = "ppo"

    # 训练参数
    TOTAL_TIMESTEPS = 1_000_000
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 2048
    N_STEPS = 2048

    # 网络结构
    HIDDEN_DIMS = [256, 256]

    # 奖励函数权重
    REWARD_WEIGHTS = {
        "return": 1.0,
        "transaction_cost": -0.001,
        "drawdown_penalty": -0.5,
        "sharpe_bonus": 0.1,
    }


@dataclass
class ExecutionConfig:
    """执行引擎配置"""

    # 券商接口选择: "xtp", "simulation"
    BROKER = "simulation"

    # XTP 配置
    XTP_CLIENT_ID = int(os.getenv("XTP_CLIENT_ID", "1"))
    XTP_ACCOUNT = os.getenv("XTP_ACCOUNT", "")
    XTP_PASSWORD = os.getenv("XTP_PASSWORD", "")
    XTP_TD_URL = os.getenv("XTP_TD_URL", "")  # 交易服务器
    XTP_MD_URL = os.getenv("XTP_MD_URL", "")  # 行情服务器

    # 风险限制
    MAX_POSITION_RATIO = 0.95  # 最大持仓比例
    MAX_SINGLE_STOCK_RATIO = 0.20  # 单股票最大持仓比例
    MAX_DAILY_LOSS_RATIO = 0.05  # 单日最大亏损比例
    STOP_LOSS_RATIO = 0.05  # 止损比例
    TAKE_PROFIT_RATIO = 0.15  # 止盈比例


@dataclass
class DataConfig:
    """数据源配置"""

    # Tushare配置
    TUSHARE_TOKEN: str = field(default_factory=lambda: os.getenv("TUSHARE_TOKEN", ""))

    # AKShare配置（免费，无需token）
    AKSHARE_ENABLED = True

    # 数据更新频率
    REALTIME_INTERVAL = 60  # 秒
    DAILY_UPDATE_TIME = "18:00:00"


@dataclass
class LoggingConfig:
    """日志配置"""

    LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    FILE_ENABLED = True
    FILE_PATH = LOGS_DIR / "quant_a.log"
    MAX_BYTES = 10 * 1024 * 1024  # 10MB
    BACKUP_COUNT = 5


# 全局配置实例
market = MarketConfig()
database = DatabaseConfig()
llm = LLMConfig()
rl = RLConfig()
execution = ExecutionConfig()
data = DataConfig()
logging = LoggingConfig()


def init_directories():
    """初始化必要的目录"""
    dirs = [DATA_DIR, LOGS_DIR, CACHE_DIR, llm.VECTOR_DB_PATH]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# 启动时初始化目录
init_directories()
