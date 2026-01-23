"""
策略参数配置
定义各种策略的参数和阈值
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TechnicalIndicatorsConfig:
    """技术指标配置"""

    # 移动平均线
    MA_PERIODS = [5, 10, 20, 60, 120, 250]

    # MACD
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # RSI
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30

    # 布林带
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2

    # KDJ
    KDJ_N = 9
    KDJ_M1 = 3
    KDJ_M2 = 3

    # ATR（波动率）
    ATR_PERIOD = 14


@dataclass
class LLMAgentConfig:
    """LLM Agent配置"""

    # 市场数据Agent
    MARKET_DATA_LOOKBACK_DAYS = 30  # 查看过去30天数据

    # 技术分析Agent
    TECHNICAL_INDICATORS = ["MA", "MACD", "RSI", "BOLL", "KDJ", "ATR", "VOLUME"]

    # 情绪分析Agent
    SENTIMENT_NEWS_COUNT = 20  # 分析最近20条新闻
    SENTIMENT_SOCIAL_COUNT = 50  # 分析最近50条社交媒体

    # 策略生成Agent
    STRATEGY_CONFIDENCE_THRESHOLD = 0.7  # 信号置信度阈值
    STRATEGY_MAX_POSITIONS = 10  # 最大持仓数

    # 风控Agent
    RISK_MAX_POSITION_RATIO = 0.2  # 单股最大仓位
    RISK_STOP_LOSS = -0.05  # 止损-5%
    RISK_TAKE_PROFIT = 0.15  # 止盈+15%


@dataclass
class RLStrategyConfig:
    """强化学习策略配置"""

    # PPO参数
    PPO_N_STEPS = 2048
    PPO_BATCH_SIZE = 64
    PPO_N_EPOCHS = 10
    PPO_GAMMA = 0.99
    PPO_GAE_LAMBDA = 0.95
    PPO_CLIP_RANGE = 0.2
    PPO_ENT_COEF = 0.01
    PPO_VF_COEF = 0.5

    # DQN参数
    DQN_BUFFER_SIZE = 100000
    DQN_LEARNING_START = 10000
    DQN_TARGET_UPDATE_FREQ = 1000
    DQN_TRAIN_FREQ = 4
    DQN_GAMMA = 0.99
    DQN_TAU = 0.005

    # 网络结构
    NETWORK_HIDDEN_DIMS = [256, 256]
    NETWORK_ACTIVATION = "relu"  # relu, tanh, gelu

    # 特征工程
    FEATURE_WINDOW_SIZE = 60  # 使用过去60根K线
    FEATURE_NORMALIZATION = "rank"  # rank, minmax, standard

    # 奖励函数
    REWARD_SHARPE_WINDOW = 20  # 夏普比率窗口
    REWARD_DRAWDOWN_PENALTY = 0.5


@dataclass
class BacktestConfig:
    """回测配置"""

    # 时间范围
    START_DATE = "2023-01-01"
    END_DATE = "2024-12-31"

    # 初始资金
    INITIAL_CAPITAL = 1_000_000  # 100万

    # 交易成本
    COMMISSION_RATE = 0.0003  # 万分之三佣金
    STAMP_DUTY_RATE = 0.001  # 千分之一印花税（仅卖出）
    MIN_COMMISSION = 5  # 最低佣金5元

    # 滑点
    SLIPPAGE_MODEL = "linear"  # linear, percentage, none
    SLIPPAGE_RATE = 0.0001  # 万分之一滑点

    # 冲击成本
    IMPACT_COST_ENABLED = True
    IMPACT_COST_FACTOR = 0.1

    # A股特殊规则
    T_PLUS_ONE = True  # T+1规则
    LIMIT_UP_HANDLING = "reject"  # reject, fill, wait
    LIMIT_DOWN_HANDLING = "freeze"  # freeze, stop_trading

    # 数据频率
    DATA_FREQUENCY = "1min"  # 1min, 5min, 1d


@dataclass
class TradingConfig:
    """实盘交易配置"""

    # 交易时段
    PREMARKET_START = "09:15:00"
    PREMARKET_END = "09:25:00"
    MORNING_START = "09:30:00"
    MORNING_END = "11:30:00"
    AFTERNOON_START = "13:00:00"
    AFTERNOON_END = "15:00:00"

    # 订单类型
    ORDER_TYPE = "limit"  # limit, market, stop
    ORDER_VALIDITY = "DAY"  # DAY, GTC, FAK, FOK

    # 订单拆分（大单）
    ALGO_ORDER_ENABLED = True
    MAX_ORDER_RATIO = 0.10  # 单笔订单不超过日成交10%
    ALGO_TYPE = "TWAP"  # TWAP, VWAP

    # 订单超时
    ORDER_TIMEOUT_SECONDS = 300  # 5分钟未成交撤单

    # 风控熔断
    CIRCUIT_BREAKER_ENABLED = True
    MAX_DAILY_LOSS_RATIO = 0.05  # 单日亏损5%熔断


# 预设策略配置
STRATEGY_CONFIGS: Dict[str, Dict] = {
    # LLM Agent策略
    "llm_agent": {
        "type": "llm_agent",
        "enabled_agents": ["market_data", "technical", "sentiment", "strategy", "risk"],
        "rebalance_interval": "1d",  # 每日调仓
        "max_positions": 10,
        "position_sizing": "equal_weight",  # equal_weight, kelly, risk_parity
    },
    # 强化学习策略
    "rl_ppo": {
        "type": "rl",
        "algorithm": "ppo",
        "model_path": "models/rl/ppo/",
        "rebalance_interval": "5min",  # 5分钟调仓
        "max_positions": 5,
    },
    # 传统量化策略
    "momentum": {
        "type": "traditional",
        "method": "momentum",
        "lookback_period": 20,
        "rebalance_interval": "1d",
    },
    "mean_reversion": {
        "type": "traditional",
        "method": "mean_reversion",
        "lookback_period": 20,
        "zscore_threshold": 2,
        "rebalance_interval": "1d",
    },
}


# 配置实例
technical = TechnicalIndicatorsConfig()
llm_agent = LLMAgentConfig()
rl_strategy = RLStrategyConfig()
backtest = BacktestConfig()
trading = TradingConfig()


__all__ = [
    "TechnicalIndicatorsConfig",
    "LLMAgentConfig",
    "RLStrategyConfig",
    "BacktestConfig",
    "TradingConfig",
    "STRATEGY_CONFIGS",
    "technical",
    "llm_agent",
    "rl_strategy",
    "backtest",
    "trading",
]
