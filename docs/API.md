# quantA API 文档

本文档提供 quantA 系统的 API 接口说明。

## 目录

- [环境配置](#环境配置)
- [回测系统](#回测系统)
- [强化学习](#强化学习)
- [LLM Agents](#llm-agents)
- [监控告警](#监控告警)
- [Rust执行引擎](#rust执行引擎)
- [券商接口](#券商接口)

---

## 环境配置

### `config/settings.py`

全局配置模块，使用 dataclass 管理所有配置项。

```python
from config import settings

# 访问配置
print(settings.market.T_PLUS_ONE)  # A股T+1规则
print(settings.database.DUCKDB_PATH)  # 数据库路径
print(settings.llm.MODEL_NAME)  # LLM模型名称

# 修改配置
settings.llm.TEMPERATURE = 0.5
```

**主要配置类：**

- `MarketConfig`: 市场规则配置
  - `MORNING_START` / `MORNING_END`: 上午交易时间
  - `MAIN_BOARD_LIMIT`: 主板涨跌停限制 (10%)
  - `T_PLUS_ONE`: T+1规则
  - `MIN_ORDER_SIZE`: 最小申报单位 (100股)

- `LLMConfig`: LLM模型配置
  - `PROVIDER`: 模型提供商 ("zhipu")
  - `MODEL_NAME`: 模型名称 ("glm-4-plus")
  - `API_KEY`: API密钥

- `RLConfig`: 强化学习配置
  - `ALGORITHM`: 算法 ("ppo", "dqn", "a2c")
  - `TOTAL_TIMESTEPS`: 总训练步数
  - `LEARNING_RATE`: 学习率

---

## 回测系统

### 事件驱动回测引擎

**核心类：** `backtest.engine.backtest.BacktestEngine`

```python
from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import StrategyBase

# 创建回测引擎
engine = BacktestEngine(
    initial_cash=1000000.0,
    commission_rate=0.0003,
)

# 运行回测
results = engine.run_backtest(
    data=data_df,
    strategy=strategy_instance,
)

# 获取结果
print(results.total_return)
print(results.sharpe_ratio)
print(results.max_drawdown)
```

### 技术指标

**核心类：** `backtest.engine.indicators.TechnicalIndicators`

```python
from backtest.engine.indicators import TechnicalIndicators

indicators = TechnicalIndicators()

# 计算移动平均
df['ma5'] = indicators.sma(df['close'], 5)
df['ema20'] = indicators.ema(df['close'], 20)

# MACD
macd_line, signal_line, histogram = indicators.macd(df['close'])

# RSI
df['rsi'] = indicators.rsi(df['close'])

# 布林带
upper, middle, lower = indicators.bollinger_bands(df['close'])
```

### A股规则引擎

**核心类：** `backtest.engine.a_share_rules.AShareRulesEngine`

```python
from backtest.engine.a_share_rules import AShareRulesEngine

rules = AShareRulesEngine()

# 检查是否可交易
is_tradable = rules.is_trading_time(datetime.now())
can_buy = rules.check_buy_rules("600519.SH", price=100.0, quantity=100)

# 计算手续费
commission = rules.calculate_commission(price=100.0, quantity=100)
stamp_duty = rules.calculate_stamp_duty(price=100.0, quantity=100)
```

---

## 强化学习

### 交易环境

**核心类：** `rl.envs.a_share_trading_env.ASharesTradingEnv`

```python
from rl.envs.a_share_trading_env import ASharesTradingEnv

# 创建环境
env = ASharesTradingEnv(
    data=data_df,
    initial_cash=1000000.0,
    commission_rate=0.0003,
    window_size=60,
)

# 重置环境
obs, info = env.reset()

# 执行动作
action = env.action_space.sample()  # 0=持有, 1=买入, 2=卖出
obs, reward, terminated, truncated, info = env.step(action)
```

**动作空间：** `Discrete(3)` - 0=持有, 1=买入, 2=卖出

**观察空间：** 20维向量，包含：
- 价格特征 (5维)
- 技术指标 (13维): MA(2), RSI(1), MACD(3), 布林带(3), 趋势(1), 收益率(3)
- 账户状态 (2维)

### 训练器

**核心类：** `rl.training.trainer.RLTrainer`

```python
from rl.training.trainer import RLTrainer

# 创建训练器
trainer = RLTrainer(
    env=env,
    algorithm="ppo",
    learning_rate=3e-4,
    tensorboard_log="./logs/",
)

# 构建模型
model = trainer.build_model(
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

# 训练
model = trainer.train(
    total_timesteps=100000,
    save_freq=10000,
)
```

### 奖励函数

**模块：** `rl.rewards.reward_functions`

```python
from rl.rewards.reward_functions import create_reward_function

# 创建奖励函数
reward_fn = create_reward_function("risk_adjusted", profit_weight=1.0)

# 或使用组合奖励函数
from rl.rewards.reward_functions import CompositeReward

reward_fn = CompositeReward([
    (sharpe_reward, 0.7),
    (drawdown_reward, 0.3),
])
```

**可用奖励函数：**
- `simple`: 简单利润奖励
- `sharpe`: 夏普比率奖励
- `risk_adjusted`: 风险调整奖励
- `max_drawdown`: 最大回撤惩罚
- `transaction_aware`: 考虑交易成本
- `asymmetric`: 不对称奖励（损失厌恶）
- `sortino`: Sortino比率奖励
- `calmar`: Calmar比率奖励

### 模型管理

**核心类：** `rl.models.model_manager.ModelManager`

```python
from rl.models.model_manager import ModelManager

# 创建管理器
manager = ModelManager(models_dir="models")

# 保存模型
version = manager.save_model(
    model=trained_model,
    algorithm="ppo",
    metadata={"total_timesteps": 100000},
)

# 加载模型
model = manager.load_model(version.version_id)

# 列出版本
versions = manager.list_versions(algorithm="ppo")

# 导出模型
manager.export_model(version.version_id, "export/")
```

### 模型评估

**核心类：** `rl.evaluation.model_evaluator.ModelEvaluator`

```python
from rl.evaluation.model_evaluator import ModelEvaluator, ModelComparator

# 评估单个模型
evaluator = ModelEvaluator(env, n_episodes=100)
result = evaluator.evaluate(model, "PPO Agent")

# 对比多个模型
comparator = ModelComparator(env, n_episodes=50)
results_df = comparator.compare({
    "PPO": ppo_model,
    "DQN": dqn_model,
})

# 获取最佳模型
best = comparator.get_best_model()
```

---

## LLM Agents

### Agent基类

**核心类：** `agents.base.agent_base.LLMAgent`

```python
from agents.base.agent_base import LLMAgent, MessageType, Message

class CustomAgent(LLMAgent):
    async def process(self, message: str) -> str:
        # 处理消息
        response = await self.llm.ainvoke(message)
        return response
```

### Agent协调器

**核心类：** `agents.base.coordinator.AgentCoordinator`

```python
from agents.base.coordinator import AgentCoordinator

# 创建协调器
coordinator = AgentCoordinator()

# 注册Agent
coordinator.register_agent("market_data", market_data_agent)
coordinator.register_agent("technical", technical_agent)

# 协同执行
result = await coordinator.coordinate(
    query="分析000001.SZ的投资价值",
    agents=["market_data", "technical"],
)
```

### 可用Agent

1. **MarketDataAgent** (`agents.market_data_agent`)
   - 获取实时和历史行情数据
   - 支持多数据源

2. **TechnicalAgent** (`agents.technical_agent`)
   - 技术指标分析
   - 趋势判断

3. **SentimentAgent** (`agents.sentiment_agent`)
   - 市场情绪分析
   - 新闻舆情

4. **StrategyAgent** (`agents.strategy_agent`)
   - 策略生成
   - 交易信号

5. **RiskAgent** (`agents.risk_agent`)
   - 风险评估
   - 仓位控制

---

## 监控告警

### 告警管理

**核心类：** `live.monitoring.alerting.AlertManager`

```python
from live.monitoring.alerting import AlertManager, Alert, AlertSeverity, AlertType

# 创建告警管理器
manager = AlertManager()

# 添加渠道
from live.monitoring.alerting import EmailAlertChannel, WebhookAlertChannel
manager.add_channel(EmailAlertChannel(smtp_config))
manager.add_channel(WebhookAlertChannel(webhook_url))

# 触发告警
alert = Alert(
    alert_id="alert_001",
    alert_type=AlertType.LOSS_LIMIT,
    severity=AlertSeverity.WARNING,
    title="亏损限制告警",
    message="单日亏损超过5%",
)
manager.trigger_alert(alert)

# 启动处理
manager.start()
```

### 监控器

**核心类：** `live.monitoring.monitor.Monitor`

```python
from live.monitoring.monitor import Monitor
from live.monitoring.alerting import create_default_alert_manager

# 创建监控器
alert_manager = create_default_alert_manager()
monitor = Monitor(alert_manager)

# 启动监控
monitor.start(update_interval=5)

# 查看状态
print(monitor.get_summary())

# 获取完整状态
status = monitor.get_status()
```

### Web监控面板

**运行Web面板：**

```bash
streamlit run live/monitoring/web_dashboard.py
```

**功能：**
- 实时系统状态
- 交易状态监控
- 绩效可视化
- 告警查看
- 持仓明细

---

## Rust执行引擎

### Python包装器

**模块：** `rust_engine.python_wrapper`

```python
from rust_engine.python_wrapper import RustOrderManager, RustOrderBook

# 创建订单管理器
order_manager = RustOrderManager()

# 创建订单
order_id = order_manager.create_order(
    symbol="000001.SZ",
    side="buy",
    order_type="limit",
    quantity=1000,
    price=10.5,
)

# 获取订单
order = order_manager.get_order(order_id)

# 取消订单
order_manager.cancel_order(order_id)

# 创建订单簿
order_book = RustOrderBook(symbol="000001.SZ")
depth = order_book.get_depth(depth=5)
```

**编译Rust引擎：**

```bash
cd rust_engine
cargo build --release

# 或使用Python包装器
python -m rust_engine.python_wrapper compile
```

---

## 券商接口

### XTP接口

**核心类：** `live.brokers.xtp_broker.XTPBroker`

```python
from live.brokers.xtp_broker import (
    XTPBroker,
    OrderType,
    OrderSide,
    create_xtp_broker,
)

# 创建接口（模拟模式）
broker = create_xtp_broker(
    account_id="your_account",
    password="your_password",
    simulated=True,
)

# 连接和登录
broker.connect()
broker.login()

# 下单
order_id = broker.place_order(
    symbol="000001.SZ",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    price=10.5,
    quantity=1000,
)

# 查询账户
account = broker.get_account()
print(account.total_asset)

# 查询持仓
positions = broker.get_positions()

# 撤单
broker.cancel_order(order_id)
```

---

## 完整示例

### 回测策略

```python
from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import MovingAverageCrossStrategy
from data.market.sources.akshare_provider import AKShareProvider

# 获取数据
provider = AKShareProvider()
provider.connect()
data = provider.get_daily_bar(
    symbol="000001",
    start_date="20230101",
    end_date="20231231",
    adjust="qfq"
)
provider.disconnect()

# 创建策略
strategy = MovingAverageCrossStrategy("000001", short_window=5, long_window=20)

# 运行回测
engine = BacktestEngine(initial_cash=1000000.0)
results = engine.run_backtest(data, strategy)

# 查看结果
print(f"总收益率: {results.total_return:.2%}")
print(f"夏普比率: {results.sharpe_ratio:.2f}")
print(f"最大回撤: {results.max_drawdown:.2%}")
```

### RL训练

```python
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer

# 准备数据
data = get_market_data()

# 创建环境
env = ASharesTradingEnv(data=data, initial_cash=1000000.0)

# 创建训练器
trainer = RLTrainer(env, algorithm="ppo")

# 训练
model = trainer.train(total_timesteps=100000)

# 评估
evaluator = ModelEvaluator(env, n_episodes=100)
results = evaluator.evaluate(model, "PPO Agent")
```

---

## 错误处理

所有模块使用统一的错误类型：

```python
from backtest.engine.errors import BacktestError
try:
    engine.run_backtest(data, strategy)
except BacktestError as e:
    print(f"回测错误: {e}")
```

---

## 更多信息

- [README.md](../README.md) - 项目概述
- [PROGRESS.md](../PROGRESS.md) - 开发进度
- [docs/](../docs/) - 详细文档

---

**文档版本**: v0.2
**最后更新**: 2026-01-15
