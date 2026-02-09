# quantA 开发指南

> A股量化AI交易系统开发者指南

本文档面向quantA项目的开发者，提供详细的开发环境配置、架构设计、开发流程、测试策略和最佳实践。

## 目录

- [开发环境配置](#开发环境配置)
- [项目架构](#项目架构)
- [核心模块设计模式](#核心模块设计模式)
- [开发工作流](#开发工作流)
- [测试策略](#测试策略)
- [性能优化技巧](#性能优化技巧)
- [调试技巧](#调试技巧)
- [发布流程](#发布流程)
- [常见问题](#常见问题)

---

## 开发环境配置

### 系统要求

- **操作系统**: Linux/macOS (推荐), Windows (WSL2)
- **Python**: 3.10+ (推荐3.11)
- **Rust**: 1.70+ (可选，用于Rust执行引擎)
- **内存**: 最少8GB，推荐16GB+
- **存储**: 最少10GB可用空间

### 环境搭建步骤

#### 1. 基础环境安装

```bash
# macOS (使用Homebrew)
brew install python@3.11 rust postgresql redis

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv cargo postgresql redis-server

# 验证安装
python --version  # 应显示 3.10+
rustc --version   # 应显示 1.70+
```

#### 2. Python虚拟环境配置

```bash
# 进入项目目录
cd /path/to/quantA

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 升级pip
pip install --upgrade pip setuptools wheel
```

#### 3. 依赖安装

```bash
# 安装核心依赖
pip install -r requirements.txt

# 安装开发依赖
make install-dev
# 或
pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy

# 验证安装
python -c "import gymnasium; print(gymnasium.__version__)"
python -c "import stable_baselines3; print(stable_baselines3.__version__)"
```

#### 4. 环境变量配置

创建 `.env` 文件：

```bash
# 复制模板
cp .env.example .env

# 编辑配置
vim .env
```

必需的环境变量：

```bash
# LLM API (智谱AI)
ZHIPUAI_API_KEY=your_zhipuai_api_key_here

# 数据源API (Tushare)
TUSHARE_TOKEN=your_tushare_token_here

# 可选: 华泰XTP实盘接口
XTP_CLIENT_ID=1
XTP_ACCOUNT=your_xtp_account
XTP_PASSWORD=your_xtp_password
XTP_TD_URL=your_trading_server_url
XTP_MD_URL=your_market_data_server_url

# 可选: ClickHouse数据库
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DATABASE=quant_a
```

#### 5. 数据库初始化

```bash
# 初始化DuckDB (默认)
make db-init
# 或
python -m data.init_db

# 备份数据库
make db-backup
```

#### 6. Rust执行引擎 (可选)

```bash
# 进入Rust引擎目录
cd rust_engine

# 构建Rust库
cargo build --release

# 返回项目根目录
cd ..

# 测试Rust集成
python -c "from rust_engine import RustEngine; print('Rust engine loaded successfully')"
```

### 开发工具配置

#### VS Code配置

创建 `.vscode/settings.json`：

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-v", "--strict-markers"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true
  }
}
```

推荐的VS Code扩展：

- Python (Microsoft)
- Pylance
- Python Test Explorer
- Rust Analyzer (用于Rust开发)
- Even Better TOML
- Code Spell Checker

#### PyCharm配置

1. **Project Interpreter**: 设置为项目venv
2. **Code Style**: Black + Isort
3. **Inspections**: 启用Pylint和MyPy
4. **Tools → External Tools**: 添加pytest、black等工具

### Git配置

```bash
# 配置用户信息
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 设置默认分支
git config init.defaultBranch master

# 配置.line-of-endings (跨平台开发)
git config core.autocrlf input  # Linux/macOS
# git config core.autocrlf true  # Windows
```

创建 `.gitignore` (如果不存在)：

```python
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# 测试
.pytest_cache/
.coverage
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 数据和日志
data/*.csv
data/*.duckdb
logs/*.log
*.log

# 环境变量
.env
.env.local

# 模型文件
models/
*.pkl
*.zip
*.h5

# Rust
rust_engine/target/
```

---

## 项目架构

### 系统分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     决策层 (Decision Layer)                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ LLM Agents  │  │    RL       │  │   传统策略           │  │
│  │  (GLM-4)    │  │  (PPO/DQN)  │  │  (技术指标)          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   策略层 (Strategy Layer)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 信号生成    │  │ 组合构建    │  │   风险控制          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   执行层 (Execution Layer)                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 订单管理    │  │  持仓管理   │  │  风险限制           │  │
│  │  (Rust)     │  │  (Rust)     │  │   (Rust)            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   数据层 (Data Layer)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ 数据源      │  │  时序DB     │  │   向量DB            │  │
│  │(AKShare)    │  │(DuckDB)     │  │  (ChromaDB)         │  │
│  │(Tushare)    │  │(ClickHouse) │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 目录结构详解

```
quantA/
├── agents/                    # LLM Agent系统
│   ├── base/                 # Agent基类和协调器
│   │   ├── agent_base.py    # Agent基类
│   │   ├── coordinator.py   # Agent协调器
│   │   ├── glm4_integration.py  # GLM-4集成
│   │   └── langgraph_integration.py  # LangGraph集成
│   ├── market_data_agent/    # 市场数据Agent
│   ├── technical_agent/      # 技术分析Agent
│   ├── sentiment_agent/      # 情绪分析Agent
│   ├── strategy_agent/       # 策略生成Agent
│   └── risk_agent/          # 风险管理Agent
│
├── backtest/                 # 回测引擎
│   ├── engine/              # 核心引擎
│   │   ├── backtest.py     # 主回测引擎
│   │   ├── indicators.py   # 技术指标计算
│   │   ├── a_share_rules.py  # A股交易规则
│   │   ├── event_engine.py # 事件驱动引擎
│   │   ├── portfolio.py    # 投资组合管理
│   │   └── execution.py    # 订单执行
│   ├── metrics/             # 性能指标和报告
│   ├── optimization/        # 参数优化
│   └── portfolio/           # 组合回测
│
├── rl/                      # 强化学习框架
│   ├── envs/               # 交易环境
│   │   └── a_share_trading_env.py
│   ├── training/           # 模型训练
│   │   └── trainer.py
│   ├── rewards/            # 奖励函数
│   ├── optimization/       # 超参数优化
│   │   └── hyperparameter_tuning.py
│   └── evaluation/         # 模型评估
│       └── model_evaluator.py
│
├── data/                    # 数据层
│   └── market/
│       ├── sources/        # 数据源
│       │   ├── tushare_provider.py
│       │   └── akshare_provider.py
│       ├── storage/        # 时序数据库
│       └── data_manager.py # 数据管理器
│
├── live/                    # 实盘交易 (计划中)
│   ├── brokers/            # 券商接口
│   └── monitoring/         # 监控告警
│
├── rust_engine/            # Rust执行引擎
│   ├── src/
│   │   ├── order.rs       # 订单管理
│   │   ├── portfolio.rs   # 持仓管理
│   │   └── execution.rs   # 执行逻辑
│   ├── Cargo.toml
│   └── python_wrapper.py   # Python绑定
│
├── config/                 # 配置文件
│   ├── settings.py        # 全局配置
│   ├── symbols.py         # 股票池
│   └── strategies.py      # 策略参数
│
├── utils/                  # 工具函数
│   ├── logging.py         # 日志工具
│   ├── time_utils.py      # 时间工具
│   └── helpers.py         # 辅助函数
│
├── trading/                # 交易模块
│   ├── __init__.py
│   └── risk.py            # 风险控制
│
├── tests/                  # 测试套件
│   ├── unit/              # 单元测试
│   ├── integration/       # 集成测试
│   ├── backtest/          # 回测测试
│   ├── agents/            # Agent测试
│   ├── rl/                # RL测试
│   └── conftest.py        # pytest配置
│
├── examples/               # 示例代码
│   ├── backtest_example.py
│   ├── rl_training_guide.py
│   └── agent_example.py
│
├── docs/                   # 文档
├── CLAUDE.md              # Claude Code指南
├── DEVELOPMENT.md         # 本文档
├── README.md              # 项目说明
├── Makefile               # 构建脚本
├── pytest.ini             # pytest配置
└── requirements.txt       # Python依赖
```

### 核心组件交互图

```
┌──────────────────────────────────────────────────────────┐
│                    Agent Coordinator                      │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      │
│  │Market│→│Tech  │→│Senti │→│Strat │→│ Risk │      │
│  │ Data │  │Agent │  │ment  │  |egy   │  │Agent │      │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                   RL Training Pipeline                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Environment│→│ PPO/DQN │→│ Model    │→│Evaluator│ │
│  │  (Gym)    │  │ Trainer │  │ Manager  │  │         │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                   Backtest Engine                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐      │
│  │ Data │→│Event │→│ Port │→│ Exec │→│ Rules│      │
│  │Handler│  │Engine│  │ folio│  │ utor │  │Engine│      │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│                   Rust Execution Engine                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │   Order  │→│ Portfolio │→│ Execution │               │
│  │ Manager  │  │ Manager   │  │ Engine   │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└──────────────────────────────────────────────────────────┘
```

---

## 核心模块设计模式

### 1. Agent系统设计模式

#### Agent基类模式

所有Agent继承自统一的基类，确保一致性：

```python
from agents.base.agent_base import Agent, Message, MessageType

class CustomAgent(Agent):
    """自定义Agent示例"""

    def __init__(self, name: str):
        super().__init__(name)
        # 初始化Agent状态

    async def process(self, message: Message) -> Optional[Message]:
        """处理接收到的消息"""
        try:
            # 1. 解析消息
            content = message.content

            # 2. 执行业务逻辑
            result = self._analyze(content)

            # 3. 构造响应
            response = Message(
                type=MessageType.RESPONSE,
                sender=self.name,
                receiver=message.sender,
                content={"result": result},
                reply_to=message.message_id
            )

            return response

        except Exception as e:
            logger.error(f"Agent处理失败: {e}")
            # 返回错误消息
            return Message(
                type=MessageType.ERROR,
                sender=self.name,
                receiver=message.sender,
                content={"error": str(e)}
            )

    def _analyze(self, data: dict) -> dict:
        """核心分析逻辑"""
        # 实现具体的分析逻辑
        pass
```

#### Agent协调模式

使用协调器管理多Agent协作：

```python
from agents.base.coordinator import AgentCoordinator, Workflow

async def trading_workflow():
    """交易工作流示例"""

    # 1. 创建协调器
    coordinator = AgentCoordinator()

    # 2. 注册Agent
    coordinator.register_agent(MarketDataAgent("market_data"))
    coordinator.register_agent(TechnicalAgent("technical"))
    coordinator.register_agent(SentimentAgent("sentiment"))
    coordinator.register_agent(StrategyAgent("strategy"))
    coordinator.register_agent(RiskAgent("risk"))

    # 3. 定义工作流
    workflow = Workflow(
        name="daily_analysis",
        description="每日市场分析工作流"
    )

    # 4. 添加工作流步骤
    async def step1_fetch_market_data(coordinator):
        agent = coordinator.get_agent("market_data")
        message = Message(
            type=MessageType.REQUEST,
            sender="coordinator",
            receiver="market_data",
            content={"action": "fetch_daily_data"}
        )
        return await agent.process(message)

    workflow.add_step(step1_fetch_market_data)
    # 添加更多步骤...

    # 5. 执行工作流
    results = await coordinator.run_workflow(workflow)

    return results
```

#### LLM调用模式

统一的LLM调用接口：

```python
from agents.base.glm4_integration import GLM4Client

async def llm_analysis_example():
    """LLM分析示例"""

    # 1. 创建LLM客户端
    client = GLM4Client(
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        model="glm-4-plus"
    )

    # 2. 构造提示词
    prompt = """
    分析以下市场数据，给出交易建议：

    技术指标：
    - RSI: 65.5
    - MACD: 金叉
    - 均线: 多头排列

    请以JSON格式返回：
    {
        "action": "buy/sell/hold",
        "confidence": 0.0-1.0,
        "reason": "原因说明"
    }
    """

    # 3. 调用LLM
    response = await client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    # 4. 解析响应
    result = json.loads(response.content)
    return result
```

### 2. 回测引擎设计模式

#### 策略基类模式

所有策略继承自统一基类：

```python
from backtest.engine.strategy import Strategy

class MyCustomStrategy(Strategy):
    """自定义策略示例"""

    def __init__(self, symbols: List[str], params: dict = None):
        super().__init__(symbols)
        self.params = params or {}

    def on_bar(self, event: BarEvent):
        """K线数据回调"""
        symbol = event.symbol
        bar = event.bar

        # 1. 获取历史数据
        history = self.data_handler.get_history(
            symbol,
            window=60
        )

        # 2. 计算技术指标
        indicators = self.calculate_indicators(history)

        # 3. 生成交易信号
        signal = self.generate_signal(indicators)

        # 4. 执行交易
        if signal == "buy":
            self.place_buy_order(symbol, 100)
        elif signal == "sell":
            self.place_sell_order(symbol, 100)

    def calculate_indicators(self, history: pd.DataFrame) -> dict:
        """计算技术指标"""
        from backtest.engine.indicators import TechnicalIndicators

        ti = TechnicalIndicators()
        return {
            "ma5": ti.calculate_ma(history, 5),
            "ma20": ti.calculate_ma(history, 20),
            "rsi": ti.calculate_rsi(history, 14),
        }

    def generate_signal(self, indicators: dict) -> str:
        """生成交易信号"""
        if indicators["ma5"] > indicators["ma20"]:
            if indicators["rsi"] < 70:
                return "buy"
        return "hold"
```

#### 事件驱动模式

回测引擎使用事件驱动架构：

```python
from backtest.engine.event_engine import EventQueue, BarEvent, OrderEvent

# 事件流转流程
# 1. 数据事件 (BarEvent)
bar_event = BarEvent(
    symbol="600519.SH",
    datetime=datetime.now(),
    bar={"open": 1850.0, "high": 1870.0, "low": 1840.0, "close": 1860.0}
)

# 2. 策略处理事件，生成订单事件 (OrderEvent)
order_event = OrderEvent(
    symbol="600519.SH",
    direction="BUY",
    quantity=100,
    price=1860.0
)

# 3. 执行引擎处理订单
# 4. 更新投资组合
```

### 3. 强化学习设计模式

#### 环境封装模式

符合Gymnasium标准的交易环境：

```python
import gymnasium as gym
from rl.envs.a_share_trading_env import ASharesTradingEnv

# 创建环境
env = ASharesTradingEnv(
    data=price_data,
    initial_cash=1_000_000,
    commission_rate=0.0003,
    window_size=60
)

# 环境交互循环
obs, info = env.reset(seed=42)

for _ in range(1000):
    # 1. 获取动作 (从策略或随机)
    action = env.action_space.sample()  # 0=hold, 1=buy, 2=sell

    # 2. 执行动作
    obs, reward, terminated, truncated, info = env.step(action)

    # 3. 记录日志
    if info.get("trade"):
        print(f"交易: {info['trade']}, 奖励: {reward:.2f}")

    # 4. 检查是否结束
    if terminated or truncated:
        break

# 关闭环境
env.close()
```

#### 训练器模式

统一的模型训练接口：

```python
from rl.training.trainer import RLTrainer
from stable_baselines3 import PPO

# 创建训练器
trainer = RLTrainer(
    algorithm="ppo",
    env=env,
    log_dir="logs/rl_training"
)

# 配置训练参数
trainer.set_params({
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
})

# 训练模型
model = trainer.train(
    total_timesteps=100_000,
    eval_freq=5000,
    save_freq=10000
)

# 保存模型
trainer.save_model("models/ppo_a_shares")
```

#### 奖励函数模式

可组合的奖励函数系统：

```python
from rl.rewards.reward_functions import (
    ReturnReward,
    TransactionCostReward,
    DrawdownPenaltyReward,
    SharpeRatioBonusReward
)

# 组合多个奖励函数
reward_config = {
    "return": {"weight": 1.0},
    "transaction_cost": {"weight": -0.001},
    "drawdown_penalty": {"weight": -0.5},
    "sharpe_bonus": {"weight": 0.1},
}

# 在环境中使用
env = ASharesTradingEnv(
    data=price_data,
    reward_config=reward_config
)
```

### 4. 数据层设计模式

#### 数据提供者模式

统一的数据源接口：

```python
from data.market.sources.tushare_provider import TushareProvider
from data.market.sources.akshare_provider import AKShareProvider

# 使用Tushare
tushare = TushareProvider(token=os.getenv("TUSHARE_TOKEN"))
data = tushare.get_daily_data(
    symbol="600519.SH",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# 使用AKShare (免费)
akshare = AKShareProvider()
data = akshare.get_daily_data(
    symbol="600519",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

#### 数据管理器模式

```python
from data.market.data_manager import DataManager

# 创建数据管理器
dm = DataManager(
    storage_type="duckdb",  # 或 "clickhouse"
    db_path="data/quant_a.duckdb"
)

# 更新数据
dm.update_daily_data(symbols=["600519.SH", "000858.SZ"])

# 查询数据
data = dm.query_data(
    symbol="600519.SH",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# 批量更新
dm.batch_update(symbols=["600519.SH", "000858.SZ"])
```

---

## 开发工作流

### 功能开发流程

#### 1. 需求分析

在开始开发前，明确需求：

- **功能描述**: 要实现什么功能？
- **影响范围**: 涉及哪些模块？
- **依赖关系**: 是否依赖其他功能？
- **测试策略**: 如何测试这个功能？

#### 2. 分支管理

```bash
# 创建功能分支
git checkout -b feature/your-feature-name

# 确保基于最新的master
git fetch origin
git rebase origin/master
```

#### 3. 开发步骤

```bash
# 1. 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 2. 创建新文件或修改现有文件
# 例如: touch agents/my_new_agent/agent.py

# 3. 编写代码
vim agents/my_new_agent/agent.py

# 4. 编写测试
vim tests/agents/test_my_new_agent.py

# 5. 运行测试
pytest tests/agents/test_my_new_agent.py -v

# 6. 代码格式化
make format

# 7. 代码检查
make lint
```

#### 4. 提交代码

```bash
# 查看变更
git status
git diff

# 暂存文件
git add agents/my_new_agent/agent.py
git add tests/agents/test_my_new_agent.py

# 提交
git commit -m "feat: add new agent for XYZ analysis

- Implement MyNewAgent class
- Add unit tests
- Update documentation"

# 推送到远程
git push origin feature/your-feature-name
```

#### 5. 代码审查

创建Pull Request后，确保：

- [ ] 代码通过所有测试
- [ ] 代码覆盖率达标 (70%+)
- [ ] 代码通过格式检查
- [ ] 代码通过类型检查 (mypy)
- [ ] 文档已更新
- [ ] 示例代码已添加

#### 6. 合并

```bash
# 合并到master (使用merge或rebase)
git checkout master
git merge feature/your-feature-name
# 或
git rebase feature/your-feature-name

# 推送
git push origin master

# 删除功能分支
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

### Bug修复流程

#### 1. 报告Bug

使用GitHub Issues报告Bug，包含：

- **Bug描述**: 清晰的问题描述
- **复现步骤**: 如何重现问题
- **期望行为**: 期望的正确行为
- **实际行为**: 实际的错误行为
- **环境信息**: Python版本、OS版本等
- **日志**: 相关的错误日志

#### 2. 修复Bug

```bash
# 创建Bug修复分支
git checkout -b fix/bug-description

# 定位问题
# 使用调试技巧 (见下文)

# 编写测试用例复现Bug
vim tests/test_bug_fix.py

# 修复代码
vim path/to/file.py

# 验证修复
pytest tests/test_bug_fix.py -v

# 提交修复
git commit -m "fix: resolve XYZ issue

- Fix bug in ABC module
- Add regression test
- Closes #123"
```

### 代码审查标准

#### 代码质量检查项

1. **代码风格**
   - [ ] 遵循PEP 8规范
   - [ ] 使用有意义的变量名
   - [ ] 函数长度适中 (< 50行)
   - [ ] 类职责单一

2. **文档**
   - [ ] 函数有docstring
   - [ ] 复杂逻辑有注释
   - [ ] 公开API有类型提示

3. **测试**
   - [ ] 单元测试覆盖率 >= 80%
   - [ ] 边界条件有测试
   - [ ] 异常情况有测试

4. **性能**
   - [ ] 无明显的性能问题
   - [ ] 无内存泄漏
   - [ ] 大数据集处理优化

#### 代码审查清单

```python
# 好的代码示例

from typing import List, Optional
import pandas as pd
from utils.logging import get_logger

logger = get_logger(__name__)

def calculate_returns(
    prices: pd.Series,
    period: int = 1,
    method: str = "simple"
) -> pd.Series:
    """
    计算收益率

    Args:
        prices: 价格序列
        period: 周期
        method: 方法 ("simple" or "log")

    Returns:
        收益率序列

    Raises:
        ValueError: 如果方法不支持
    """
    if method not in ["simple", "log"]:
        raise ValueError(f"不支持的方法: {method}")

    if method == "simple":
        return prices.pct_change(period)
    else:
        return np.log(prices / prices.shift(period))
```

---

## 测试策略

### 测试金字塔

```
            /\
           /  \
          / E2E \         ← 端到端测试 (少量)
         /──────\
        /        \
       /Integration\      ← 集成测试 (中等)
      /────────────\
     /              \
    /    Unit Tests   \   ← 单元测试 (大量)
   /──────────────────\
```

### 单元测试

#### 测试组织

```python
# tests/agents/test_my_agent.py

import pytest
from agents.my_agent import MyAgent
from agents.base.agent_base import Message

class TestMyAgent:
    """MyAgent单元测试"""

    @pytest.fixture
    def agent(self):
        """创建Agent实例"""
        return MyAgent("test_agent")

    @pytest.fixture
    def sample_message(self):
        """创建测试消息"""
        return Message(
            type="request",
            sender="test",
            receiver="test_agent",
            content={"data": "test"}
        )

    @pytest.mark.asyncio
    async def test_process_message(self, agent, sample_message):
        """测试消息处理"""
        response = await agent.process(sample_message)

        assert response is not None
        assert response.type == "response"
        assert "result" in response.content

    def test_error_handling(self, agent):
        """测试错误处理"""
        with pytest.raises(ValueError):
            agent.some_method("invalid_input")

    @pytest.mark.parametrize("input,expected", [
        ("data1", "result1"),
        ("data2", "result2"),
        ("data3", "result3"),
    ])
    def test_parameterized(self, agent, input, expected):
        """参数化测试"""
        result = agent.process_data(input)
        assert result == expected
```

#### 测试最佳实践

```python
# 1. 使用pytest fixtures管理测试数据
@pytest.fixture
def sample_data():
    """加载测试数据"""
    return pd.read_csv("tests/fixtures/sample_data.csv")

# 2. 使用mock隔离外部依赖
from unittest.mock import Mock, patch

def test_with_mock():
    """使用mock测试"""
    mock_client = Mock()
    mock_client.get_data.return_value = {"status": "ok"}

    with patch("agents.my_agent.APIClient", return_value=mock_client):
        agent = MyAgent()
        result = agent.fetch_data()
        assert result == {"status": "ok"}

# 3. 测试异常情况
def test_exception_handling():
    """测试异常处理"""
    agent = MyAgent()

    with pytest.raises(ValueError, match="无效参数"):
        agent.process_data(None)

# 4. 使用临时文件和目录
import tempfile

def test_with_temp_file():
    """使用临时文件测试"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test data")
        temp_path = f.name

    try:
        result = process_file(temp_path)
        assert result is not None
    finally:
        os.unlink(temp_path)
```

### 集成测试

#### 回测集成测试

```python
# tests/integration/test_backtest_integration.py

import pytest
from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategies import BuyAndHoldStrategy

@pytest.mark.integration
def test_full_backtest_workflow():
    """完整回测流程测试"""

    # 1. 准备数据
    data = generate_test_data()

    # 2. 创建策略
    strategy = BuyAndHoldStrategy(symbol="TEST", quantity=100)

    # 3. 创建引擎
    engine = BacktestEngine(
        data={"TEST": data},
        strategy=strategy,
        initial_cash=100000,
    )

    # 4. 运行回测
    results = engine.run()

    # 5. 验证结果
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert results["total_return"] > 0
```

#### Agent集成测试

```python
# tests/integration/test_agent_workflow.py

import pytest
from agents.base.coordinator import AgentCoordinator

@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_agent_workflow():
    """多Agent协作测试"""

    # 1. 创建协调器
    coordinator = AgentCoordinator()

    # 2. 注册Agent
    coordinator.register_agent(MarketDataAgent("market"))
    coordinator.register_agent(StrategyAgent("strategy"))

    # 3. 创建工作流
    workflow = Workflow(name="test_workflow")
    workflow.add_step(lambda coord: fetch_market_data(coord))
    workflow.add_step(lambda coord: generate_strategy(coord))

    # 4. 执行工作流
    results = await coordinator.run_workflow(workflow)

    # 5. 验证结果
    assert "step_1" in results
    assert "step_2" in results
    assert results["step_1"]["status"] == "success"
```

### E2E测试

#### 端到端测试示例

```python
# tests/integration/test_e2e.py

import pytest
from rl.training.trainer import RLTrainer
from rl.envs.a_share_trading_env import ASharesTradingEnv

@pytest.mark.integration
@pytest.mark.slow
def test_rl_training_e2e():
    """RL训练端到端测试"""

    # 1. 准备环境
    data = load_test_data()
    env = ASharesTradingEnv(data=data)

    # 2. 创建训练器
    trainer = RLTrainer(algorithm="ppo", env=env)

    # 3. 训练 (短时间)
    model = trainer.train(total_timesteps=1000)

    # 4. 评估
    eval_results = trainer.evaluate(model, n_episodes=10)

    # 5. 验证
    assert eval_results["mean_reward"] > -1000
    assert model is not None
```

### 测试运行

```bash
# 运行所有测试
make test
# 或
pytest -v

# 运行特定测试
pytest tests/backtest/test_backtest.py -v

# 运行特定标记的测试
pytest -m "unit" -v              # 只运行单元测试
pytest -m "integration" -v       # 只运行集成测试
pytest -m "not slow" -v          # 排除慢速测试

# 运行测试并生成覆盖率报告
make test-cov
# 或
pytest --cov=. --cov-report=html --cov-report=term

# 查看覆盖率报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# 并行运行测试 (需要pytest-xdist)
pytest -n auto

# 只运行失败的测试
pytest --lf

# 详细输出
pytest -vv -s
```

### 性能测试

```python
# tests/performance/test_performance.py

import pytest
import time

@pytest.mark.benchmark
def test_backtest_performance():
    """回测性能测试"""

    data = generate_large_test_data(n_symbols=100, n_days=1000)

    start_time = time.time()
    engine = BacktestEngine(data=data, strategy=strategy)
    results = engine.run()
    elapsed = time.time() - start_time

    # 性能要求: 1000天回测在10秒内完成
    assert elapsed < 10.0
    print(f"回测耗时: {elapsed:.2f}秒")
```

---

## 性能优化技巧

### Python性能优化

#### 1. 使用向量化操作

```python
# 慢速: 循环
def calculate_returns_slow(prices):
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(ret)
    return returns

# 快速: 向量化
import numpy as np

def calculate_returns_fast(prices):
    prices = np.array(prices)
    return np.diff(prices) / prices[:-1]

# 更快: Pandas内置
import pandas as pd

def calculate_returns_pandas(prices):
    return pd.Series(prices).pct_change()
```

#### 2. 避免重复计算

```python
# 使用缓存
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_rsi(prices_hash, period=14):
    """带缓存的RSI计算"""
    # 计算逻辑...
    pass

# 或使用类属性缓存
class TechnicalIndicators:
    def __init__(self):
        self._cache = {}

    def calculate_ma(self, prices, period):
        cache_key = (id(prices), period)
        if cache_key not in self._cache:
            self._cache[cache_key] = prices.rolling(period).mean()
        return self._cache[cache_key]
```

#### 3. 使用高效的数据结构

```python
# 慢速: 列表查找
symbols_list = ["AAPL", "MSFT", "GOOGL", ...]
if symbol in symbols_list:  # O(n)

# 快速: 集合查找
symbols_set = {"AAPL", "MSFT", "GOOGL", ...}
if symbol in symbols_set:  # O(1)

# 快速: 字典查找
symbol_map = {"AAPL": 1, "MSFT": 2, "GOOGL": 3, ...}
if symbol in symbol_map:  # O(1)
```

#### 4. 使用生成器

```python
# 内存占用大: 列表
def process_large_file(filename):
    with open(filename) as f:
        lines = f.readlines()  # 所有行加载到内存
        for line in lines:
            yield process_line(line)

# 内存占用小: 生成器
def process_large_file(filename):
    with open(filename) as f:
        for line in f:  # 逐行读取
            yield process_line(line)
```

#### 5. 多进程/多线程

```python
from multiprocessing import Pool
import numpy as np

def process_symbol(symbol):
    """处理单个股票"""
    data = load_data(symbol)
    return backtest(data)

# 并行处理多个股票
symbols = ["600519.SH", "000858.SZ", "600036.SH", ...]

with Pool(processes=4) as pool:
    results = pool.map(process_symbol, symbols)
```

### Rust集成优化

#### 1. 性能关键部分用Rust实现

```rust
// rust_engine/src/indicators.rs

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

/// 计算移动平均线 (Rust实现)
#[pyfunction]
fn calculate_ma(prices: PyReadonlyArray1<f64>, period: usize) -> PyResult<Vec<f64>> {
    let prices = prices.as_slice()?;
    let mut ma = Vec::with_capacity(prices.len());

    for i in 0..prices.len() {
        if i < period - 1 {
            ma.push(f64::NAN);
        } else {
            let sum: f64 = prices[i-period+1..=i].iter().sum();
            ma.push(sum / period as f64);
        }
    }

    Ok(ma)
}

/// Python模块
#[pymodule]
fn quanta_rust_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_ma, m)?)?;
    Ok(())
}
```

```python
# Python中使用
from rust_engine import calculate_ma
import numpy as np

prices = np.random.randn(10000)
ma = calculate_ma(prices, 20)  # Rust实现，更快
```

#### 2. 使用Rayon并行计算

```rust
use rayon::prelude::*;

/// 并行计算多个股票的指标
#[pyfunction]
fn calculate_ma_parallel(
    prices_matrix: PyReadonlyArray2<f64>,
    period: usize
) -> PyResult<Vec<Vec<f64>>> {
    let prices = prices_matrix.as_slice()?;

    // 并行处理每行
    let results: Vec<Vec<f64>> = prices
        .par_chunks(prices_matrix.shape()[1])
        .map(|row| {
            // 计算逻辑...
        })
        .collect();

    Ok(results)
}
```

### 数据库优化

#### 1. DuckDB优化

```python
import duckdb

# 使用列式存储
conn = duckdb.connect("data/quant_a.duckdb")

# 创建优化的表结构
conn.execute("""
    CREATE TABLE IF NOT EXISTS daily_bars (
        symbol VARCHAR,
        date DATE,
        open DOUBLE,
        high DOUBLE,
        low DOUBLE,
        close DOUBLE,
        volume BIGINT,
        PRIMARY KEY (symbol, date)
    )
""")

# 批量插入 (更快)
conn.execute("""
    INSERT INTO daily_bars
    SELECT * FROM read_csv('data/*.csv')
""")

# 创建索引
conn.execute("""
    CREATE INDEX idx_symbol_date ON daily_bars(symbol, date)
""")
```

#### 2. ClickHouse优化

```python
import clickhouse_connect

# 使用MergeTree引擎 (最适合时序数据)
client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    database='quant_a'
)

# 创建优化的表
client.execute("""
    CREATE TABLE IF NOT EXISTS daily_bars (
        symbol String,
        date Date,
        open Float64,
        high Float64,
        low Float64,
        close Float64,
        volume UInt64
    )
    ENGINE = MergeTree()
    ORDER BY (symbol, date)
    PARTITION BY toYYYYMM(date)
""")
```

### 内存优化

#### 1. 使用适当的数据类型

```python
import pandas as pd
import numpy as np

# 使用更小的数据类型
df['price'] = df['price'].astype(np.float32)  # 而非float64
df['volume'] = df['volume'].astype(np.int32)  # 而非int64
df['symbol'] = df['symbol'].astype('category')  # 分类类型

# 检查内存使用
print(df.memory_usage(deep=True))
```

#### 2. 分块处理大数据

```python
# 分块读取CSV
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# 使用Dask处理超大数据
import dask.dataframe as dd

df = dd.read_csv('very_large_file.csv')
result = df.groupby('symbol').mean().compute()  # 并行计算
```

---

## 调试技巧

### 日志调试

#### 1. 配置日志

```python
from utils.logging import get_logger

# 获取logger
logger = get_logger(__name__)

# 设置日志级别
import logging
logger.setLevel(logging.DEBUG)

# 不同级别的日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

#### 2. 结构化日志

```python
# 使用结构化日志
logger.info(
    "订单执行成功",
    extra={
        "symbol": "600519.SH",
        "action": "BUY",
        "quantity": 100,
        "price": 1850.0,
        "cost": 185000.0
    }
)

# 或使用f-string
logger.info(
    f"订单执行: symbol={symbol}, action={action}, "
    f"quantity={quantity}, price={price:.2f}"
)
```

### 性能分析

#### 1. 使用cProfile

```bash
# 运行性能分析
python -m cProfile -o profile.stats your_script.py

# 查看结果
python -m pstats profile.stats
```

```python
# 在代码中使用
import cProfile
import pstats

def profile_function():
    pr = cProfile.Profile()
    pr.enable()

    # 你的代码
    your_function()

    pr.disable()

    # 打印统计
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 打印前10个最慢的函数
```

#### 2. 使用line_profiler

```bash
# 安装
pip install line_profiler

# 使用
kernprof -l -v your_script.py
```

```python
# 在代码中标记要分析的函数
@profile
def slow_function():
    # 函数实现
    pass
```

#### 3. 内存分析

```bash
# 使用memory_profiler
pip install memory_profiler

# 运行
python -m memory_profiler your_script.py
```

```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 函数实现
    pass
```

### 断点调试

#### 1. 使用pdb

```python
# 在代码中设置断点
import pdb; pdb.set_trace()

# 或使用breakpoint() (Python 3.7+)
breakpoint()

# 常用命令
# n (next): 执行下一行
# s (step): 进入函数
# c (continue): 继续执行
# p variable: 打印变量
# l (list): 显示代码
# q (quit): 退出
```

#### 2. 使用IPython调试器

```python
# 使用IPython的增强调试器
from IPython import embed; embed()

# 或使用ipdb
import ipdb; ipdb.set_trace()
```

#### 3. VS Code调试

配置 `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["-v", "${file}"],
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

### 常见问题调试

#### 1. 数值计算错误

```python
import numpy as np

# 检查NaN和Inf
def check_array(arr):
    if np.any(np.isnan(arr)):
        print("发现NaN值")
        print(f"NaN位置: {np.where(np.isnan(arr))}")
    if np.any(np.isinf(arr)):
        print("发现Inf值")
        print(f"Inf位置: {np.where(np.isinf(arr))}")

# 检查数值范围
def check_range(arr, min_val, max_val):
    out_of_range = (arr < min_val) | (arr > max_val)
    if np.any(out_of_range):
        print(f"发现{np.sum(out_of_range)}个超出范围的值")
```

#### 2. 内存泄漏

```python
import gc
import sys

def check_memory():
    """检查内存使用"""
    # 强制垃圾回收
    gc.collect()

    # 获取对象引用计数
    objects = gc.get_objects()
    print(f"总对象数: {len(objects)}")

    # 查找大对象
    large_objects = [
        obj for obj in objects
        if sys.getsizeof(obj) > 1000000  # > 1MB
    ]
    print(f"大对象数: {len(large_objects)}")
```

#### 3. 死锁和竞态条件

```python
import asyncio
import logging

# 启用异步调试
asyncio.run(main(), debug=True)

# 或设置日志级别
logging.basicConfig(level=logging.DEBUG)
```

---

## 发布流程

### 版本管理

#### 1. 语义化版本

遵循语义化版本规范 (Semantic Versioning):

```
MAJOR.MINOR.PATCH

例如: 1.2.3

- MAJOR: 不兼容的API更改
- MINOR: 向后兼容的功能新增
- PATCH: 向后兼容的Bug修复
```

#### 2. 版本号管理

在 `config/settings.py` 中定义版本:

```python
__version__ = "1.0.0"
```

使用标签标记版本:

```bash
# 创建版本标签
git tag -a v1.0.0 -m "Release version 1.0.0"

# 推送标签
git push origin v1.0.0

# 查看所有标签
git tag -l

# 删除标签
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
```

### 发布检查清单

#### 发布前检查

- [ ] 所有测试通过 (`make test`)
- [ ] 代码覆盖率达标 (>= 70%)
- [ ] 文档已更新
- [ ] CHANGELOG.md已更新
- [ ] 版本号已更新
- [ ] 依赖已锁定 (`requirements.txt` 或 `Pipfile.lock`)
- [ ] 安全扫描无漏洞
- [ ] 性能测试通过
- [ ] 兼容性测试通过

#### 预发布测试

```bash
# 1. 运行完整测试套件
make test-cov

# 2. 运行集成测试
pytest -m "integration" -v

# 3. 运行性能测试
pytest -m "benchmark" -v

# 4. 测试安装
pip install -e .

# 5. 测试示例
python examples/backtest_example.py
python examples/rl_training_guide.py
```

### 发布流程

#### 1. 准备发布

```bash
# 1. 切换到master分支
git checkout master
git pull origin master

# 2. 创建发布分支
git checkout -b release/v1.0.0

# 3. 更新版本号
vim config/settings.py
# __version__ = "1.0.0"

# 4. 更新CHANGELOG
vim CHANGELOG.md
```

#### 2. 构建发布

```bash
# 1. 构建Rust引擎 (如果使用)
cd rust_engine
cargo build --release
cd ..

# 2. 运行完整测试
make test

# 3. 生成文档 (如果有)
make docs

# 4. 提交更改
git add config/settings.py CHANGELOG.md
git commit -m "chore: prepare for v1.0.0 release"
```

#### 3. 创建标签

```bash
# 合并到master
git checkout master
git merge release/v1.0.0

# 创建标签
git tag -a v1.0.0 -m "Release v1.0.0

Features:
- Add multi-agent coordination
- Implement RL training pipeline
- Add Rust execution engine

Bug fixes:
- Fix A-share T+1 rule issue
- Fix memory leak in backtest engine"

# 推送
git push origin master
git push origin v1.0.0

# 删除发布分支
git branch -d release/v1.0.0
```

#### 4. 发布到PyPI (可选)

```bash
# 1. 安装构建工具
pip install build twine

# 2. 构建包
python -m build

# 3. 检查包
twine check dist/*

# 4. 上传到TestPyPI
twine upload --repository testpypi dist/*

# 5. 测试安装
pip install --index-url https://test.pypi.org/simple/ quantA

# 6. 上传到PyPI
twine upload dist/*
```

### 发布后

#### 1. 通知

- 在GitHub发布页面创建Release Notes
- 发送邮件通知用户
- 更新网站/文档

#### 2. 监控

- 监控错误日志
- 收集用户反馈
- 跟踪性能指标

#### 3. 维护

```bash
# 创建维护分支 (如果需要)
git checkout -b maintain/v1.0.x

# 修复Bug
git commit -m "fix: patch for v1.0.1"

# 发布补丁版本
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin v1.0.1
```

---

## 常见问题 (FAQ)

### 环境配置问题

#### Q1: ImportError: No module named 'xxx'

**问题**: 导入模块失败

**解决方案**:

```bash
# 1. 检查PYTHONPATH
echo $PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 2. 检查虚拟环境
which python
pip list

# 3. 重新安装依赖
pip install -r requirements.txt

# 4. 清除缓存
pip cache purge
pip install --no-cache-dir -r requirements.txt
```

#### Q2: Rust编译失败

**问题**: Rust引擎编译错误

**解决方案**:

```bash
# 1. 更新Rust
rustup update

# 2. 清除Rust缓存
cd rust_engine
cargo clean
cargo build --release

# 3. 检查Rust版本
rustc --version  # 需要 >= 1.70

# 4. 检查PyO3版本
cargo tree | grep pyo3
```

### 代码问题

#### Q3: 如何添加新的技术指标？

**问题**: 需要添加自定义技术指标

**解决方案**:

```python
# 在 backtest/engine/indicators.py 中添加

class TechnicalIndicators:
    def calculate_custom_indicator(self, prices: pd.Series, period: int = 20):
        """
        自定义指标

        Args:
            prices: 价格序列
            period: 周期

        Returns:
            指标值
        """
        # 实现你的逻辑
        indicator = prices.rolling(period).apply(
            lambda x: custom_logic(x)
        )
        return indicator

# 使用
ti = TechnicalIndicators()
custom = ti.calculate_custom_indicator(data['close'], period=20)
```

#### Q4: 如何实现自定义奖励函数？

**问题**: 需要自定义RL奖励函数

**解决方案**:

```python
# 在 rl/rewards/reward_functions.py 中添加

from abc import ABC, abstractmethod

class CustomReward(ABC):
    """自定义奖励函数基类"""

    @abstractmethod
    def calculate(self, env) -> float:
        """计算奖励"""
        pass

class MyCustomReward(CustomReward):
    """我的自定义奖励"""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def calculate(self, env) -> float:
        """
        根据你的逻辑计算奖励

        例如: 考虑收益率、波动率、最大回撤等
        """
        # 获取环境状态
        current_value = env.portfolio.current_value
        previous_value = env.portfolio.previous_value
        drawdown = env.portfolio.max_drawdown

        # 计算奖励
        return_weight = (current_value - previous_value) / previous_value
        drawdown_penalty = -abs(drawdown)

        reward = self.weight * (return_weight + 0.1 * drawdown_penalty)

        return reward

# 在环境中使用
from rl.rewards.reward_functions import MyCustomReward

reward_fn = MyCustomReward(weight=1.0)
reward = reward_fn.calculate(env)
```

### 性能问题

#### Q5: 回测速度慢，如何优化？

**问题**: 回测运行太慢

**解决方案**:

```python
# 1. 使用Rust引擎
from rust_engine import RustEngine

engine = RustEngine()
engine.run_backtest(data, strategy)

# 2. 减少数据量
# 只加载需要的日期范围
data = data.loc["2023-01-01":"2023-12-31"]

# 3. 预计算技术指标
from backtest.engine.indicators import TechnicalIndicators

ti = TechnicalIndicators()
data['ma5'] = ti.calculate_ma(data['close'], 5)
data['rsi'] = ti.calculate_rsi(data['close'], 14)

# 4. 使用更高效的数据结构
data = data.astype({
    'open': 'float32',
    'high': 'float32',
    'low': 'float32',
    'close': 'float32',
    'volume': 'int32'
})

# 5. 并行化
from multiprocessing import Pool

symbols = ['600519.SH', '000858.SZ', ...]
with Pool(4) as pool:
    results = pool.map(backtest_single_symbol, symbols)
```

#### Q6: 内存溢出

**问题**: 处理大量数据时内存不足

**解决方案**:

```python
# 1. 分块处理
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)

# 2. 使用生成器
def data_generator():
    for symbol in symbols:
        yield load_data(symbol)

# 3. 及时释放内存
del large_dataframe
import gc
gc.collect()

# 4. 使用更高效的数据类型
df['price'] = df['price'].astype('float32')  # 而非float64

# 5. 使用Dask
import dask.dataframe as dd
df = dd.read_csv('very_large_file.csv')
result = df.groupby('symbol').mean().compute()
```

### 测试问题

#### Q7: 测试失败，如何调试？

**问题**: 某些测试失败

**解决方案**:

```bash
# 1. 运行失败的测试并显示详细输出
pytest tests/your_test.py -vv -s

# 2. 只运行失败的测试
pytest --lf

# 3. 在第一个失败时停止
pytest -x

# 4. 使用pdb调试
pytest --pdb

# 5. 查看print输出
pytest -s

# 6. 使用log
pytest --log-cli-level=DEBUG
```

#### Q8: 测试覆盖率不足

**问题**: 代码覆盖率不达标

**解决方案**:

```python
# 1. 查看未覆盖的代码
pytest --cov=. --cov-report=html
open htmlcov/index.html

# 2. 为未覆盖的代码添加测试
# 查看htmlcov报告中的红色部分

# 3. 测试边界条件
def test_edge_cases():
    # 测试空输入
    result = function(None)
    assert result is None

    # 测试极端值
    result = function(float('inf'))
    assert result == expected

    # 测试异常
    with pytest.raises(ValueError):
        function("invalid")

# 4. 测试异常分支
def test_exception_handling():
    with patch('module.external_api', side_effect=Exception):
        result = function()
        assert result == fallback_value
```

### 部署问题

#### Q9: Docker构建失败

**问题**: Docker镜像构建错误

**解决方案**:

```dockerfile
# Dockerfile优化

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置PYTHONPATH
ENV PYTHONPATH=/app

# 运行
CMD ["python", "examples/backtest_example.py"]
```

```bash
# 构建命令
docker build -t quanta:latest .

# 运行
docker run -v $(pwd)/data:/app/data quanta:latest

# 调试
docker run -it quanta:latest /bin/bash
```

#### Q10: 生产环境配置

**问题**: 如何配置生产环境

**解决方案**:

```bash
# 1. 使用配置文件
cp config/settings.py config/settings_prod.py

# 2. 设置环境变量
export ENV=production
export LOG_LEVEL=WARNING

# 3. 使用生产级数据库
export TIMESERIES_DB=clickhouse
export CLICKHOUSE_HOST=production-db

# 4. 配置日志
export LOG_FILE_ENABLED=true
export LOG_LEVEL=INFO

# 5. 配置监控
export PROMETHEUS_ENABLED=true
export PROMETHEUS_PORT=9090
```

---

## 贡献指南

### 如何贡献

欢迎贡献代码！请遵循以下流程：

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

### 代码规范

- 遵循PEP 8代码风格
- 使用有意义的变量名
- 函数和类添加docstring
- 添加单元测试
- 更新相关文档

### Pull Request模板

```markdown
## 描述
简要描述此PR的更改

## 类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 代码重构
- [ ] 文档更新
- [ ] 性能优化

## 测试
- [ ] 单元测试已添加/更新
- [ ] 所有测试通过
- [ ] 代码覆盖率达标

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 文档已更新
- [ ] CHANGELOG已更新
- [ ] 无合并冲突

## 相关Issue
Closes #123
```

---

## 联系方式

- **项目主页**: https://github.com/yourusername/quantA
- **问题反馈**: https://github.com/yourusername/quantA/issues
- **讨论区**: https://github.com/yourusername/quantA/discussions
- **邮箱**: your.email@example.com

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

**最后更新**: 2025-01-30
**文档版本**: 1.0.0
**维护者**: quantA Team
