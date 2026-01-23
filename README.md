# quantA - A股量化AI交易系统

> 基于强化学习和LLM Agent技术的A股量化交易系统

## 项目概述

quantA是一个面向A股市场的量化交易系统，结合了强化学习和大语言模型Agent技术，旨在实现中频/日内交易的盈利。

### 核心特性

- **多Agent决策**: 市场数据、技术分析、情绪分析、策略生成、风控Agent协同工作
- **强化学习**: 基于PPO/DQN算法的智能交易策略
- **A股适配**: 支持T+1、涨跌停、最小申报单位等A股特殊规则
- **高性能执行**: Python策略研发 + Rust执行引擎
- **实时监控**: 完整的监控告警和合规报告系统

### 系统架构

```
LLM Agent决策层 → 强化学习策略层 → Rust执行引擎 → 数据基础设施
```

- **决策层**: GLM-4驱动的多Agent系统
- **策略层**: Stable-Baselines3强化学习框架
- **执行层**: Rust高性能订单管理和风控
- **数据层**: ClickHouse/DuckDB时序数据存储

## 技术栈

| 层级 | 技术选型 |
|------|----------|
| LLM模型 | GLM-4/GLM-4-Plus (智谱AI) |
| LLM框架 | LangChain / LangGraph |
| RL框架 | Stable-Baselines3 / FinRL |
| 数据存储 | DuckDB / ClickHouse |
| 数据处理 | Pandas / NumPy |
| 执行引擎 | Rust + PyO3 |
| 券商接口 | 华泰XTP |

## 📊 项目状态

- **测试通过率**: ✅ 100% (262/262)
- **代码覆盖率**: 42.47%
- **文档完整性**: ⭐⭐⭐⭐⭐
- **代码质量**: ⭐⭐⭐⭐⭐ (Black + Isort + Flake8)

## 🚀 快速开始

### 5分钟上手

```python
from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategies import BuyAndHoldStrategy
import pandas as pd

# 1. 准备数据
data = pd.read_csv('your_data.csv')

# 2. 创建策略
strategy = BuyAndHoldStrategy(symbol="600519.SH", quantity=1000)

# 3. 运行回测
engine = BacktestEngine(
    data={"600519.SH": data},
    strategy=strategy,
    initial_cash=1_000_000,
)

results = engine.run()

# 4. 查看结果
print(f"收益率: {results['total_return_pct']:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
```

### 环境要求

- Python 3.10+
- Rust 1.70+ (可选，用于高性能执行引擎)
- Linux/macOS (推荐)

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/quantA.git
cd quantA

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入API密钥
vim .env
```

### 配置

1. **获取API密钥**:
   - 智谱AI: https://open.bigmodel.cn/
   - Tushare: https://tushare.pro/
   - 华泰XTP: 需要开通华泰证券账户

2. **配置环境变量** (`.env`):
   ```bash
   ZHIPUAI_API_KEY=your_key_here
   TUSHARE_TOKEN=your_token_here
   ```

### 运行示例

```bash
# 初始化数据库
make db-init

# 运行回测示例
make run
# 或
python examples/backtest_example.py

# 运行Agent示例
make run-agent
# 或
python examples/agent_example.py

# 运行RL训练示例
python examples/rl_training_guide.py

# 运行高级策略示例
python examples/advanced_strategy_example.py

# 启动监控面板
make monitor

# 运行测试
make test
```

### 开发命令

项目使用 Makefile 管理常用开发任务：

```bash
# 测试相关
make test              # 运行所有测试
make test-cov          # 运行测试并生成覆盖率报告
make test-unit         # 只运行单元测试
make test-agents       # 只运行Agent测试
make test-rl           # 只运行强化学习测试

# 代码质量
make format            # 格式化代码 (black + isort)
make lint              # 代码检查 (flake8 + mypy)

# 数据库
make db-init           # 初始化数据库
make db-backup         # 备份数据库

# 其他
make clean             # 清理临时文件
make monitor           # 启动监控面板
make api               # 启动API服务
```

## 使用指南

### 强化学习训练

完整的RL训练工作流示例：

```bash
# 运行完整RL训练流程
python examples/rl_complete_workflow.py
```

该示例包含：
1. 数据准备和预处理
2. 交易环境创建
3. PPO/DQN模型训练
4. 模型评估和对比
5. 模型保存和加载

#### 模型管理

```python
from rl.models import ModelManager

# 创建模型管理器
manager = ModelManager(models_dir="models")

# 保存训练好的模型
version = manager.save_model(
    model=trained_model,
    algorithm="ppo",
    metadata={"total_timesteps": 50000}
)

# 加载模型
model = manager.load_model(version.version_id)

# 列出所有版本
versions = manager.list_versions(algorithm="ppo")
```

### 监控告警

使用监控告警系统：

```python
from live.monitoring import create_default_alert_manager, Monitor

# 创建告警管理器
alert_manager = create_default_alert_manager()
alert_manager.start()

# 创建监控器
monitor = Monitor(alert_manager)
monitor.start()

# 查看状态摘要
print(monitor.get_summary())

# 获取完整状态
status = monitor.get_status()
```

#### 告警规则

系统内置了多种告警规则：
- 亏损限制告警（单日亏损超过5%）
- 回撤限制告警（最大回撤超过10%）
- 持仓集中度告警（单一持仓超过30%）
- 数据延迟告警（数据延迟超过5分钟）

## 项目结构

```
quantA/
├── agents/              # LLM Agent系统 ✅
│   ├── base/           # Agent基类和协调器
│   ├── market_data_agent/  # 市场数据Agent
│   ├── technical_agent/    # 技术分析Agent
│   ├── sentiment_agent/    # 情绪分析Agent
│   ├── strategy_agent/     # 策略生成Agent
│   └── risk_agent/         # 风控Agent
├── rl/                 # 强化学习系统 ✅
│   ├── envs/           # A股交易环境 (Gymnasium)
│   ├── models/         # 模型管理和持久化
│   ├── training/       # 训练器 (PPO/DQN/A2C/SAC/TD3)
│   ├── rewards/        # 奖励函数库 (8+种)
│   └── evaluation/     # 模型评估和对比
├── rust_engine/        # Rust执行引擎 🔄
│   ├── src/            # 源代码
│   │   ├── execution.rs    # 订单执行
│   │   ├── portfolio.rs    # 投资组合
│   │   └── error.rs        # 错误处理
│   └── Cargo.toml      # Rust项目配置
├── data/               # 数据层 ✅
│   ├── market/         # 行情数据采集
│   ├── storage/        # 数据存储 (DuckDB/ClickHouse)
│   └── fundamental/    # 基本面数据
├── backtest/           # 回测系统 ✅
│   ├── engine/         # 回测引擎、技术指标、A股规则
│   └── metrics/        # 绩效分析和报告
├── live/               # 实盘交易 🔄
│   ├── brokers/        # 券商接口 (XTP)
│   └── monitoring/     # 监控告警系统
│       ├── alerting.py     # 告警管理
│       └── monitor.py      # 实时监控
├── config/             # 配置文件
│   ├── settings.py     # 全局配置
│   ├── symbols.py      # 股票池配置
│   └── strategies.py   # 策略参数
├── utils/              # 工具函数
├── examples/           # 示例代码
├── tests/              # 测试套件
└── docs/               # 文档
```

## 开发路线

**整体进度**: 70% (28/40 任务完成) | **测试通过率**: 100% (262/262) | **代码覆盖率**: 42.47% | 详细进度见 [PROGRESS.md](PROGRESS.md)

### 已实现功能

可立即使用的功能模块：

- ✅ **回测系统** - 完整的事件驱动回测引擎，支持20+技术指标
- ✅ **A股规则** - T+1、涨跌停、交易时间等A股特定规则
- ✅ **绩效分析** - 夏普比率、最大回撤、胜率等20+绩效指标
- ✅ **LLM Agent** - 5个专业Agent + 协调器，基于GLM-4
- ✅ **技术指标** - MA/EMA/MACD/RSI/KDJ/布林带/ATR等
- ✅ **报告生成** - 自动生成HTML格式的回测报告
- ✅ **强化学习** - 完整的RL训练框架（PPO/DQN/A2C/SAC/TD3）
- ✅ **模型管理** - 模型版本控制和持久化
- ✅ **监控告警** - 实时监控和告警系统

### Phase 1: 基础设施 ✅
- [x] 项目结构搭建
- [x] 配置管理 (settings.py, symbols.py, strategies.py)
- [x] 日志和时间工具

### Phase 2: 策略研发框架 ✅
- [x] 技术指标计算模块 (20+ 指标)
- [x] A股规则引擎 (T+1/涨跌停/交易时间)
- [x] 事件驱动回测引擎
- [x] 滑点和交易成本模型
- [x] 绩效分析模块
- [x] HTML报告生成
- [x] 双均线策略示例

### Phase 3: LLM Agent系统 ✅
- [x] Agent基类和消息协议
- [x] Agent协调器 (LangGraph集成)
- [x] GLM-4模型集成
- [x] 5个核心Agent实现
  - [x] 市场数据Agent
  - [x] 技术分析Agent
  - [x] 情绪分析Agent
  - [x] 策略生成Agent
  - [x] 风控Agent

### Phase 4: 强化学习系统 ✅
- [x] A股交易环境 (Gymnasium接口)
- [x] 奖励函数设计 (8+种奖励函数)
- [x] PPO/DQN模型实现 (支持A2C/SAC/TD3)
- [x] 训练和评估框架
- [x] 模型训练和测试
- [x] 模型版本管理
- [x] 大规模训练示例

### Phase 5: 执行引擎 🔄 (50%)
- [x] Rust基础架构 (错误处理、事件系统)
- [x] 订单管理模块
- [x] 订单簿模块
- [x] 投资组合模块
- [x] Python-Rust绑定 (PyO3)
- [x] Python包装器
- [ ] 完整编译和部署
- [ ] 性能测试和优化

### Phase 6: 实盘对接 🔄 (40%)
- [x] XTP接口基础框架
- [x] 模拟交易实现
- [x] 监控告警系统
- [x] Web监控面板 (Streamlit)
- [ ] XTP C++ SDK集成
- [ ] 实盘数据对接
- [ ] 合规报告系统

## 风险提示

本系统仅供学习和研究使用。量化交易存在风险，过往业绩不代表未来表现。

- 市场风险：市场波动可能导致亏损
- 模型风险：AI模型可能失效
- 技术风险：系统故障可能影响交易
- 合规风险：需遵守监管要求

## 免责声明

本项目仅用于教育和研究目的。使用本系统进行实盘交易的所有风险由使用者自行承担。作者不对任何损失负责。

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 致谢

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- [LangChain](https://github.com/langchain-ai/langchain)
- [AKShare](https://github.com/akfamily/akshare)
