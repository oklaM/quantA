# quantA 快速开始指南

欢迎使用 quantA - A股量化AI交易系统！本指南将帮助您在5分钟内快速上手。

## 📋 目录

- [系统要求](#系统要求)
- [快速安装](#快速安装)
- [环境验证](#环境验证)
- [运行第一个示例](#运行第一个示例)
- [核心功能概览](#核心功能概览)
- [常见问题](#常见问题)
- [下一步](#下一步)

---

## 系统要求

### 必需环境

- **Python**: 3.9 或更高版本
- **操作系统**: Linux, macOS, Windows
- **内存**: 至少 2GB RAM (推荐 4GB+)
- **磁盘**: 至少 5GB 可用空间

### 可选依赖

- Git (用于版本控制)
- C++ 编译器 (用于编译某些Python包)

---

## 快速安装

### 方法1：一键安装（推荐）

```bash
# 1. 克隆或下载项目
cd /path/to/quantA

# 2. 运行一键安装脚本
bash scripts/install.sh

# 3. 按照提示选择要安装的组件
```

安装脚本会自动：
- ✅ 检查Python版本
- ✅ 创建虚拟环境
- ✅ 安装所有依赖
- ✅ 验证安装
- ✅ 创建配置文件

### 方法2：手动安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 2. 升级pip
pip install --upgrade pip setuptools wheel

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装开发工具（可选）
pip install pytest pytest-cov black flake8
```

---

## 环境验证

### 运行环境检查

```bash
# 激活虚拟环境后
python scripts/check_env.py
```

这将检查：
- Python版本和环境
- 所有依赖包
- 系统资源（内存、磁盘等）
- 配置文件
- 数据源连接

### 运行系统验证

```bash
bash scripts/verify_system.sh
```

这将验证：
- 项目结构完整性
- 测试文件数量
- 关键文件存在性
- 文档完整性

**预期结果**：通过率 >= 80% ✅

---

## 运行第一个示例

### 激活环境

```bash
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows
```

### 运行简单策略回测

创建文件 `my_first_strategy.py`:

```python
from backtest.engine import BacktestEngine
from backtest.strategies import BuyAndHoldStrategy

# 创建回测引擎
engine = BacktestEngine(
    initial_cash=1000000,  # 100万初始资金
    commission=0.0003,     # 万三手续费
)

# 生成模拟数据
data = engine.generate_mock_data(
    symbols=['600000.SH', '000001.SZ'],
    start_date='2023-01-01',
    end_date='2023-12-31',
)

# 运行回测
strategy = BuyAndHoldStrategy()
results = engine.run(strategy, data)

# 打印结果
print(f"总收益率: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

运行：

```bash
python my_first_strategy.py
```

### 使用便捷脚本

```bash
# 运行测试
./quanta.sh test

# 运行示例
./quanta.sh example

# 验证系统
./quanta.sh verify
```

---

## 核心功能概览

### 1. 回测引擎

```python
from backtest.engine import BacktestEngine

engine = BacktestEngine(
    initial_cash=1000000,
    commission=0.0003,
    slippage=0.0001,
)
```

### 2. 技术指标

```python
from backtest.indicators import *

# 移动平均线
sma = SMA(data['close'], period=20)
ema = EMA(data['close'], period=20)

# MACD
macd_line, signal_line, histogram = MACD(data['close'])

# RSI
rsi = RSI(data['close'], period=14)

# 布林带
upper, middle, lower = BOLLINGER_BANDS(data['close'])
```

### 3. LLM智能体

```python
from agents.glmmarket_agent import GLMMarketAgent

agent = GLMMarketAgent(
    api_key='your_api_key',
    model='glm-4',
)

response = agent.analyze_market(
    symbol='600000.SH',
    data=market_data,
)
```

### 4. 强化学习

```python
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer

env = ASharesTradingEnv(data=data)
trainer = RLTrainer(env, algorithm='ppo')

model = trainer.train(total_timesteps=10000)
results = trainer.evaluate(model)
```

### 5. 风控系统

```python
from trading.risk import RiskController

controller = RiskController({
    'max_daily_loss_ratio': 0.03,  # 日亏损限制3%
    'max_single_order_amount': 1000000,  # 单笔限额100万
})

allowed, rejects = controller.validate_order(
    symbol='600000.SH',
    action='buy',
    quantity=1000,
    price=10.50,
    context=context,
)
```

---

## 配置说明

### 环境变量配置

编辑 `.env` 文件：

```bash
# 数据源配置
TUSHARE_TOKEN=your_token_here
AKSHARE_ENABLED=true

# LLM配置
GLM_API_KEY=your_api_key_here
GLM_MODEL=glm-4

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/quanta.log

# 性能配置
NUMBA_ENABLED=true
MULTIPROCESSING=true
```

### 数据源配置

**使用AKShare（免费）**:

```python
from data.market.sources import AKShareProvider

provider = AKShareProvider()
provider.connect()
data = provider.get_daily_bar('600000.SH', '20230101', '20231231')
```

**使用Tushare（需Token）**:

```python
from data.market.sources import TushareProvider

provider = TushareProvider(token='your_token')
provider.connect()
data = provider.get_daily_bar('600000.SH', '20230101', '20231231')
```

---

## 运行测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定模块测试

```bash
# 回测引擎测试
pytest tests/backtest/ -v

# 技术指标测试
pytest tests/backtest/test_indicators.py -v

# 风控系统测试
pytest tests/trading/test_risk_controls.py -v

# RL模块测试
pytest tests/rl/ -v
```

### 查看测试覆盖率

```bash
pytest --cov=. --cov-report=html --cov-report=term
```

覆盖率报告将保存在 `htmlcov/index.html`

---

## 常见问题

### Q1: 安装时提示权限错误

**A**:
```bash
# 使用用户安装模式
pip install --user -r requirements.txt

# 或使用虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Q2: 导入模块失败

**A**: 确保在项目根目录运行：

```bash
cd /path/to/quantA
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python your_script.py
```

### Q3: 数据获取失败

**A**:
- 检查网络连接
- 确认API Token正确（Tushare）
- AKShare需要稳定的网络连接
- 查看日志文件 `logs/quanta.log`

### Q4: 内存不足

**A**:
- 减少回测的股票数量
- 缩短时间范围
- 启用数据分批处理
- 增加系统交换空间

### Q5: Rust引擎相关错误

**A**: Rust引擎是可选的优化功能。如果遇到问题：

```bash
# 禁用Rust引擎，使用Python引擎
export USE_RUST_ENGINE=false
```

---

## 项目结构

```
quantA/
├── agents/           # LLM智能体模块
├── backtest/         # 回测引擎和策略
├── data/             # 数据获取和处理
├── live/             # 实盘交易接口
├── monitoring/       # 监控和告警
├── rl/               # 强化学习模块
├── trading/          # 交易执行和风控
├── utils/            # 工具函数
├── tests/            # 测试文件
├── examples/         # 示例代码
├── scripts/          # 脚本工具
├── docs/             # 文档
└── logs/             # 日志文件
```

---

## 示例代码

### 示例1：双均线策略

```python
from backtest.engine import BacktestEngine
from backtest.strategies import MovingAverageCrossStrategy

engine = BacktestEngine(initial_cash=1000000)
data = engine.generate_mock_data(symbols=['600000.SH'])

strategy = MovingAverageCrossStrategy(
    short_window=5,
    long_window=20,
)

results = engine.run(strategy, data)
print(results)
```

### 示例2：技术指标分析

```python
from backtest.indicators import SMA, RSI, MACD
import pandas as pd

# 计算指标
data['sma_20'] = SMA(data['close'], 20)
data['rsi'] = RSI(data['close'], 14)
macd_line, signal_line, histogram = MACD(data['close'])

# 生成交易信号
data['signal'] = 0
data.loc[data['sma_20'] > data['close'], 'signal'] = -1  # 卖出
data.loc[data['sma_20'] < data['close'], 'signal'] = 1   # 买入
```

### 示例3：参数优化

```python
from backtest.optimization import GridSearchOptimizer

optimizer = GridSearchOptimizer(
    engine=engine,
    strategy=MovingAverageCrossStrategy,
    param_grid={
        'short_window': [5, 10, 15],
        'long_window': [20, 30, 40],
    },
)

best_params = optimizer.optimize(data)
print(f"最优参数: {best_params}")
```

更多示例请查看 `examples/` 目录。

---

## 性能优化建议

### 1. 使用Numba加速

```python
from numba import jit

@jit(nopython=True)
def fast_indicator_calculation(data):
    # 你的计算逻辑
    return result
```

### 2. 并行处理

```python
from multiprocessing import Pool

def process_symbol(symbol):
    # 处理单个股票
    pass

with Pool(processes=4) as pool:
    results = pool.map(process_symbol, symbols)
```

### 3. 数据缓存

```python
import pickle

# 保存数据
with open('cache.pkl', 'wb') as f:
    pickle.dump(data, f)

# 加载数据
with open('cache.pkl', 'rb') as f:
    data = pickle.load(f)
```

---

## 下一步

### 学习资源

- 📖 [完整文档](docs/README.md)
- 📓 [API参考](docs/API_REFERENCE.md)
- 💡 [示例集合](examples/)
- 🧪 [测试用例](tests/)

### 进阶功能

1. **自定义策略** → [策略开发指南](docs/STRATEGY_GUIDE.md)
2. **RL训练** → [强化学习教程](docs/RL_TUTORIAL.md)
3. **实盘交易** → [实盘部署指南](docs/DEPLOYMENT.md)
4. **性能优化** → [优化指南](docs/OPTIMIZATION.md)

### 参与贡献

欢迎贡献代码、报告问题或提出建议！

- 🐛 [报告问题](https://github.com/yourusername/quantA/issues)
- 💬 [讨论区](https://github.com/yourusername/quantA/discussions)
- 📧 Email: your@email.com

---

## 获取帮助

### 查看日志

```bash
tail -f logs/quanta.log
```

### 环境诊断

```bash
python scripts/check_env.py
```

### 系统验证

```bash
bash scripts/verify_system.sh
```

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 致谢

感谢所有贡献者和开源项目的支持！

---

**版本**: 1.0.0
**更新日期**: 2026-01-13

🚀 **开始您的量化交易之旅吧！**
