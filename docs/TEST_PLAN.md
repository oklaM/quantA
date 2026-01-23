# quantA 系统全面测试计划

## 测试概述

**项目名称**: quantA - A股量化AI交易系统
**测试日期**: 2026-01-13
**测试范围**: 全系统功能测试
**测试目标**: 验证所有19个任务模块的功能完整性和稳定性

---

## 测试环境

### 系统要求
- Python 3.9+
- 必需包: pytest, pandas, numpy, matplotlib
- 可选包: stable-baselines3, optuna, plotly

### 测试数据
- 模拟数据（程序生成）
- 时间范围: 500个交易日
- 股票数量: 4-10只

---

## 测试矩阵

| 模块 | 测试类型 | 优先级 | 预期时间 | 状态 |
|------|---------|--------|----------|------|
| 1. 单元测试框架 | 单元测试 | P0 | 5min | 待执行 |
| 2. 回测引擎 | 单元测试 | P0 | 10min | 待执行 |
| 3. 技术指标 | 单元测试 | P0 | 10min | 待执行 |
| 4. 数据集成 | 集成测试 | P1 | 15min | 待执行 |
| 5. LLM Agents | 集成测试 | P1 | 10min | 待执行 |
| 6. 回测策略 | 功能测试 | P1 | 10min | 待执行 |
| 7. RL环境 | 功能测试 | P1 | 15min | 待执行 |
| 8. 奖励函数 | 单元测试 | P1 | 10min | 待执行 |
| 9. RL训练 | 功能测试 | P1 | 20min | 待执行 |
| 10. RL评估 | 功能测试 | P2 | 15min | 待执行 |
| 11. 参数优化 | 功能测试 | P2 | 20min | 待执行 |
| 12. 可视化 | 功能测试 | P2 | 15min | 待执行 |
| 13. 组合回测 | 功能测试 | P2 | 20min | 待执行 |
| 14. 监控告警 | 功能测试 | P2 | 15min | 待执行 |
| 15. 风控系统 | 单元测试 | P0 | 15min | 待执行 |
| 16. 系统集成 | 端到端测试 | P0 | 30min | 待执行 |
| 17. 性能测试 | 性能测试 | P2 | 20min | 待执行 |
| 18. 文档完整性 | 检查 | P3 | 10min | 待执行 |
| 19. 代码质量 | 静态分析 | P3 | 10min | 待执行 |

**总计**: 约 4.5 小时

---

## 详细测试用例

### 模块1: 单元测试框架

**测试目标**: 验证pytest配置和测试基础设施

**测试用例**:
1. ✅ 验证pytest.ini配置正确
2. ✅ 验证conftest.py fixtures工作正常
3. ✅ 运行简单测试用例验证框架
4. ✅ 验证测试标记（markers）功能
5. ✅ 验证覆盖率配置

**预期结果**: 所有测试正常通过，覆盖率报告生成

**命令**:
```bash
cd /home/rowan/Projects/quantA
pytest tests/ -v --tb=short
pytest tests/conftest.py::test_sample_data -v
```

---

### 模块2: 回测引擎核心

**测试目标**: 验证事件驱动回测引擎

**关键测试点**:
- 事件队列管理
- 订单处理流程
- 持仓更新逻辑
- 资金计算准确性
- 手续费计算

**命令**:
```bash
pytest tests/backtest/test_event_engine.py -v
pytest tests/backtest/test_portfolio.py -v
pytest tests/backtest/test_execution.py -v
```

**预期结果**:
- 至少60个测试用例
- 通过率 >= 95%
- 无关键失败

---

### 模块3: 技术指标

**测试目标**: 验证20+技术指标计算准确性

**关键指标**:
- SMA, EMA, MACD
- RSI, KDJ, Bollinger Bands
- ATR, OBV, WVAD

**测试用例**:
1. 标准场景计算
2. 边界条件处理
3. 异常值处理
4. 与已知结果对比

**命令**:
```bash
pytest tests/backtest/test_indicators.py -v
```

**预期结果**: 至少20个指标的测试全部通过

---

### 模块4: 数据集成

**测试目标**: 验证数据源集成

**测试用例**:
1. AKShare连接和数据获取
2. Tushare连接和数据获取
3. 数据格式转换
4. 增量更新功能
5. 数据存储和读取

**示例代码**:
```python
from data.market.sources import AKShareProvider

provider = AKShareProvider()
provider.connect()
df = provider.get_daily_bar('000001.SZ', '20230101', '20231231')
assert len(df) > 0
assert 'datetime' in df.columns
```

---

### 模块5: LLM Agents

**测试目标**: 验证GLM-4集成和智能体协作

**测试用例**:
1. GLM-4客户端初始化
2. 单智能体对话
3. 多智能体协作流程
4. 错误处理机制

**命令**:
```bash
pytest tests/agents/test_agent_collaboration.py -v
```

---

### 模块6: RL环境和训练

**测试目标**: 验证强化学习框架

**测试用例**:
1. ASharesTradingEnv初始化
2. 环境步进逻辑
3. 奖励函数计算
4. 模型训练流程
5. 模型评估

**示例**:
```python
from rl.envs.a_share_trading_env import ASharesTradingEnv
from rl.training.trainer import RLTrainer

env = ASharesTradingEnv(data=data)
trainer = RLTrainer(env, algorithm='ppo')
model = trainer.train(total_timesteps=1000)
```

---

### 模块15: 风控系统

**测试目标**: 验证实盘风控规则

**测试用例**:
1. 资金限制规则
2. 单笔订单限制
3. 持仓限制
4. 股票黑名单
5. 日亏损限制

**命令**:
```bash
pytest tests/trading/test_risk_controls.py -v
```

---

### 模块16: 系统集成测试

**测试目标**: 端到端完整流程测试

**测试场景**:
1. 数据获取 → 回测 → 分析 → 可视化
2. 策略开发 → 参数优化 → 回测验证
3. RL训练 → 评估 → 部署
4. 实盘风控流程

---

## 测试执行计划

### Phase 1: 基础测试 (30分钟)
```bash
# 1. 单元测试
pytest tests/ -m "unit" -v

# 2. 回测引擎
pytest tests/backtest/ -v

# 3. 数据模块
pytest tests/data/ -v
```

### Phase 2: 功能测试 (1小时)
```bash
# 4. RL模块
pytest tests/rl/ -v

# 5. Agent模块
pytest tests/agents/ -v

# 6. 风控模块
pytest tests/trading/ -v
```

### Phase 3: 集成测试 (30分钟)
```bash
# 7. 运行示例验证
python examples/strategies_example.py
python examples/optimization_example.py
python examples/monitoring_example.py
```

### Phase 4: 覆盖率分析 (15分钟)
```bash
pytest --cov=. --cov-report=html --cov-report=term
```

---

## 测试通过标准

### 必须满足的条件 (P0)
- ✅ 所有单元测试通过率 >= 95%
- ✅ 无关键功能失败
- ✅ 代码覆盖率 >= 60%

### 应该满足的条件 (P1)
- ⚠️ 集成测试通过率 >= 90%
- ⚠️ 示例代码可运行

### 可以满足的条件 (P2)
- ⚠️ 性能基准达标
- ⚠️ 文档完整性 >= 80%

---

## 测试报告模板

```markdown
## 测试执行报告

**执行时间**: YYYY-MM-DD HH:MM:SS
**测试人员**: Claude
**测试版本**: v1.0.0

### 测试结果汇总
- 总测试用例: XXX
- 通过: XXX (XX%)
- 失败: XXX (XX%)
- 跳过: XXX (XX%)

### 模块测试详情
[每个模块的详细结果]

### 缺陷列表
[发现的缺陷和问题]

### 性能指标
[性能测试结果]

### 改进建议
[优化建议]
```

---

## 回归测试清单

每次更新后需验证:
- [ ] 现有测试用例全部通过
- [ ] 新增功能有对应测试
- [ ] 代码覆盖率未降低
- [ ] 文档已更新
- [ ] 示例代码可运行

---

## 风险评估

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| 依赖包缺失 | 高 | 中 | 提供安装说明 |
| 数据源不稳定 | 中 | 低 | 使用模拟数据 |
| 性能问题 | 中 | 低 | Rust优化计划 |
| 文档不完整 | 低 | 中 | 持续完善 |

---

## 测试完成标准

1. ✅ 所有P0测试用例通过
2. ✅ 覆盖率 >= 60%
3. ✅ 关键功能验证完成
4. ✅ 文档齐全
5. ✅ 示例可运行

---

**测试负责人**: Claude (AI Assistant)
**审批**: 待用户确认
