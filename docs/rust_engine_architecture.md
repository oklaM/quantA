# Rust执行引擎架构设计

## 1. 概述

### 1.1 目标
使用Rust实现高性能回测执行引擎，解决Python回测速度慢的问题，提供：
- **10-100倍性能提升**：相比纯Python实现
- **内存安全**：Rust的零成本抽象和所有权系统
- **并发支持**：多线程并行回测
- **易用性**：Python FFI接口无缝集成

### 1.2 适用场景
- 大规模历史数据回测（10年+数据）
- 参数优化和网格搜索
- 多策略并行回测
- 实时交易信号计算
- 高频策略回测

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        Python Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   策略定义    │  │   数据接口    │  │   结果分析    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────────┬─────────────────────────────────┘
                            │ PyO3 FFI
┌───────────────────────────┴─────────────────────────────────┐
│                        Rust Core                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Event Bus (事件总线)                     │  │
│  │  - 事件队列管理                                       │  │
│  │  - 事件路由分发                                       │  │
│  │  - 时间线管理                                         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │Execution │  │Portfolio │  │  Risk    │  │ Metrics  │  │
│  │  Engine  │  │ Manager  │  │ Manager  │  │ Collector│  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Data Management (数据管理)                  │  │
│  │  - OHLCV数据存储                                      │  │
│  │  - 指标缓存                                           │  │
│  │  - 内存池管理                                         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块设计

#### 2.2.1 核心模块

**Event Bus (事件总线)**
```rust
pub struct EventBus {
    queue: RwLock<VecDeque<Event>>,
    subscribers: HashMap<EventType, Vec<SubscriberId>>,
    timeline: BTreeMap<DateTime, Vec<Event>>,
}

impl EventBus {
    pub fn publish(&self, event: Event) -> Result<()>;
    pub fn subscribe(&mut self, event_type: EventType) -> SubscriberId;
    pub fn next_event(&self) -> Option<Event>;
    pub fn advance_time(&mut self, dt: DateTime) -> Result<()>;
}
```

**Execution Engine (执行引擎)**
```rust
pub struct ExecutionEngine {
    event_bus: Arc<EventBus>,
    portfolio: Arc<Portfolio>,
    risk_mgr: Arc<RiskManager>,
    metrics: Arc<MetricsCollector>,
}

impl ExecutionEngine {
    pub fn run_backtest(&mut self, data: &[BarData]) -> Result<BacktestResult>;
    pub fn process_bar(&mut self, bar: &BarData) -> Result<()>;
    pub fn handle_order(&mut self, order: Order) -> Result<Fill>;
}
```

**Portfolio Manager (组合管理)**
```rust
pub struct Portfolio {
    cash: Atomic<f64>,
    positions: RwLock<HashMap<Symbol, Position>>,
    orders: RwLock<Vec<Order>>,
    transactions: Vec<Transaction>,
}

impl Portfolio {
    pub fn update_position(&mut self, symbol: Symbol, qty: i64, price: f64) -> Result<()>;
    pub fn get_total_value(&self, prices: &HashMap<Symbol, f64>) -> f64;
    pub fn check_margin(&self, order: &Order) -> bool;
}
```

#### 2.2.2 数据结构

**事件类型**
```rust
pub enum Event {
    Bar(BarEvent),
    Order(OrderEvent),
    Fill(FillEvent),
    Signal(SignalEvent),
    Control(ControlEvent),
}

pub struct BarEvent {
    pub timestamp: DateTime,
    pub symbol: Symbol,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
```

**订单系统**
```rust
pub struct Order {
    pub id: OrderId,
    pub symbol: Symbol,
    pub side: Side,
    pub quantity: i64,
    pub order_type: OrderType,
    pub price: Option<f64>,
    pub status: OrderStatus,
    pub created_at: DateTime,
    pub filled_at: Option<DateTime>,
}

pub enum Side { Buy, Sell }
pub enum OrderType { Market, Limit, StopLimit }
pub enum OrderStatus { Pending, PartiallyFilled, Filled, Cancelled, Rejected }
```

## 3. 性能优化策略

### 3.1 内存管理
- **对象池**: 复用事件对象，减少内存分配
- **栈分配**: 小对象使用栈分配而非堆
- **Arena分配器**: 批量分配，统一释放

```rust
use object_pool::{Pool, Reusable};

pub struct EventPool {
    pool: Pool<BarEvent>,
}

impl EventPool {
    pub fn acquire_event(&self) -> Reusable<BarEvent> {
        self.pool.pull()
    }
}
```

### 3.2 并发处理
- **多线程事件处理**: 不同股票并行处理
- **无锁数据结构**: Atomic/RwLock减少锁竞争
- **工作窃取**: Rayon并行迭代器

```rust
use rayon::prelude::*;

pub fn process_bars_parallel(&mut self, bars: &[BarData]) -> Result<()> {
    bars.par_chunks(1000)
        .for_each(|chunk| {
            chunk.iter().for_each(|bar| {
                self.process_bar_single_threaded(bar);
            });
        });
    Ok(())
}
```

### 3.3 SIMD优化
- **指标计算**: 使用SIMD加速技术指标计算
- **批量操作**: 向量化价格数据处理

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn calculate_sma_simd(prices: &[f64], period: usize) -> Vec<f64> {
    // AVX2加速的移动平均计算
}
```

## 4. Python集成

### 4.1 PyO3绑定

**使用PyO3创建Python接口**

```rust
use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyclass]
pub struct RustBacktestEngine {
    engine: ExecutionEngine,
}

#[pymethods]
impl RustBacktestEngine {
    #[new]
    pub fn new(
        initial_cash: f64,
        commission_rate: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            engine: ExecutionEngine::new(initial_cash, commission_rate)?,
        })
    }

    pub fn run_backtest(
        &mut self,
        data: &PyList,
        strategy: &PyAny,
    ) -> PyResult<PyObject> {
        // 从Python接收数据
        let bars = parse_py_data(data)?;

        // 运行回测
        let result = self.engine.run_backtest(&bars)?;

        // 转换结果为Python对象
        Ok(to_py_object(result))
    }

    pub fn get_equity_curve(&self) -> PyResult<Vec<f64>> {
        Ok(self.engine.get_equity_curve())
    }
}
```

### 4.2 编译配置

**Cargo.toml**
```toml
[package]
name = "quanta-rust-engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "quanta_rust_engine"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
rayon = "1.8"
chrono = "0.4"
serde = { version = "1.0", features = ["derive"] }
ndarray = "0.15"
polars = "0.35"  # 高性能DataFrame
numpy = "0.20"

[dependencies.object-pool]
version = "0.5"
```

**maturin配置**
```toml
[tool.maturin]
python-source = "python"
module-name = "quanta._rust"
```

## 5. 指标库实现

### 5.1 技术指标

```rust
pub mod indicators {
    use std::collections::VecDeque;

    pub struct SMA {
        period: usize,
        values: VecDeque<f64>,
        sum: f64,
    }

    impl SMA {
        pub fn new(period: usize) -> Self {
            Self {
                period,
                values: VecDeque::with_capacity(period),
                sum: 0.0,
            }
        }

        pub fn update(&mut self, price: f64) -> Option<f64> {
            self.values.push_back(price);
            self.sum += price;

            if self.values.len() > self.period {
                let removed = self.values.pop_front().unwrap();
                self.sum -= removed;
            }

            if self.values.len() == self.period {
                Some(self.sum / self.period as f64)
            } else {
                None
            }
        }
    }

    // 其他指标: EMA, MACD, RSI, Bollinger Bands, etc.
}
```

### 5.2 指标缓存

```rust
pub struct IndicatorCache {
    sma_cache: HashMap<(Symbol, usize), Vec<f64>>,
    rsi_cache: HashMap<(Symbol, usize), Vec<f64>>,
}

impl IndicatorCache {
    pub fn get_or_compute_sma(
        &mut self,
        symbol: Symbol,
        prices: &[f64],
        period: usize,
    ) -> &[f64] {
        // 检查缓存
        let key = (symbol, period);
        if !self.sma_cache.contains_key(&key) {
            let values = compute_sma(prices, period);
            self.sma_cache.insert(key, values);
        }

        &self.sma_cache[&key]
    }
}
```

## 6. 实现路线图

### Phase 1: 核心框架 (2-3周)
- [ ] 基础数据结构定义
- [ ] 事件总线实现
- [ ] 简单回测流程
- [ ] Python FFI接口

### Phase 2: 完整功能 (3-4周)
- [ ] 订单管理系统
- [ ] 持仓管理
- [ ] 风险控制
- [ ] 技术指标库

### Phase 3: 性能优化 (2-3周)
- [ ] 并行处理
- [ ] SIMD优化
- [ ] 内存优化
- [ ] 性能基准测试

### Phase 4: 集成测试 (1-2周)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 与Python对比验证
- [ ] 文档完善

## 7. 预期性能提升

### 7.1 基准测试场景

| 场景 | Python时间 | Rust预期 | 加速比 |
|------|-----------|---------|--------|
| 10年单股票回测 | 5s | 0.5s | 10x |
| 100次参数优化 | 500s | 10s | 50x |
| 100股票组合回测 | 200s | 2s | 100x |
| 1分钟K线5年回测 | 50s | 1s | 50x |

### 7.2 内存使用

- **Python**: ~500MB (大量对象开销)
- **Rust**: ~50MB (紧凑内存布局)
- **减少**: 90%内存占用

## 8. 使用示例

### 8.1 Python调用

```python
from quanta._rust import RustBacktestEngine
from quanta.data import load_bars

# 创建Rust引擎
engine = RustBacktestEngine(
    initial_cash=1_000_000.0,
    commission_rate=0.0003,
)

# 加载数据
bars = load_bars('000001.SZ', start='2020-01-01', end='2023-12-31')

# 运行回测
result = engine.run_backtest(bars, strategy)

# 获取结果
equity_curve = engine.get_equity_curve()
returns = result['total_return']
sharpe = result['sharpe_ratio']
```

### 8.2 混合模式

```python
# Python策略 + Rust执行
from quanta._rust import RustExecutionEngine

class MyStrategy:
    def on_bar(self, bar):
        # Python策略逻辑
        if self.should_buy(bar):
            self.engine.submit_order(
                symbol=bar.symbol,
                side='BUY',
                quantity=1000,
            )

# Rust执行引擎执行
engine = RustExecutionEngine()
engine.register_strategy(MyStrategy())
engine.run()
```

## 9. 注意事项

### 9.1 开发环境
- Rust 1.70+
- Python 3.9+
- Maturin (PyO3打包工具)
- Cargo (Rust包管理器)

### 9.2 调试建议
- 使用`cargo expand`查看宏展开
- 使用`valgrind`检查内存泄漏
- 使用`flamegraph`分析性能瓶颈

### 9.3 部署
```bash
# 编译Rust扩展
maturin develop --release

# 运行测试
pytest tests/test_rust_engine.py

# 性能测试
python benchmarks/rust_vs_python.py
```

## 10. 相关资源

- [PyO3文档](https://pyo3.rs/)
- [Rust性能指南](https://nnethercote.github.io/perf-book/)
- [Rayon并行库](https://github.com/rayon-rs/rayon)
- [Maturin打包工具](https://github.com/PyO3/maturin)
