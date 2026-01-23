# quantA 性能优化指南

本文档提供quantA系统的性能优化技术和最佳实践。

## 目录

1. [技术指标优化](#技术指标优化)
2. [数据处理优化](#数据处理优化)
3. [回测引擎优化](#回测引擎优化)
4. [并行处理](#并行处理)
5. [内存优化](#内存优化)
6. [使用Numba加速](#使用numba加速)
7. [使用Cython优化](#使用cython优化)
8. [常见瓶颈和解决方案](#常见瓶颈和解决方案)

---

## 技术指标优化

### 问题：循环计算的SMA

**原始代码（慢）：**
```python
def SMA_slow(data, period):
    """慢速SMA实现"""
    result = []
    for i in range(len(data)):
        if i < period - 1:
            result.append(np.nan)
        else:
            avg = sum(data[i-period+1:i+1]) / period
            result.append(avg)
    return pd.Series(result)
```

**优化方案1：使用pandas内置方法（快10-50倍）**
```python
def SMA_fast(data, period):
    """快速SMA实现"""
    return data.rolling(window=period).mean()
```

**优化方案2：使用Numba（快100-1000倍）**
```python
from numba import jit
import numpy as np

@jit(nopython=True)
def SMA_numba(data, period):
    """Numba加速SMA"""
    n = len(data)
    result = np.empty(n)
    result[:] = np.nan

    for i in range(period - 1, n):
        total = 0.0
        for j in range(i - period + 1, i + 1):
            total += data[j]
        result[i] = total / period

    return result
```

### 问题：重复计算的EMA

**原始代码（慢）：**
```python
def EMA_slow(data, period):
    result = []
    alpha = 2 / (period + 1)

    for i in range(len(data)):
        if i == 0:
            result.append(data[i])
        else:
            ema = alpha * data[i] + (1 - alpha) * result[i-1]
            result.append(ema)

    return pd.Series(result)
```

**优化方案：向量化计算**
```python
def EMA_fast(data, period):
    """快速EMA实现"""
    return data.ewm(span=period, adjust=False).mean()
```

### 技术指标缓存

```python
from functools import lru_cache

class IndicatorCache:
    """指标计算缓存"""

    def __init__(self):
        self._cache = {}

    def get_sma(self, data, period):
        """获取SMA，使用缓存"""
        cache_key = ('sma', id(data), period)

        if cache_key not in self._cache:
            self._cache[cache_key] = data.rolling(window=period).mean()

        return self._cache[cache_key]

    def clear(self):
        """清空缓存"""
        self._cache.clear()
```

---

## 数据处理优化

### 问题：频繁的数据复制

**原始代码（慢）：**
```python
def process_data_slow(df):
    # 每次操作都创建新对象
    df2 = df.copy()
    df2['returns'] = df2['close'].pct_change()
    df3 = df2.copy()
    df3['sma'] = df3['close'].rolling(20).mean()
    return df3
```

**优化方案：原地操作**
```python
def process_data_fast(df):
    """原地操作，避免复制"""
    df['returns'] = df['close'].pct_change()
    df['sma'] = df['close'].rolling(20).mean()
    return df
```

### 问题：逐行处理DataFrame

**原始代码（慢）：**
```python
def add_features_slow(df):
    for i in range(len(df)):
        if df.loc[i, 'close'] > df.loc[i, 'open']:
            df.loc[i, 'bullish'] = 1
        else:
            df.loc[i, 'bullish'] = 0
    return df
```

**优化方案：向量化操作**
```python
def add_features_fast(df):
    """向量化操作"""
    df['bullish'] = (df['close'] > df['open']).astype(int)
    return df
```

### 使用category类型节省内存

```python
def optimize_memory(df):
    """优化DataFrame内存使用"""
    # 转换字符串为category类型
    for col in df.select_dtypes(include=['object']):
        if df[col].nunique() / len(df[col]) < 0.5:  # 唯一值少于50%
            df[col] = df[col].astype('category')

    # 转换数值类型
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')

    return df
```

---

## 回测引擎优化

### 问题：频繁的事件创建

**优化方案：使用对象池**

```python
class EventPool:
    """事件对象池"""

    def __init__(self):
        self._pool = []

    def get_market_event(self, timestamp, symbol, open_, high, low, close, volume):
        """从池中获取事件对象"""
        if self._pool:
            event = self._pool.pop()
            event.timestamp = timestamp
            event.symbol = symbol
            event.open = open_
            event.high = high
            event.low = low
            event.close = close
            event.volume = volume
            return event
        else:
            return MarketEvent(timestamp, symbol, open_, high, low, close, volume)

    def return_event(self, event):
        """归还事件对象到池"""
        self._pool.append(event)
```

### 问题：订单处理慢

**优化方案：批量处理**

```python
def process_orders_batch(orders, data):
    """批量处理订单"""
    results = {}

    # 预处理：按股票分组
    orders_by_symbol = defaultdict(list)
    for order in orders:
        orders_by_symbol[order.symbol].append(order)

    # 批量处理每个股票的订单
    for symbol, symbol_orders in orders_by_symbol.items():
        market_data = data[symbol]

        # 向量化计算所有订单
        for order in symbol_orders:
            if order.direction == 'buy':
                results[order.id] = fill_buy_order(order, market_data)
            else:
                results[order.id] = fill_sell_order(order, market_data)

    return results
```

---

## 并行处理

### 使用多进程处理多个股票

```python
from multiprocessing import Pool, cpu_count
from functools import partial

def backtest_single_stock(symbol, strategy, start_date, end_date):
    """单个股票的回测"""
    engine = BacktestEngine(initial_cash=1000000)
    data = engine.generate_mock_data(symbols=[symbol],
                                    start_date=start_date,
                                    end_date=end_date)
    return engine.run(strategy, data)

def parallel_backtest(symbols, strategy, start_date, end_date):
    """并行回测多个股票"""
    num_processes = min(cpu_count(), len(symbols))

    with Pool(processes=num_processes) as pool:
        func = partial(backtest_single_stock,
                      strategy=strategy,
                      start_date=start_date,
                      end_date=end_date)

        results = pool.map(func, symbols)

    return results
```

### 使用Joblib并行化参数优化

```python
from joblib import Parallel, delayed

def optimize_parameters_parallel(strategy_class, param_grid, data, n_jobs=-1):
    """并行参数优化"""
    # 生成所有参数组合
    import itertools
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # 并行评估
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_params)(strategy_class, params, data)
        for params in combinations
    )

    return results
```

---

## 内存优化

### 使用生成器处理大数据

```python
def data_generator(symbols, start_date, end_date, batch_size=10):
    """分批生成数据"""
    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]

        engine = BacktestEngine(initial_cash=1000000)
        batch_data = engine.generate_mock_data(
            symbols=batch_symbols,
            start_date=start_date,
            end_date=end_date,
        )

        yield batch_data

        # 显式删除以释放内存
        del batch_data

# 使用生成器
for batch_data in data_generator(all_symbols, '2023-01-01', '2023-12-31'):
    process_batch(batch_data)
```

### 及时释放大对象

```python
def process_large_dataset(data):
    """处理大数据集"""
    # 处理数据
    result = expensive_computation(data)

    # 及时删除大对象
    del data

    # 强制垃圾回收
    import gc
    gc.collect()

    return result
```

---

## 使用Numba加速

### JIT编译关键计算

```python
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def calculate_returns_numba(prices):
    """Numba加速收益率计算"""
    n = len(prices)
    returns = np.empty(n)
    returns[0] = np.nan

    for i in range(1, n):
        returns[i] = (prices[i] - prices[i-1]) / prices[i-1]

    return returns

@jit(nopython=True, cache=True)
def calculate_sharpe_numba(returns, risk_free_rate=0.0):
    """Numba加速夏普比率计算"""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)
```

### 并行化Numba函数

```python
from numba import prange

@jit(nopython=True, parallel=True)
def calculate_indicators_parallel(data):
    """并行计算多个指标"""
    n = len(data)
    sma_20 = np.empty(n)
    sma_50 = np.empty(n)

    for i in prange(n):
        if i >= 19:
            sma_20[i] = np.mean(data[i-19:i+1])
        if i >= 49:
            sma_50[i] = np.mean(data[i-49:i+1])

    return sma_20, sma_50
```

---

## 使用Cython优化

### Cython化关键函数

**创建文件 `indicators.pyx`:**

```cython
# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def SMA_cython(double[:] data, int period):
    """Cython加速SMA"""
    cdef int n = data.shape[0]
    cdef double[:] result = np.empty(n)
    cdef int i, j
    cdef double total

    for i in range(n):
        if i < period - 1:
            result[i] = np.nan
        else:
            total = 0.0
            for j in range(i - period + 1, i + 1):
                total += data[j]
            result[i] = total / period

    return np.asarray(result)
```

**编译和安装:**

```bash
# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("indicators.pyx"),
    include_dirs=[np.get_include()]
)

# 编译
python setup.py build_ext --inplace
```

---

## 常见瓶颈和解决方案

### 1. 数据加载慢

**问题：** 从CSV/数据库加载数据慢

**解决方案：**
- 使用二进制格式（pickle, parquet, HDF5）
- 压缩数据
- 只加载需要的列

```python
# 使用Parquet格式（快10-20倍）
data.to_parquet('data.parquet', compression='snappy')
data = pd.read_parquet('data.parquet')

# 只加载需要的列
data = pd.read_csv('data.csv', usecols=['datetime', 'open', 'high', 'low', 'close', 'volume'])
```

### 2. 指标重复计算

**问题：** 相同指标被多次计算

**解决方案：**
- 缓存计算结果
- 增量计算

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_indicator(data_id, indicator_name, period):
    """缓存指标计算"""
    # 计算...
    return result
```

### 3. 内存泄漏

**问题：** 长时间运行后内存占用过高

**解决方案：**
- 及时删除大对象
- 使用弱引用
- 监控内存使用

```python
import weakref
import gc

class DataManager:
    def __init__(self):
        self._cache = weakref.WeakValueDictionary()

    def cleanup(self):
        """清理缓存"""
        self._cache.clear()
        gc.collect()
```

### 4. 字符串操作慢

**问题：** 频繁的字符串处理

**解决方案：**
- 使用category类型
- 预处理字符串
- 使用数值编码

```python
# 字符串转数值
symbol_to_id = {symbol: i for i, symbol in enumerate(symbols)}
df['symbol_id'] = df['symbol'].map(symbol_to_id)

# 使用category
df['symbol'] = df['symbol'].astype('category')
```

---

## 性能分析工具

### 使用cProfile

```python
import cProfile
import pstats

def profile_backtest():
    """分析回测性能"""
    profiler = cProfile.Profile()
    profiler.enable()

    # 运行回测
    engine.run(strategy, data)

    profiler.disable()

    # 打印结果
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # 打印前20个最慢的函数
```

### 使用line_profiler

```bash
# 安装
pip install line_profiler

# 使用
@profile
def slow_function():
    # code
    pass

# 运行
kernprof -l -v script.py
```

### 使用memory_profiler

```bash
# 安装
pip install memory_profiler

# 使用
from memory_profiler import profile

@profile
def memory_intensive_function():
    # code
    pass
```

---

## 性能优化清单

- [ ] 使用向量化操作替代循环
- [ ] 缓存重复计算的结果
- [ ] 使用适当的数据类型（category, int8 vs int64）
- [ ] 避免不必要的数据复制
- [ ] 使用并行处理（multiprocessing, joblib）
- [ ] 使用Numba JIT编译关键代码
- [ ] 考虑使用Cython优化热点
- [ ] 使用生成器处理大数据集
- [ ] 及时释放大对象
- [ ] 使用高效的文件格式（parquet, pickle）
- [ ] 优化数据库查询
- [ ] 使用连接池
- [ ] 考虑使用Rust/C++扩展

---

## 进一步优化

### 使用Rust扩展

参考 `rust_engine/` 目录，将性能关键代码用Rust重写。

### 使用GPU加速

对于某些计算密集型任务，可以使用GPU加速：

```python
import cupy as cp

def calculate_returns_gpu(prices):
    """GPU加速收益率计算"""
    prices_gpu = cp.array(prices)
    returns_gpu = cp.diff(prices_gpu) / prices_gpu[:-1]
    return cp.asnumpy(returns_gpu)
```

---

## 总结

性能优化是一个持续的过程。建议的优化顺序：

1. **首先分析**：找到真正的瓶颈
2. **算法优化**：改进算法和数据结构
3. **向量化**：使用NumPy/Pandas向量化操作
4. **并行化**：利用多核CPU
5. **JIT编译**：使用Numba加速
6. **扩展**：使用Cython/Rust编写扩展

记住："过早优化是万恶之源" - 先确保代码正确，再优化性能。
