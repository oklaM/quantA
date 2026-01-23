# 数据模块使用指南

## 概述

quantA的数据模块提供了完整的A股市场数据获取、存储和管理功能。支持多种数据源和存储方式。

## 数据源

### 1. AKShare（推荐，免费）

**优点：**
- 完全免费
- 无需token注册
- 数据覆盖全面
- 社区活跃

**使用示例：**
```python
from data.market.sources.akshare_provider import AKShareProvider

# 创建数据源
provider = AKShareProvider()
provider.connect()

# 获取日线数据
df = provider.get_daily_bar(
    symbol="000001.SZ",
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"  # 前复权
)

provider.disconnect()
```

### 2. Tushare

**优点：**
- 数据质量高
- 接口稳定
- 数据全面

**缺点：**
- 需要token（免费额度有限）
- 高级功能需要付费

**使用示例：**
```python
from data.market.sources.tushare_provider import TushareProvider

# 创建数据源（需要token）
provider = TushareProvider(token="your_token_here")
provider.connect()

# 获取日线数据
df = provider.get_daily_bar(
    symbol="600519.SH",
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"
)

provider.disconnect()
```

**获取token：**
1. 访问 https://tushare.pro/register
2. 注册账号
3. 在个人中心获取token
4. 将token添加到 `.env` 文件：`TUSHARE_TOKEN=your_token_here`

## 数据存储

### DuckDB（默认，推荐）

**优点：**
- 轻量级
- 无需安装服务器
- 性能优秀
- 支持SQL查询

**配置：**
```python
# config/settings.py
database:
  DUCKDB_PATH: "data/market.db"  # 数据库文件路径
```

### ClickHouse（可选）

**优点：**
- 高性能时序数据库
- 适合海量数据

**缺点：**
- 需要安装服务器
- 配置复杂

## 数据采集

### 基本使用

```python
from data.market.collector import create_collector

# 创建采集器
collector = create_collector(provider="akshare")

# 采集股票列表
collector.collect_stock_list()

# 采集日线数据
symbols = ["000001.SZ", "600000.SH"]
collector.collect_daily_bar(
    symbols=symbols,
    start_date="20240101",
    end_date="20241231",
    adjust="qfq"
)

# 从数据库读取数据
df = collector.get_daily_bar(
    symbol="000001.SZ",
    start_date="20240101",
    end_date="20241231"
)

# 关闭采集器
collector.close()
```

### 增量更新

```python
# 更新最近5天的数据
collector.update_daily_data(symbols=symbols, days=5)

# 更新今日数据
collector.update_today_data(symbols=symbols)
```

## 数据格式

### 日线数据格式

```python
# DataFrame列名
{
    'symbol': '000001.SZ',      # 股票代码
    'date': '2024-01-01',       # 日期
    'open': 10.50,              # 开盘价
    'high': 10.80,              # 最高价
    'low': 10.40,               # 最低价
    'close': 10.75,             # 收盘价
    'volume': 1000000,          # 成交量（股）
    'amount': 10750000.0        # 成交额（元）
}
```

### 股票列表格式

```python
{
    'symbol': '000001.SZ',      # 股票代码
    'name': '平安银行',          # 股票名称
    'market': 'SZ',             # 市场（SH/SZ）
    'industry': '银行',          # 行业
    'list_date': '19910403'     # 上市日期
}
```

## 完整示例

### 示例1：获取数据并回测

```python
from data.market.sources.akshare_provider import AKShareProvider
from backtest.engine.backtest import BacktestEngine
from backtest.engine.strategy import BuyAndHoldStrategy
from datetime import datetime, timedelta

# 1. 获取数据
provider = AKShareProvider()
provider.connect()

symbol = "000001.SZ"
end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")

df = provider.get_daily_bar(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    adjust="qfq"
)
provider.disconnect()

# 2. 转换数据格式
df = df.rename(columns={'date': 'datetime'})
data = {symbol: df}

# 3. 运行回测
strategy = BuyAndHoldStrategy(symbol=symbol, quantity=1000)

engine = BacktestEngine(
    data=data,
    strategy=strategy,
    initial_cash=1000000.0,
)

results = engine.run()

# 4. 查看结果
print(f"总收益率: {results['account']['total_return_pct']:.2f}%")
```

### 示例2：批量采集并存储

```python
from data.market.collector import create_collector
from datetime import datetime, timedelta

# 创建采集器
collector = create_collector(provider="akshare")

# 1. 采集股票列表
collector.collect_stock_list()

# 2. 获取所有股票代码
symbols = collector.get_all_symbols()
print(f"共有 {len(symbols)} 只股票")

# 3. 采集部分股票数据
symbols_to_collect = symbols[:50]  # 采集前50只
end_date = datetime.now().strftime("%Y%m%d")
start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

collector.collect_daily_bar(
    symbols=symbols_to_collect,
    start_date=start_date,
    end_date=end_date
)

# 4. 验证数据
for symbol in symbols_to_collect[:5]:
    df = collector.get_daily_bar(symbol=symbol)
    print(f"{symbol}: {len(df)} 条数据")

collector.close()
```

## 定时任务

### 使用APScheduler定时更新数据

```python
from apscheduler.schedulers.blocking import BlockingScheduler
from data.market.collector import create_collector

def update_job():
    """定时更新任务"""
    collector = create_collector(provider="akshare")

    # 获取所有股票
    symbols = collector.get_all_symbols()

    # 更新最近5天数据
    collector.update_daily_data(symbols=symbols, days=5)

    collector.close()
    print(f"数据更新完成: {datetime.now()}")

# 创建调度器
scheduler = BlockingScheduler()

# 每天收盘后更新（15:30）
scheduler.add_job(update_job, 'cron', hour=15, minute=30)

# 每周一早上更新所有数据
scheduler.add_job(update_job, 'cron', day_of_week='mon', hour=9, minute=0)

print("调度器已启动...")
scheduler.start()
```

## 性能优化

### 1. 批量获取

```python
# 一次性获取多只股票
symbols = ["000001.SZ", "000002.SZ", "600000.SH"]

for symbol in symbols:
    df = provider.get_daily_bar(symbol=symbol, ...)
```

### 2. 并发采集

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_symbol(symbol):
    provider = AKShareProvider()
    provider.connect()
    df = provider.get_daily_bar(symbol=symbol, ...)
    provider.disconnect()
    return df

symbols = ["000001.SZ", "000002.SZ", "600000.SH"]

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(fetch_symbol, symbols))
```

### 3. 数据压缩

DuckDB会自动压缩数据，无需额外配置。

## 常见问题

### Q1: AKShare数据获取失败？

**A:** 检查以下几点：
1. 网络连接是否正常
2. akshare包是否已更新：`pip install --upgrade akshare`
3. 股票代码格式是否正确（如"000001.SZ"或"000001"）

### Q2: Tushare提示额度不足？

**A:**
- 免费账户有积分限制
- 可以通过签到、分享等方式获取积分
- 或使用AKShare作为替代

### Q3: 数据库文件太大？

**A:**
1. 只采集需要的时间范围
2. 定期清理旧数据
3. 考虑使用ClickHouse等高性能数据库

### Q4: 如何获取实时行情？

**A:**
```python
provider = AKShareProvider()
provider.connect()

# 获取实时行情
df = provider.get_realtime_quote(symbols=["000001", "600000"])

print(df[['symbol', 'price', 'change_pct']])

provider.disconnect()
```

### Q5: 复权类型选择？

**A:**
- `qfq`（前复权）：适合回测，保证价格连续性
- `hfq`（后复权）：适合分析当前价格
- `""`（不复权）：查看历史真实价格

## 最佳实践

1. **数据源选择**：
   - 开发测试：使用AKShare（免费）
   - 生产环境：使用Tushare（质量高）

2. **存储选择**：
   - 小规模数据：DuckDB（简单）
   - 大规模数据：ClickHouse（性能好）

3. **更新策略**：
   - 日线数据：每天收盘后更新一次
   - 分钟线：按需采集
   - 股票列表：每周更新一次

4. **数据验证**：
   - 检查数据完整性
   - 验证数据连续性
   - 监控异常数据

## 更多资源

- AKShare文档：https://akshare.akfamily.xyz
- Tushare文档：https://tushare.pro/document/2
- DuckDB文档：https://duckdb.org/docs
- 项目示例：`examples/data_collection_example.py`
