# 测试指南

## 安装测试依赖

### 1. 创建虚拟环境（推荐）

```bash
# 使用venv创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 安装所有依赖（包括测试依赖）
pip install -r requirements.txt

# 或者只安装测试依赖
pip install pytest pytest-asyncio pytest-cov
```

## 运行测试

### 运行所有测试

```bash
pytest
```

### 运行特定测试文件

```bash
pytest tests/test_example.py
```

### 运行特定测试函数

```bash
pytest tests/test_example.py::test_sample_fixture
```

### 运行特定标记的测试

```bash
# 只运行单元测试
pytest -m unit

# 只运行集成测试
pytest -m integration

# 只运行回测相关测试
pytest -m backtest

# 排除慢速测试
pytest -m "not slow"
```

### 生成覆盖率报告

```bash
# 生成HTML覆盖率报告
pytest --cov=. --cov-report=html

# 查看报告
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### 详细输出

```bash
# 显示详细输出
pytest -v

# 显示print输出
pytest -s

# 显示简短错误信息
pytest --tb=short
```

## 测试目录结构

```
tests/
├── __init__.py          # 测试模块初始化
├── conftest.py          # pytest配置和共享fixtures
├── test_example.py      # 示例测试
├── backtest/            # 回测系统测试
├── agents/              # Agent系统测试
├── rl/                  # 强化学习测试
├── data/                # 数据模块测试
└── utils/               # 工具函数测试
```

## 编写测试

### 示例测试

```python
import pytest
from backtest.engine import BacktestEngine

@pytest.mark.unit
def test_backtest_initialization():
    """测试回测引擎初始化"""
    engine = BacktestEngine()
    assert engine is not None
    assert engine.cash == 1000000

@pytest.mark.integration
def test_backtest_run(sample_price_data):
    """测试回测运行"""
    engine = BacktestEngine()
    result = engine.run(sample_price_data)
    assert result is not None
    assert 'total_return' in result
```

### 使用Fixtures

```python
def test_with_fixture(sample_price_data):
    """使用共享fixture"""
    assert len(sample_price_data) > 0
    assert 'close' in sample_price_data.columns
```

### 参数化测试

```python
@pytest.mark.parametrize("symbol,expected", [
    ("000001.SZ", "平安银行"),
    ("600000.SH", "浦发银行"),
])
def test_symbol_mapping(symbol, expected):
    """参数化测试示例"""
    result = get_symbol_name(symbol)
    assert result == expected
```

## 测试标记

使用标记对测试进行分类：

- `@pytest.mark.unit`: 单元测试（快速运行）
- `@pytest.mark.integration`: 集成测试（可能需要外部依赖）
- `@pytest.mark.slow`: 慢速测试（运行时间较长）
- `@pytest.mark.backtest`: 回测相关测试
- `@pytest.mark.agents`: Agent相关测试
- `@pytest.mark.rl`: 强化学习相关测试
- `@pytest.mark.data`: 数据相关测试

## CI/CD集成

### GitHub Actions示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## 最佳实践

1. **保持测试独立**: 每个测试应该独立运行，不依赖其他测试
2. **使用描述性名称**: 测试名称应该清楚说明测试的内容
3. **遵循AAA模式**: Arrange（准备）-> Act（执行）-> Assert（断言）
4. **使用fixtures**: 共享测试数据和设置
5. **适当标记**: 使用测试标记方便分类运行
6. **测试边界情况**: 不仅测试正常情况，还要测试错误情况
7. **保持测试快速**: 单元测试应该快速运行
8. **模拟外部依赖**: 使用mock对象模拟外部服务

## 调试测试

### 使用pdb调试

```bash
# 在测试中添加断点
def test_with_debug():
    import pdb; pdb.set_trace()
    assert True
```

### 只运行失败的测试

```bash
# 只运行上次失败的测试
pytest --lf

# 先运行失败的测试，然后运行其他测试
pytest --ff
```

### 显示详细输出

```bash
# 显示print输出
pytest -s

# 显示本地变量
pytest -l
```

## 常见问题

### Q: 测试需要很长时间运行怎么办？

A: 使用标记跳过慢速测试：
```bash
pytest -m "not slow"
```

### Q: 如何处理需要真实数据的测试？

A: 使用 `@pytest.mark.integration` 标记，并使用mock数据进行单元测试。

### Q: 如何测试异步代码？

A: 使用 `pytest-asyncio`:
```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_func()
    assert result is not None
```

## 参考资源

- [pytest文档](https://docs.pytest.org/)
- [pytest-cov文档](https://pytest-cov.readthedocs.io/)
- [Python测试最佳实践](https://docs.python-guide.org/writing/tests/)
