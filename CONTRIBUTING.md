# quantA 贡献指南

感谢您对 quantA 项目的关注！我们欢迎任何形式的贡献，包括但不限于代码提交、Bug 报告、功能建议、文档改进等。

本指南将帮助您了解如何为 quantA 项目做出贡献。

---

## 目录

- [行为准则](#行为准则)
- [如何报告 Bug](#如何报告-bug)
- [功能请求](#功能请求)
- [开发环境设置](#开发环境设置)
- [代码贡献流程](#代码贡献流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [提交信息规范](#提交信息规范)
- [代码审查标准](#代码审查标准)
- [项目结构说明](#项目结构说明)
- [常见问题](#常见问题)

---

## 行为准则

### 我们的承诺

为了营造开放和友好的环境，我们承诺让每个人都能参与到项目中来，无论其经验水平、性别、性别认同和表达、性取向、残疾、个人外貌、体型、种族、民族、年龄、宗教或国籍如何。

### 我们的标准

积极行为包括：
- 使用包容性语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

不可接受的行为包括：
- 使用性别化语言或图像，以及不受欢迎的性关注或过度关注
- 恶意攻击、侮辱/贬损的评论，以及个人或政治攻击
- 公开或私下骚扰
- 未经明确许可发布他人的私人信息
- 其他在专业场合可能被合理认为不适当的行为

### 责任

项目维护者负责阐明可接受行为的标准，并应对任何不可接受的行为采取适当和公平的纠正措施。

---

## 如何报告 Bug

Bug 报告对于帮助我们改进项目非常重要。请按照以下步骤提交高质量的 Bug 报告：

### 1. 搜索现有的 Issues

在提交新的 Bug 报告之前，请先搜索现有的 Issues，看看是否已经有人报告了相同或类似的问题。

### 2. 使用 Bug 报告模板

创建新 Issue 时，请使用 Bug 报告模板并填写以下信息：

```markdown
### Bug 描述
简要描述遇到的问题。

### 复现步骤
1. 执行的操作 '...'
2. 点击按钮 '....'
3. 滚动到 '....'
4. 看到错误

### 期望行为
清晰简洁地描述您期望发生的行为。

### 实际行为
清晰简洁地描述实际发生的行为。

### 环境信息
- OS: [例如 macOS 12.0, Ubuntu 20.04]
- Python 版本: [例如 3.10.0]
- quantA 版本: [例如 0.1.0]
- 相关依赖版本: [例如 pandas 2.0.0]

### 日志输出
如果适用，请粘贴相关的日志输出。

### 附加信息
- 截图（如果有帮助）
- 代码片段（如果有帮助）
- 任何其他相关信息
```

### 3. 提供最小可复现示例

如果可能，请提供一个最小可复现示例，这将大大加快我们解决问题的速度。

```python
# 示例：最小可复现代码
from backtest.engine import BacktestEngine

# 创建引擎
engine = BacktestEngine(
    initial_cash=100000,
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 运行回测
result = engine.run(strategy='my_strategy')
# 这里出现了错误：...
```

---

## 功能请求

我们欢迎功能请求！但在提交之前，请考虑以下几点：

### 1. 检查是否已存在

搜索现有的 Issues，看看是否已经有人提出了类似的功能请求。

### 2. 使用功能请求模板

```markdown
### 功能描述
清晰简洁地描述您希望添加的功能。

### 问题或背景
这个功能解决什么问题？为什么需要它？
请尽可能提供真实的用例。

### 期望的解决方案
描述您希望如何实现这个功能。

### 替代方案
描述您考虑过的任何替代解决方案或功能。

### 附加信息
如果有截图、代码示例或其他相关信息，请在此添加。
```

### 3. 提供用例

详细描述这个功能的实际应用场景，帮助我们理解它的重要性。

---

## 开发环境设置

### 系统要求

- **操作系统**: Linux, macOS, Windows (推荐 Linux 或 macOS)
- **Python**: 3.9 或更高版本（推荐 3.10+）
- **Rust**: 1.70.0 或更高版本（可选，用于开发 Rust 引擎）
- **Git**: 最新版本
- **内存**: 至少 8GB RAM
- **磁盘空间**: 至少 5GB 可用空间

### 详细设置步骤

#### 1. Fork 并克隆仓库

```bash
# 1. 在 GitHub 上 Fork quantA 仓库
# 访问 https://github.com/your-username/quanta/fork

# 2. 克隆您的 Fork
git clone https://github.com/your-username/quanta.git

# 3. 进入项目目录
cd quanta

# 4. 添加上游远程仓库（用于同步最新更改）
git remote add upstream https://github.com/original-owner/quanta.git
```

#### 2. 创建虚拟环境

我们强烈推荐使用虚拟环境来隔离项目依赖。

```bash
# 使用 venv 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# 或者使用 Makefile（推荐）
make venv
source venv/bin/activate
```

#### 3. 安装依赖

```bash
# 安装所有依赖（包括开发依赖）
make install-dev

# 或手动安装
pip install --upgrade pip
pip install -r requirements.txt

# 安装开发工具
pip install pytest pytest-asyncio pytest-cov pytest-mock
pip install black isort flake8 mypy pylint
```

#### 4. 配置环境变量

创建 `.env` 文件并配置必要的环境变量：

```bash
# 创建 .env 文件
cp .env.example .env

# 编辑 .env 文件，添加您的 API 密钥
# ZHIPUAI_API_KEY=your_zhipuai_api_key_here
# TUSHARE_TOKEN=your_tushare_token_here
# OPENAI_API_KEY=your_openai_api_key_here  # 可选
```

#### 5. 设置 Python 路径

quantA 项目需要从根目录导入模块，因此需要设置 PYTHONPATH：

```bash
# 临时设置（当前终端会话）
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export PYTHONPATH="${PYTHONPATH}:/path/to/quanta"' >> ~/.bashrc
source ~/.bashrc
```

#### 6. 验证安装

```bash
# 运行测试确保一切正常
make test

# 运行格式检查
make format-check

# 运行代码检查
make lint

# 运行示例程序
python examples/backtest_example.py
```

#### 7. 配置 Git Hooks（可选）

安装 pre-commit 钩子以在提交前自动运行代码检查：

```bash
pip install pre-commit

# 安装 git hooks
pre-commit install

# 手动运行所有 hooks
pre-commit run --all-files
```

创建 `.pre-commit-config.yaml`（如果不存在）：

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/pycqa/isort
    rev: 5.13.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ["--max-line-length=127"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

#### 8. 配置 IDE

我们推荐使用以下 IDE 进行开发：

**VS Code**
安装推荐的扩展：
- Python
- Pylance
- Black Formatter
- isort
- pytest

创建 `.vscode/settings.json`：

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true
    }
}
```

**PyCharm**
- Settings → Tools → Black → Enable Black Formatter
- Settings → Tools → External Tools → Add isort
- Settings → Tools → Python Integrated Tools → Testing → Pytest

#### 9. Rust 开发环境（可选）

如果您计划开发 Rust 引擎：

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装 PyO3 和 maturin
pip install maturin

# 构建并测试 Rust 引擎
cd rust_engine
cargo build --release
cargo test --release

# 构建 Python 扩展
maturin develop --release
```

---

## 代码贡献流程

### 1. 选择任务

查看我们的 [Issues](https://github.com/your-username/quanta/issues) 页面：
- 标记为 `good first issue` 的任务适合新手
- 标记为 `help wanted` 的任务需要社区帮助
- 标记为 `enhancement` 的是新功能请求

### 2. 创建功能分支

```bash
# 确保您的本地仓库是最新的
git checkout master
git pull upstream master

# 创建新的功能分支
git checkout -b feature/your-feature-name

# 或者修复 Bug
git checkout -b fix/bug-description

# 或者文档改进
git checkout -b docs/documentation-update
```

**分支命名规范**：
- `feature/` - 新功能
- `fix/` - Bug 修复
- `hotfix/` - 紧急修复
- `docs/` - 文档更新
- `refactor/` - 代码重构
- `test/` - 测试相关
- `perf/` - 性能优化

### 3. 编写代码

在您的分支上进行开发，遵循我们的代码规范（见下文）。

### 4. 测试您的更改

```bash
# 运行所有测试
make test

# 运行特定模块的测试
pytest tests/backtest/test_indicators.py -v

# 运行测试并生成覆盖率报告
make test-cov

# 检查代码格式
make format-check

# 格式化代码
make format

# 运行代码检查
make lint
```

### 5. 提交您的更改

```bash
# 查看更改的文件
git status

# 添加文件到暂存区
git add path/to/file.py
# 或添加所有更改
git add .

# 提交更改（使用语义化提交信息）
git commit -m "feat: 添加新的技术指标计算函数"

# 查看提交历史
git log --oneline -5
```

### 6. 同步上游更改

在提交 Pull Request 之前，确保您的分支与上游仓库保持同步：

```bash
# 获取上游更改
git fetch upstream

# 合并上游的 master 分支到您的分支
git rebase upstream/master

# 如果有冲突，解决冲突后：
git add .
git rebase --continue

# 推送到您的 Fork
git push origin feature/your-feature-name
```

### 7. 创建 Pull Request

1. 访问您在 GitHub 上的 Fork 页面
2. 点击 "Compare & pull request" 按钮
3. 填写 PR 模板：

```markdown
### 描述
简要描述此 PR 的更改内容和目的。

### 更改类型
- [ ] Bug 修复（不破坏现有功能的修复）
- [ ] 新功能（不破坏现有功能的新增功能）
- [ ] 破坏性更改（会导致现有功能无法正常工作的修复或功能）
- [ ] 文档更新
- [ ] 代码重构
- [ ] 性能优化
- [ ] 测试相关

### 相关 Issue
关闭 #(issue number)

### 测试
描述您如何测试这些更改：
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 手动测试通过
- [ ] 添加了新的测试用例

### 截图（如果适用）
添加截图以展示更改效果。

### 检查清单
- [ ] 我的代码遵循此项目的代码规范
- [ ] 我已执行自我审查
- [ ] 我已对我的代码进行了注释，特别是在难以理解的区域
- [ ] 我已对文档进行了相应的更改
- [ ] 我的更改不会产生新的警告
- [ ] 我已添加了测试以证明我的修复有效或我的功能有效
- [ ] 新的和现有的单元测试都在本地通过
- [ ] 任何依赖的更改都已合并和发布
```

4. 等待代码审查和维护者的反馈

### 8. 处理审查反馈

维护者可能会要求您进行修改。请：

1. 在您的分支上进行修改
2. 提交更改
3. 推送到您的 Fork
4. PR 会自动更新

```bash
# 进行修改
# ... 编辑代码 ...

# 提交修改
git add .
git commit -m "fix: 根据审查反馈修改代码"

# 推送更新
git push origin feature/your-feature-name
```

### 9. 合并后清理

一旦您的 PR 被合并：

```bash
# 切换到 master 分支
git checkout master

# 拉取最新的更改
git pull upstream master

# 删除本地分支
git branch -d feature/your-feature-name

# 删除远程分支
git push origin --delete feature/your-feature-name
```

---

## 代码规范

### Python 代码风格

我们遵循 PEP 8 编码规范，并使用以下工具强制执行：

#### 1. Black - 代码格式化

[Black](https://github.com/psf/black) 是我们使用的代码格式化工具。

```bash
# 格式化代码
black .

# 检查格式（不修改文件）
black --check .

# 配置选项在 pyproject.toml 中
```

**Black 配置规则**：
- 行长度：127 字符（不是默认的 88）
- 双引号优先
- 不使用 trailing commas（除非在括号内）

#### 2. isort - 导入排序

[isort](https://github.com/pycqa/isort) 用于排序和格式化导入。

```bash
# 排序导入
isort .

# 检查导入排序（不修改文件）
isort --check-only .

# 配置文件：.isort.cfg
```

**isort 配置**（`.isort.cfg`）：
```ini
[settings]
profile = black
line_length = 127
known_first_party = agents,backtest,config,data,live,rl,trading,utils
known_third_party = pandas,numpy,torch,stable_baselines3,gymnasium,langchain
skip = .git,__pycache__,venv,.venv,env,build,dist
```

#### 3. Flake8 - 代码风格检查

[Flake8](https://flake8.pycqa.org/) 用于检查代码风格和编程错误。

```bash
# 运行 flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# 只报告严重错误
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

**忽略的警告**（在 `.flake8` 或 `setup.cfg` 中配置）：
```ini
[flake8]
max-line-length = 127
extend-ignore = E203, W503
exclude = .git,__pycache__,venv,.venv,env,build,dist
```

#### 4. mypy - 类型检查

[mypy](https://mypy.readthedocs.io/) 用于静态类型检查。

```bash
# 运行 mypy
mypy .

# 只检查特定模块
mypy backtest/engine/

# 配置文件：mypy.ini 或 pyproject.toml
```

**mypy 配置示例**（`mypy.ini`）：
```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-numpy.*]
ignore_missing_imports = True
```

#### 5. pylint - 代码质量检查

[pylint](https://pylint.pycqa.org/) 用于查找代码中的错误和异味。

```bash
# 运行 pylint
pylint backtest/

# 只显示错误
pylint backtest/ --errors-only
```

### 代码组织规范

#### 1. 文件命名

- Python 模块：使用小写字母和下划线 `my_module.py`
- Python 包：使用小写字母和下划线 `my_package/`
- 测试文件：`test_*.py` 或 `*_test.py`
- 类：使用大驼峰命名 `MyClass`
- 函数和变量：使用小写字母和下划线 `my_function`
- 常量：使用大写字母和下划线 `MY_CONSTANT`

#### 2. 导入顺序

按照以下顺序组织导入（由 isort 自动处理）：

```python
# 1. 标准库导入
import os
import sys
from datetime import datetime

# 2. 第三方库导入
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 3. 本地应用/库导入
from backtest.engine import BacktestEngine
from utils.logging import get_logger
```

#### 3. 文档字符串

使用 Google 风格的文档字符串：

```python
def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """计算简单移动平均线。

    Args:
        data: 价格数据序列
        window: 窗口大小

    Returns:
        SMA 值序列

    Raises:
        ValueError: 如果 window 小于 1

    Examples:
        >>> prices = pd.Series([1, 2, 3, 4, 5])
        >>> calculate_sma(prices, 3)
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        4    4.0
    """
    if window < 1:
        raise ValueError("窗口大小必须大于 0")
    return data.rolling(window=window).mean()
```

#### 4. 类型注解

所有函数都应该包含类型注解：

```python
from typing import List, Dict, Optional

def process_data(
    data: pd.DataFrame,
    columns: List[str],
    options: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """处理数据并返回结果。"""
    if options is None:
        options = {}
    # ... 实现代码 ...
    return result
```

#### 5. 异常处理

```python
# 推荐：捕获具体的异常
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"操作失败: {e}")
    handle_error(e)

# 不推荐：捕获所有异常
try:
    result = risky_operation()
except Exception:  # 太宽泛
    pass
```

#### 6. 日志记录

使用项目统一的日志工具：

```python
from utils.logging import get_logger

logger = get_logger(__name__)

def some_function():
    logger.info("开始处理数据")
    try:
        # ... 代码 ...
        logger.debug("处理完成，结果: %s", result)
    except Exception as e:
        logger.error("处理失败", exc_info=True)
```

#### 7. 配置管理

始终使用 `config/settings.py` 中的配置，不要硬编码值：

```python
# 推荐
from config.settings import get_settings
settings = get_settings()
initial_cash = settings.backtest.initial_cash

# 不推荐
initial_cash = 100000  # 硬编码
```

### Rust 代码规范

对于 Rust 引擎代码，我们遵循标准的 Rust 规范：

```bash
# 格式化 Rust 代码
cd rust_engine
cargo fmt

# 运行 Clippy（Rust linter）
cargo clippy -- -D warnings

# 运行测试
cargo test
```

**Rust 命名规范**：
- 模块：`snake_case`
- 类型：`UpperCamelCase`
- 函数和变量：`snake_case`
- 常量：`SCREAMING_SNAKE_CASE`

---

## 测试要求

### 测试原则

1. **测试覆盖率**: 新代码的测试覆盖率必须至少达到 70%
2. **独立测试**: 每个测试应该独立运行，不依赖其他测试
3. **快速执行**: 单元测试应该快速执行（每个测试 < 1 秒）
4. **可读性**: 测试代码应该像文档一样清晰易懂

### 测试结构

项目使用 pytest 作为测试框架，测试结构如下：

```
tests/
├── unit/              # 单元测试
│   ├── test_indicators.py
│   ├── test_strategy.py
│   └── ...
├── integration/       # 集成测试
│   ├── test_end_to_end.py
│   └── ...
├── backtest/          # 回测模块测试
├── agents/            # Agent 模块测试
├── rl/                # 强化学习模块测试
├── performance/       # 性能测试
└── conftest.py        # pytest 配置
```

### 编写单元测试

```python
import pytest
import pandas as pd
from backtest.engine.indicators import calculate_sma

class TestSMA:
    """SMA 指标测试类。"""

    def test_calculate_sma_basic(self):
        """测试基本的 SMA 计算。"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = calculate_sma(data, window=3)

        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_sma_invalid_window(self):
        """测试无效的窗口大小。"""
        data = pd.Series([1, 2, 3])

        with pytest.raises(ValueError, match="窗口大小必须大于 0"):
            calculate_sma(data, window=0)

    @pytest.mark.parametrize("window,expected", [
        (2, [np.nan, 1.5, 2.5, 3.5, 4.5]),
        (3, [np.nan, np.nan, 2.0, 3.0, 4.0]),
    ])
    def test_calculate_sma_different_windows(self, window, expected):
        """测试不同窗口大小的 SMA 计算。"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = calculate_sma(data, window=window)
        pd.testing.assert_series_equal(result, pd.Series(expected))
```

### 使用 Fixtures

```python
import pytest
from backtest.engine import BacktestEngine

@pytest.fixture
def sample_data():
    """提供示例数据。"""
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'close': np.random.randn(100).cumsum() + 100
    })

@pytest.fixture
def backtest_engine():
    """提供回测引擎实例。"""
    engine = BacktestEngine(
        initial_cash=100000,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )
    yield engine
    # 清理代码
    engine.reset()

def test_backtest_with_sample_data(backtest_engine, sample_data):
    """使用示例数据测试回测。"""
    result = backtest_engine.run(data=sample_data)
    assert result['total_return'] > 0
```

### 测试标记

使用 pytest 标记来分类测试：

```python
import pytest

@pytest.mark.unit
def test_indicator_calculation():
    """单元测试。"""
    pass

@pytest.mark.integration
def test_full_backtest_workflow():
    """集成测试。"""
    pass

@pytest.mark.slow
def test_long_running_operation():
    """慢速测试。"""
    pass

@pytest.mark.requires_data
def test_with_external_data():
    """需要外部数据的测试。"""
    pass

@pytest.mark.requires_internet
def test_api_call():
    """需要互联网连接的测试。"""
    pass
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定标记的测试
pytest -m "unit"                    # 只运行单元测试
pytest -m "integration and not slow"  # 集成测试，排除慢速测试
pytest -m "not slow"                # 排除慢速测试

# 运行特定文件
pytest tests/backtest/test_indicators.py

# 运行特定测试类或函数
pytest tests/backtest/test_indicators.py::TestSMA::test_calculate_sma_basic

# 显示详细输出
pytest -v

# 显示打印输出
pytest -s

# 在第一个失败时停止
pytest -x

# 运行上次失败的测试
pytest --lf

# 并行运行测试（需要 pytest-xdist）
pytest -n auto

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### Mock 和 Patch

使用 `unittest.mock` 进行模拟：

```python
from unittest.mock import Mock, patch, MagicMock
import pytest

def test_with_mock():
    """使用 Mock 对象测试。"""
    mock_api = Mock()
    mock_api.get_data.return_value = {'price': 100}

    result = process_data(mock_api)
    assert result == 100
    mock_api.get_data.assert_called_once()

@patch('backtest.engine.data_handler.fetch_data')
def test_with_patch(mock_fetch):
    """使用 patch 装饰器测试。"""
    mock_fetch.return_value = pd.DataFrame({'close': [1, 2, 3]})

    result = backtest.run()
    assert not result.empty
```

### 测试异常

```python
def test_exception_handling():
    """测试异常处理。"""
    with pytest.raises(ValueError, match="错误消息"):
        raise_function()

def test_warning():
    """测试警告。"""
    with pytest.warns(UserWarning):
        warning_function()
```

### 性能测试

使用 pytest-benchmark 进行性能测试：

```python
import pytest

@pytest.mark.benchmark
def test_indicator_performance(benchmark):
    """测试指标计算性能。"""
    data = pd.Series(np.random.randn(10000))

    result = benchmark(calculate_sma, data, window=20)
    assert len(result) == len(data)
```

### 测试覆盖率要求

- **总体覆盖率**: 最低 70%（目标：80%+）
- **新增代码覆盖率**: 必须达到 80%
- **关键模块覆盖率**:
  - `backtest/engine/`: 75%+
  - `agents/`: 70%+
  - `rl/`: 70%+
  - `data/`: 65%+

查看覆盖率报告：

```bash
# 生成 HTML 报告
pytest --cov=. --cov-report=html

# 在浏览器中查看
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 持续集成中的测试

CI 管道会自动运行：

1. **Lint 检查**: Black, isort, Flake8, mypy
2. **单元测试**: 所有标记为 `unit` 的测试
3. **集成测试**: 所有标记为 `integration` 的测试
4. **覆盖率检查**: 确保覆盖率不低于 70%
5. **性能测试**: 标记为 `benchmark` 的测试（仅 main 分支）

确保所有测试在本地通过后再提交 PR。

---

## 提交信息规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范来编写提交信息。

### 提交信息格式

```
<类型>(<范围>): <简短描述>

<详细描述>

<页脚>
```

### 类型（Type）

必须使用以下类型之一：

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行的变动，如空格、格式化等）
- `refactor`: 重构（既不是新增功能，也不是修复 Bug）
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动
- `ci`: CI 配置文件和脚本的变动
- `revert`: 回滚之前的提交

### 范围（Scope）

范围指明提交影响的模块，例如：

- `backtest`: 回测引擎
- `agents`: LLM Agent 系统
- `rl`: 强化学习模块
- `data`: 数据层
- `config`: 配置
- `docs`: 文档
- `tests`: 测试

### 简短描述

- 使用现在时态："add" 而不是 "added" 或 "adds"
- 不要大写首字母
- 不要以句号结尾
- 限制在 50 个字符以内

### 详细描述（可选）

- 说明"是什么"和"为什么"，而不是"怎么做"
- 每行限制在 72 个字符以内

### 页脚（可选）

- 关联 Issue: `Closes #123`, `Fixes #456`, `Refs #789`
- 破坏性更改: 以 `BREAKING CHANGE:` 开头

### 示例

#### 新功能

```bash
git commit -m "feat(backtest): 添加威廉指标计算

实现了威廉 %R 指标的计算函数，支持自定义参数。

Closes #123"
```

#### Bug 修复

```bash
git commit -m "fix(agents): 修复 Agent 协调器的死锁问题

修复了在多线程环境下 Agent 消息传递可能导致的死锁。
现在使用线程安全的队列进行消息传递。

Fixes #456"
```

#### 文档更新

```bash
git commit -m "docs: 更新 API 文档

添加了新的示例代码和详细的使用说明。"
```

#### 性能优化

```bash
git commit -m "perf(data): 优化数据查询性能

通过添加索引和使用批量查询，将数据加载时间
从 2 秒降低到 0.5 秒。"
```

#### 破坏性更改

```bash
git commit -m "feat(rl)!: 重构奖励函数接口

BREAKING CHANGE: 奖励函数接口已经从简单的函数调用
改为类继承方式。现有代码需要相应更新。

迁移指南：
1. 将奖励函数继承自 BaseReward
2. 实现 calculate() 方法
3. 更新配置文件"
```

#### 重构

```bash
git commit -m "refactor(backtest): 简化事件处理逻辑

将复杂的事件处理逻辑拆分为多个小函数，
提高代码可读性和可维护性。"
```

### 使用 Commitlint

项目可能配置了 commitlint 来强制执行提交信息规范：

```bash
# 安装 commitlint
npm install -g @commitlint/cli @commitlint/config-conventional

# 在提交前检查
git commit -m "feat: add new feature" | commitlint

# 或使用 git hooks 自动检查
```

---

## 代码审查标准

### 审查流程

1. **自动检查**：CI 管道会自动运行测试、lint 和覆盖率检查
2. **人工审查**：至少一位维护者会审查您的代码
3. **反馈**：维护者可能会提出修改建议
4. **修改**：您需要根据反馈进行修改
5. **批准**：所有审查通过后，PR 会被合并

### 审查标准

#### 1. 功能正确性

- [ ] 代码实现了 PR 描述中承诺的功能
- [ ] 代码没有引入新的 Bug
- [ ] 边界情况得到正确处理
- [ ] 错误处理得当

#### 2. 代码质量

- [ ] 代码遵循项目的代码规范
- [ ] 代码结构清晰，易于理解
- [ ] 没有重复代码（DRY 原则）
- [ ] 函数和类职责单一（SRP 原则）
- [ ] 适当的抽象和模块化

#### 3. 测试

- [ ] 新增代码有相应的测试
- [ ] 测试覆盖率达到 70% 以上
- [ ] 测试用例覆盖各种场景
- [ ] 测试代码清晰易懂

#### 4. 文档

- [ ] 代码有适当的注释
- [ ] 公共 API 有文档字符串
- [ ] 复杂逻辑有解释说明
- [ ] 如果是用户可见的功能，文档已更新

#### 5. 性能

- [ ] 代码性能合理
- [ ] 没有明显的性能问题
- [ ] 如果涉及性能敏感操作，已优化

#### 6. 安全性

- [ ] 没有安全漏洞
- [ ] 敏感信息（密钥、密码等）不在代码中
- [ ] 用户输入得到适当验证

#### 7. 兼容性

- [ ] 代码在支持的 Python 版本上运行
- [ ] 不破坏现有 API
- [ ] 如果是破坏性更改，已明确说明

### 审查者的责任

- **及时响应**：在 48 小时内审查 PR
- **建设性反馈**：提供具体、可操作的反馈
- **友好态度**：保持尊重和礼貌
- **解释原因**：如果拒绝更改，解释原因
- **认可贡献**：感谢贡献者的工作

### 贡献者的责任

- **响应反馈**：及时响应审查者的评论
- **保持开放**：接受建设性批评
- **持续改进**：根据反馈修改代码
- **测试更改**：确保修改后的代码仍然通过测试

### 常见审查反馈

#### "请添加测试"

```python
# 修改前
def calculate_rsi(data, period=14):
    # ... 计算逻辑 ...
    return rsi

# 修改后
def calculate_rsi(data, period=14):
    """计算相对强弱指标。

    Args:
        data: 价格数据
        period: 计算周期，默认 14

    Returns:
        RSI 值序列

    Raises:
        ValueError: 如果 period 小于 2 或数据长度不足
    """
    if period < 2:
        raise ValueError("周期必须大于 1")
    # ... 计算逻辑 ...
    return rsi

# 添加测试
def test_calculate_rsi_invalid_period():
    with pytest.raises(ValueError):
        calculate_rsi(data, period=1)
```

#### "请改进文档字符串"

```python
# 修改前
def process(data):
    # 处理数据
    return result

# 修改后
def process(data: pd.DataFrame) -> pd.DataFrame:
    """处理并清理输入数据。

    执行以下操作：
    1. 删除重复行
    2. 填充缺失值
    3. 标准化列名

    Args:
        data: 输入的数据框

    Returns:
        清理后的数据框

    Examples:
        >>> df = pd.DataFrame({'A': [1, None, 3]})
        >>> process(df)
           A
        0  1.0
        1  1.5  # 填充的缺失值
        2  3.0
    """
    # ... 实现代码 ...
```

#### "请遵循代码规范"

```python
# 修改前（不符合规范）
def  CalculateSMA(data,window):
    if len(data)==0:
        return None
    return data.rolling(window).mean()

# 修改后（符合规范）
def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """计算简单移动平均线。"""
    if len(data) == 0:
        return pd.Series(dtype=float)
    return data.rolling(window).mean()
```

---

## 项目结构说明

### 顶层目录结构

```
quanta/
├── agents/                 # LLM Agent 系统
│   ├── base/              # Agent 基类和协调器
│   ├── market_data_agent/ # 市场数据 Agent
│   ├── technical_agent/   # 技术分析 Agent
│   ├── sentiment_agent/   # 情绪分析 Agent
│   ├── strategy_agent/    # 策略生成 Agent
│   └── risk_agent/        # 风险管理 Agent
├── backtest/              # 回测引擎
│   ├── engine/            # 核心引擎
│   │   ├── backtest.py    # 主回测引擎
│   │   ├── indicators.py  # 技术指标
│   │   ├── a_share_rules.py  # A股交易规则
│   │   └── event_engine.py    # 事件引擎
│   ├── metrics/           # 性能指标和分析
│   ├── portfolio/         # 投资组合回测
│   ├── optimization/      # 策略优化
│   └── visualization/     # 可视化
├── config/                # 配置文件
│   ├── settings.py        # 主配置
│   ├── symbols.py         # 股票池配置
│   └── strategies.py      # 策略参数配置
├── data/                  # 数据层
│   ├── market/            # 市场数据
│   │   ├── sources/       # 数据源（Tushare, AKShare）
│   │   └── storage/       # 数据存储（DuckDB, ClickHouse）
│   ├── fundamental/       # 基本面数据
│   └── alternative/       # 另类数据
├── live/                  # 实盘交易
│   ├── brokers/           # 券商接口
│   ├── monitoring/        # 监控面板
│   └── compliance/        # 合规检查
├── rl/                    # 强化学习
│   ├── envs/              # 交易环境
│   ├── training/          # 模型训练
│   ├── rewards/           # 奖励函数
│   ├── optimization/      # 超参数优化
│   └── evaluation/        # 模型评估
├── rust_engine/           # Rust 执行引擎
│   ├── src/               # Rust 源代码
│   └── Cargo.toml         # Rust 项目配置
├── tests/                 # 测试
│   ├── unit/              # 单元测试
│   ├── integration/       # 集成测试
│   ├── backtest/          # 回测测试
│   ├── agents/            # Agent 测试
│   ├── rl/                # RL 测试
│   └── conftest.py        # pytest 配置
├── utils/                 # 工具函数
│   ├── logging.py         # 日志工具
│   ├── time_utils.py      # 时间工具
│   ├── helpers.py         # 通用助手函数
│   └── performance.py     # 性能工具
├── examples/              # 示例代码
├── docs/                  # 文档
├── .github/               # GitHub 配置
│   └── workflows/         # CI/CD 工作流
├── .env.example           # 环境变量示例
├── .gitignore             # Git 忽略文件
├── .isort.cfg             # isort 配置
├── pytest.ini             # pytest 配置
├── requirements.txt       # 依赖列表
├── Makefile               # Make 命令
├── CLAUDE.md              # Claude Code 指南
├── CONTRIBUTING.md        # 贡献指南（本文件）
├── README.md              # 项目说明
└── PROGRESS.md            # 项目进度
```

### 核心模块说明

#### 1. agents/ - LLM Agent 系统

**职责**: 使用 GLM-4 和 LangGraph 实现的多 Agent 决策系统

**关键文件**:
- `base/coordinator.py`: Agent 协调器，管理消息路由
- `base/agent_base.py`: Agent 基类
- `base/glm4_integration.py`: GLM-4 API 集成
- `base/langgraph_integration.py`: LangGraph 集成

**开发指南**:
- 每个 Agent 继承自 `AgentBase`
- 使用结构化的 JSON 消息进行通信
- Agent 响应应该是确定性的（相同的输入产生相同的输出）

#### 2. backtest/ - 回测引擎

**职责**: 事件驱动的回测引擎，支持 A 股交易规则

**关键文件**:
- `engine/backtest.py`: 主回测引擎
- `engine/indicators.py`: 技术指标库（20+ 指标）
- `engine/a_share_rules.py`: A 股规则实现（T+1, 涨跌停等）
- `engine/event_engine.py`: 事件处理引擎
- `metrics/performance.py`: 性能指标计算
- `metrics/report.py`: HTML 报告生成

**开发指南**:
- 新增技术指标在 `indicators.py` 中实现
- 确保所有交易逻辑遵循 A 股规则
- 使用事件驱动架构，不要直接调用下单函数
- 性能指标应该与主流回测框架一致

#### 3. rl/ - 强化学习框架

**职责**: 基于 Stable-Baselines3 的强化学习交易系统

**关键文件**:
- `envs/a_share_trading_env.py`: Gymnasium 交易环境
- `training/trainer.py`: 模型训练器（PPO, DQN, A2C, SAC, TD3）
- `rewards/reward_functions.py`: 8+ 奖励函数实现
- `optimization/hyperparameter_tuning.py`: 超参数优化
- `evaluation/model_evaluator.py`: 模型评估和比较

**开发指南**:
- 环境必须符合 Gymnasium 规范
- 观察空间：19-20 维
- 动作空间：离散（0=持有, 1=买入, 2=卖出）
- 新增奖励函数应该继承自 `BaseReward`

#### 4. data/ - 数据层

**职责**: 市场数据收集、存储和管理

**关键文件**:
- `market/sources/tushare_provider.py`: Tushare 数据源
- `market/sources/akshare_provider.py`: AKShare 数据源
- `market/storage/duckdb_storage.py`: DuckDB 存储
- `market/storage/clickhouse_storage.py`: ClickHouse 存储

**开发指南**:
- 新增数据源应该实现统一的数据接口
- 数据应该以标准化的格式存储
- 考虑数据缓存以提高性能
- 处理数据缺失和异常值

#### 5. config/ - 配置管理

**职责**: 集中式配置管理

**关键文件**:
- `settings.py`: 主配置（使用 dataclasses）

**开发指南**:
- 所有配置应该在 `settings.py` 中定义
- 使用环境变量存储敏感信息
- 不要硬编码配置值
- 配置应该有类型注解和默认值

#### 6. live/ - 实盘交易

**职责**: 实盘交易执行和监控

**关键文件**:
- `brokers/xtp_broker.py`: 华泰 XTP 接口
- `monitoring/dashboard.py`: Streamlit 监控面板

**开发指南**:
- 实盘交易代码必须有充分的测试
- 实现风险控制和熔断机制
- 监控和日志记录至关重要
- 考虑使用模拟模式进行测试

#### 7. rust_engine/ - Rust 执行引擎

**职责**: 高性能执行引擎（开发中）

**关键文件**:
- `src/order_manager.rs`: 订单管理
- `src/portfolio.rs`: 投资组合管理
- `src/execution.rs`: 订单执行

**开发指南**:
- 使用 PyO3 提供 Python 绑定
- Rust 代码应该有完整的测试
- 注意 Python 和 Rust 之间的数据转换开销
- 文档应该详细说明 API

### 添加新功能的指南

#### 添加新技术术指标

1. 在 `backtest/engine/indicators.py` 中实现函数
2. 添加完整的文档字符串和类型注解
3. 在 `tests/backtest/test_indicators.py` 中添加测试
4. 更新 `backtest/metrics/` 中的使用示例
5. 更新文档

```python
def calculate_new_indicator(
    data: pd.Series,
    param1: int = 14,
    param2: float = 0.5
) -> pd.Series:
    """计算新的技术指标。

    Args:
        data: 价格数据
        param1: 参数1说明
        param2: 参数2说明

    Returns:
        指标值序列

    Raises:
        ValueError: 如果参数无效
    """
    # 实现代码
    pass
```

#### 添加新 Agent

1. 在 `agents/` 下创建新目录 `new_agent/`
2. 实现 `agent.py`，继承 `AgentBase`
3. 在 `agents/base/coordinator.py` 中注册新 Agent
4. 添加测试到 `tests/agents/`
5. 更新文档

```python
from agents.base.agent_base import AgentBase

class NewAgent(AgentBase):
    """新 Agent 的描述。"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(
            name="new_agent",
            role="描述 Agent 的角色",
            config=config
        )

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析上下文并返回结果。"""
        # 实现代码
        pass
```

#### 添加新强化学习算法

1. 在 `rl/training/trainer.py` 中添加训练器
2. 确保环境兼容
3. 添加超参数配置
4. 添加测试和文档

```python
def train_new_algorithm(
    env: gym.Env,
    total_timesteps: int = 100000,
    **kwargs
) -> Any:
    """训练新算法模型。

    Args:
        env: 交易环境
        total_timesteps: 总训练步数
        **kwargs: 其他超参数

    Returns:
        训练好的模型
    """
    # 实现代码
    pass
```

---

## 常见问题

### Q: 我的 PR 没有得到响应怎么办？

A: 请等待至少 7 天，然后在 PR 中友好地提醒维护者。您也可以在 Discord/Slack 社区中询问。

### Q: CI 测试失败但我本地测试通过，怎么办？

A: 这通常是由于环境差异：
1. 检查 Python 版本是否一致
2. 确保所有依赖都已更新
3. 检查是否有环境特定的代码
4. 查看具体的 CI 错误日志

### Q: 我可以同时提交多个相关的 PR 吗？

A: 可以，但建议：
1. 第一个 PR 应该是基础性的，其他 PR 依赖它
2. 在 PR 描述中说明依赖关系
3. 或者考虑合并为一个大的 PR

### Q: 如何处理合并冲突？

A: 解决合并冲突的步骤：
```bash
# 1. 获取最新的上游代码
git fetch upstream

# 2. 变基到最新的 master
git rebase upstream/master

# 3. 解决冲突
# 编辑冲突文件，解决冲突标记

# 4. 标记冲突已解决
git add <conflicted-files>
git rebase --continue

# 5. 强制推送（谨慎使用）
git push origin <branch-name> --force-with-lease
```

### Q: 我不知道该从哪里开始贡献？

A: 这里有一些好的起点：
- 标记为 `good first issue` 的 Issues
- 改进文档
- 添加测试以提高覆盖率
- 修复小的 Bug
- 回答社区中的问题

### Q: 我可以贡献不在我专业领域的代码吗？

A: 当然可以！但请：
1. 先阅读相关模块的文档和现有代码
2. 从小的改动开始
3. 在 Issue 中讨论您的计划
4. 欢迎提问和学习

### Q: 如何设置开发环境以避免权限问题？

A:
```bash
# 使用虚拟环境
python3 -m venv venv
source venv/bin/activate

# 或使用 pip 用户安装
pip install --user -r requirements.txt

# 确保 PYTHONPATH 正确设置
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Q: 测试太慢，有什么建议？

A:
1. 使用 pytest 标记排除慢速测试：`pytest -m "not slow"`
2. 使用并行测试：`pytest -n auto`（需要 pytest-xdist）
3. 只运行相关测试：`pytest tests/backtest/test_indicators.py`
4. 使用 pytest 的缓存：`pytest --cache-clear`

### Q: 我发现了安全漏洞，应该怎么报告？

A: 请不要在公开的 Issue 中报告安全漏洞。发送邮件到项目维护者的安全邮箱，详情见 SECURITY.md。

### Q: 如何申请成为维护者？

A: 我们欢迎活跃的贡献者成为维护者：
1. 持续贡献高质量代码
2. 积极审查其他人的 PR
3. 帮助回答社区问题
4. 表现出对项目的深入理解
5. 联系现任维护者表达兴趣

---

## 获取帮助

如果您有任何问题或需要帮助：

- **GitHub Issues**: 报告 Bug 或功能请求
- **GitHub Discussions**: 询问问题、分享想法
- **Discord/Slack**: 实时讨论（链接在 README 中）
- **邮件**: 联系维护者（见 README）

---

## 许可证

通过贡献代码，您同意您的贡献将使用项目的 [MIT 许可证](LICENSE) 进行许可。

---

## 致谢

感谢所有为 quantA 项目做出贡献的人！您的贡献使这个项目变得更好。

---

## 更新日志

本指南可能会随着项目的发展而更新。请定期查看最新版本。

最后更新：2025-01-30
