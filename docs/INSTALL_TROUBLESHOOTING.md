# 安装故障排除指南

## 常见问题和解决方案

### 问题1: Python版本不兼容

**错误信息**: `ERROR: No matching distribution found for xxx`

**解决方案**:
quantA 需要 Python 3.9 或更高版本。推荐使用 Python 3.9-3.11。

```bash
# 检查Python版本
python --version

# 如果版本过低，安装新版本
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11 python3.11-venv

# macOS (使用Homebrew):
brew install python@3.11
```

### 问题2: efinance版本问题

**错误信息**: `ERROR: No matching distribution found for efinance>=0.5.6`

**原因**: PyPI上efinance的最新版本是 0.5.5.2，requirements.txt要求0.5.6不存在。

**解决方案**: 已修复，使用以下命令安装：

```bash
# 方案1: 使用快速安装脚本（推荐）
bash scripts/quick_install.sh

# 方案2: 手动安装正确版本
source venv/bin/activate
pip install efinance>=0.5.0
```

### 问题3: 某些包安装失败

**常见失败的包**:
- `backtrader` - Python 3.12支持有限
- `TA-Lib` - 需要系统依赖
- `stable-baselines3` - 依赖复杂
- `torch` - 包很大，下载慢

**解决方案**: 这些都是可选包，可以跳过：

```bash
# 只安装核心依赖
pip install -r requirements-core.txt

# 或者单独安装需要的包
pip install pandas numpy scipy matplotlib
pip install akshare ta
pip install pytest
```

### 问题4: 虚拟环境激活失败

**错误信息**: `source venv/bin/activate` 失败

**解决方案**:

```bash
# 删除旧虚拟环境
rm -rf venv

# 重新创建
python3 -m venv venv
source venv/bin/activate
```

### 问题5: 权限错误

**错误信息**: `Permission denied` 或 `[Errno 13]`

**解决方案**: 不要使用 sudo 安装包，始终在虚拟环境中安装：

```bash
# 正确的做法
source venv/bin/activate
pip install xxx

# 错误的做法（不要这样做）
sudo pip install xxx
```

### 问题6: 网络问题导致下载慢

**解决方案**: 使用国内镜像源：

```bash
# 临时使用
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

其他可用的镜像源：
- 清华: https://pypi.tuna.tsinghua.edu.cn/simple
- 阿里云: https://mirrors.aliyun.com/pypi/simple/
- 中科大: https://pypi.mirrors.ustc.edu.cn/simple/

### 问题7: 编译错误（某些包需要编译）

**错误信息**: `error: command 'gcc' failed`

**解决方案**: 安装编译依赖

```bash
# Ubuntu/Debian:
sudo apt install build-essential python3-dev

# macOS:
xcode-select --install

# 然后重试安装
pip install xxx
```

## 推荐的安装流程

### 最小化安装（推荐新手）

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 升级pip
pip install --upgrade pip

# 3. 安装核心依赖
bash scripts/quick_install.sh

# 4. 验证安装
python -c "import pandas, numpy; print('✓ 核心依赖安装成功')"
```

### 完整安装（需要时间）

```bash
# 1. 运行安装脚本，选择"完整安装"
bash scripts/install.sh

# 2. 根据提示选择要安装的组件
# - 强化学习包
# - LLM包
# - 等等

# 3. 验证安装
pytest tests/ -v
```

## 手动安装特定包

```bash
# 激活虚拟环境
source venv/bin/activate

# 数据处理
pip install pandas numpy scipy

# 数据源
pip install akshare

# 技术指标
pip install ta

# 可视化
pip install matplotlib seaborn plotly

# 测试
pip install pytest pytest-cov

# 机器学习（可选）
pip install scikit-learn

# 强化学习（可选）
pip install stable-baselines3 gymnasium

# LLM（可选）
pip install langchain zhipuai
```

## 验证安装

运行以下命令验证安装是否成功：

```bash
# 激活虚拟环境
source venv/bin/activate

# 检查Python版本
python --version

# 检查关键包
python -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta import add_all_ta_features
print('✓ 所有核心包导入成功')
"

# 运行环境检查
python scripts/check_env.py

# 运行系统验证
bash scripts/verify_system.sh

# 运行测试
pytest tests/backtest/test_indicators.py -v
```

## 获取帮助

如果遇到其他问题：

1. 查看日志文件: `logs/quanta.log`
2. 运行环境检查: `python scripts/check_env.py`
3. 查看 GitHub Issues: https://github.com/yourusername/quantA/issues
4. 查看文档: `docs/QUICKSTART.md`

## 包版本建议

基于测试，以下是推荐的最小版本：

```txt
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
akshare>=1.11.0
ta>=0.11.0
pytest>=7.4.0
pydantic>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
loguru>=0.7.0
```

## 卸载和重新安装

如果需要完全重新开始：

```bash
# 1. 停用虚拟环境
deactivate

# 2. 删除虚拟环境
rm -rf venv

# 3. 重新安装
bash scripts/install.sh
```
