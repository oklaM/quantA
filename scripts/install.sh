#!/bin/bash
# quantA 一键安装脚本
# 自动配置开发环境和安装所有依赖

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_step "quantA 一键安装程序"
print_info "项目目录: $PROJECT_ROOT"
print_info "开始安装..."

# Step 1: 检查系统要求
print_step "步骤 1/7: 检查系统要求"

# 检查Python版本
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    print_info "检测到 Python 版本: $PYTHON_VERSION"

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
        print_error "Python 版本过低！需要 Python 3.9 或更高版本"
        print_info "请升级 Python 后重试"
        exit 1
    fi

    print_success "Python 版本检查通过"
else
    print_error "未找到 Python 3！"
    print_info "请安装 Python 3.9 或更高版本"
    exit 1
fi

# 检查pip
if command_exists python3 -m pip; then
    print_success "pip 可用"
else
    print_error "pip 不可用"
    print_info "请安装 pip"
    exit 1
fi

# 检查git
if command_exists git; then
    print_success "Git 可用"
else
    print_warning "未找到 Git，建议安装以获得完整功能"
fi

# Step 2: 创建虚拟环境
print_step "步骤 2/7: 创建虚拟环境"

VENV_DIR="$PROJECT_ROOT/venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除旧的虚拟环境..."
        rm -rf "$VENV_DIR"
    else
        print_info "使用现有虚拟环境"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_info "创建虚拟环境..."
    python3 -m venv "$VENV_DIR"
    print_success "虚拟环境创建成功"
fi

# 激活虚拟环境
print_info "激活虚拟环境..."
source "$VENV_DIR/bin/activate"

# 升级pip
print_info "升级 pip 到最新版本..."
pip install --upgrade pip setuptools wheel

# Step 3: 安装基础依赖
print_step "步骤 3/7: 安装基础依赖"

# 询问用户安装模式
echo ""
read -p "选择安装模式: (1) 完整安装 (2) 核心安装 (推荐) [1/2]: " -n 1 -r
echo ""
INSTALL_MODE=$REPLY

if [ "$INSTALL_MODE" = "2" ]; then
    print_info "使用核心依赖安装..."

    if [ -f "$PROJECT_ROOT/requirements-core.txt" ]; then
        print_info "安装 requirements-core.txt 中的依赖..."
        pip install -r "$PROJECT_ROOT/requirements-core.txt" || {
            print_warning "部分依赖安装失败，继续安装..."
        }
        print_success "核心依赖安装完成"
    else
        print_error "未找到 requirements-core.txt"
        exit 1
    fi
else
    print_info "使用完整依赖安装..."

    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_info "安装 requirements.txt 中的依赖..."
        print_info "注意: 某些包可能需要系统依赖或编译"

        # 分批安装，遇到错误继续
        print_info "安装基础数据处理包..."
        pip install pandas numpy scipy || true

        print_info "安装数据源..."
        pip install akshare || true
        pip install efinance>=0.5.0 || true
        # tushare需要token，跳过
        print_warning "tushare需要手动配置token，已跳过"

        print_info "安装技术指标..."
        pip install ta || true

        print_info "安装可视化..."
        pip install matplotlib seaborn || true

        print_info "安装工具库..."
        pip install requests loguru tqdm python-dateutil pytz || true

        print_info "安装配置管理..."
        pip install pydantic python-dotenv || true

        print_info "安装测试工具..."
        pip install pytest pytest-cov || true

        print_info "安装代码质量工具..."
        pip install black isort flake8 || true

        # 可选：安装较大的包
        read -p "是否安装机器学习相关包 (torch, scikit-learn)? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "安装机器学习包..."
            pip install scikit-learn || true
        fi

        # 可选：安装LLM相关包
        read -p "是否安装LLM相关包 (langchain等)? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "安装LLM包..."
            pip install langchain || true
        fi

        print_success "基础依赖安装完成"
        print_warning "某些可选包可能未安装，您可以稍后手动安装"
    else
        print_warning "未找到 requirements.txt，跳过基础依赖安装"
    fi
fi

# Step 4: 安装开发依赖
print_step "步骤 4/7: 安装开发依赖"

DEV_DEPENDENCIES=(
    "pytest>=7.0.0"
    "pytest-cov>=4.0.0"
    "pytest-asyncio>=0.21.0"
    "pytest-mock>=3.10.0"
    "black>=23.0.0"
    "flake8>=6.0.0"
    "mypy>=1.0.0"
    "isort>=5.12.0"
)

print_info "安装开发工具..."

for dep in "${DEV_DEPENDENCIES[@]}"; do
    print_info "安装 $dep ..."
    pip install "$dep"
done

print_success "开发依赖安装完成"

# Step 5: 安装可选依赖
print_step "步骤 5/7: 安装可选依赖"

read -p "是否安装强化学习相关依赖? (stable-baselines3, gym等) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "安装强化学习依赖..."
    pip install stable-baselines3>=2.0.0 gymnasium>=0.29.0 shimmy>=0.2.0
    print_success "强化学习依赖安装完成"
fi

read -p "是否安装数据可视化相关依赖? (plotly, bokeh等) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "安装可视化依赖..."
    pip install plotly>=5.14.0 bokeh>=3.0.0
    print_success "可视化依赖安装完成"
fi

read -p "是否安装性能优化依赖? (numba, cython等) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "安装性能优化依赖..."
    pip install numba>=0.57.0 cython>=3.0.0
    print_success "性能优化依赖安装完成"
fi

# Step 6: 验证安装
print_step "步骤 6/7: 验证安装"

print_info "运行验证测试..."

# 检查关键包
CRITICAL_PACKAGES=(
    "numpy"
    "pandas"
    "matplotlib"
    "pytest"
)

ALL_INSTALLED=true

for package in "${CRITICAL_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        VERSION=$(python3 -c "import $package; print($package.__version__)")
        print_success "$package ($VERSION) 已安装"
    else
        print_error "$package 未安装"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = false ]; then
    print_error "关键包安装失败！"
    exit 1
fi

# 运行快速测试
print_info "运行快速测试..."

cd "$PROJECT_ROOT"

# 测试导入
python3 -c "
import sys
sys.path.insert(0, '.')

# 测试核心模块导入
try:
    from backtest.engine.engine import BacktestEngine
    print('✓ 回测引擎导入成功')
except Exception as e:
    print(f'✗ 回测引擎导入失败: {e}')

try:
    from backtest.indicators import *
    print('✓ 技术指标导入成功')
except Exception as e:
    print(f'✗ 技术指标导入失败: {e}')

try:
    from trading.risk import RiskController
    print('✓ 风控系统导入成功')
except Exception as e:
    print(f'✗ 风控系统导入失败: {e}')

try:
    from agents.base.agent_base import AgentBase
    print('✓ Agent基类导入成功')
except Exception as e:
    print(f'✗ Agent基类导入失败: {e}')

print('\\n快速导入测试完成！')
"

print_success "安装验证完成"

# Step 7: 创建配置文件
print_step "步骤 7/7: 创建配置文件"

# 创建 .env 文件示例
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    print_info "创建 .env 配置文件..."

    cat > "$PROJECT_ROOT/.env" << EOF
# quantA 环境配置文件
# 复制此文件并根据实际情况修改

# 数据源配置
AKSHARE_ENABLED=true
TUSHARE_TOKEN=your_tushare_token_here
TUSHARE_ENABLED=false

# LLM 配置
GLM_API_KEY=your_glm_api_key_here
GLM_MODEL=glm-4

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/quanta.log

# 测试配置
TEST_DATA_DIR=tests/data
TEST_PARALLEL=true

# 性能配置
NUMBA_ENABLED=true
MULTIPROCESSING=true
EOF

    print_success ".env 配置文件已创建"
fi

# 创建启动脚本
print_info "创建便捷启动脚本..."

cat > "$PROJECT_ROOT/quanta.sh" << 'EOF'
#!/bin/bash
# quantA 快速启动脚本

# 激活虚拟环境
source venv/bin/activate

# 根据参数执行不同操作
case "$1" in
    test)
        echo "运行测试..."
        pytest tests/ -v
        ;;
    lint)
        echo "代码风格检查..."
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        ;;
    format)
        echo "格式化代码..."
        black . --line-length=100
        isort . --profile=black
        ;;
    example)
        echo "运行示例..."
        python examples/strategies_example.py
        ;;
    verify)
        echo "验证系统..."
        bash scripts/verify_system.sh
        ;;
    *)
        echo "quantA - A股量化AI交易系统"
        echo ""
        echo "用法: ./quanta.sh [命令]"
        echo ""
        echo "命令:"
        echo "  test      - 运行测试"
        echo "  lint      - 代码风格检查"
        echo "  format    - 格式化代码"
        echo "  example   - 运行示例"
        echo "  verify    - 验证系统"
        echo ""
        echo "未指定命令时，启动 Python REPL"
        python3
        ;;
esac
EOF

chmod +x "$PROJECT_ROOT/quanta.sh"
print_success "启动脚本已创建: ./quanta.sh"

# 安装完成
print_step "安装完成！"

echo ""
print_success "quantA 安装成功！"
echo ""
echo -e "${GREEN}接下来的步骤:${NC}"
echo ""
echo "1. 激活虚拟环境:"
echo "   source venv/bin/activate"
echo ""
echo "2. 运行测试:"
echo "   pytest tests/ -v"
echo ""
echo "3. 查看示例:"
echo "   python examples/strategies_example.py"
echo ""
echo "4. 验证系统:"
echo "   bash scripts/verify_system.sh"
echo ""
echo "5. 使用便捷脚本:"
echo "   ./quanta.sh [test|lint|format|example|verify]"
echo ""
echo -e "${BLUE}提示:${NC} 请根据需要编辑 .env 文件配置数据源和API密钥"
echo ""

# 询问是否立即运行测试
read -p "是否立即运行测试验证安装? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "运行测试..."
    pytest tests/ -v --tb=short
fi

print_success "安装程序执行完毕！"
