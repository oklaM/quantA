#!/bin/bash
# quantA 自动化测试脚本
# 用于本地运行完整的测试套件

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

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 解析命令行参数
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=false
MARKERS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --performance)
            TEST_TYPE="performance"
            shift
            ;;
        --edge-cases)
            TEST_TYPE="edge-cases"
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --marker)
            MARKERS="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --unit          运行单元测试"
            echo "  --integration   运行集成测试"
            echo "  --performance   运行性能测试"
            echo "  --edge-cases    运行异常场景测试"
            echo "  --coverage      生成覆盖率报告"
            echo "  --verbose       详细输出"
            echo "  --parallel      并行运行测试"
            echo "  --marker MARK   运行特定标记的测试"
            echo ""
            echo "示例:"
            echo "  $0 --unit --coverage"
            echo "  $0 --integration --verbose"
            echo "  $0 --marker 'not slow'"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

print_step "quantA 自动化测试"
print_info "项目目录: $PROJECT_ROOT"
print_info "测试类型: $TEST_TYPE"

# 检查pytest是否安装
if ! command -v pytest &> /dev/null; then
    print_error "pytest 未安装！"
    print_info "请运行: pip install pytest"
    exit 1
fi

# 构建pytest命令
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=. --cov-report=html --cov-report=term --cov-report=xml"
fi

if [ "$PARALLEL" = true ]; then
    if command -v pytest-xdist &> /dev/null; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    else
        print_warning "pytest-xdist 未安装，无法并行运行"
        print_info "安装命令: pip install pytest-xdist"
    fi
fi

# 根据测试类型选择测试目录
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/ -m 'not integration and not performance and not edge_case'"
        TEST_DESC="单元测试"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/ -m 'integration'"
        TEST_DESC="集成测试"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD tests/performance/ -m 'performance or benchmark'"
        TEST_DESC="性能测试"
        ;;
    edge-cases)
        PYTEST_CMD="$PYTEST_CMD tests/edge_cases/ -m 'edge_case'"
        TEST_DESC="异常场景测试"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/ -m 'not slow'"
        TEST_DESC="全部测试"
        ;;
esac

# 添加自定义标记
if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD -m '$MARKERS'"
fi

print_step "运行 $TEST_DESC"
print_info "命令: $PYTEST_CMD"

# 记录开始时间
START_TIME=$(date +%s)

# 运行测试
eval $PYTEST_CMD
TEST_RESULT=$?

# 计算耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# 打印结果
echo ""
if [ $TEST_RESULT -eq 0 ]; then
    print_success "$TEST_DESC 通过！"
    print_info "耗时: ${DURATION}秒"
else
    print_error "$TEST_DESC 失败！"
    exit 1
fi

# 显示覆盖率报告
if [ "$COVERAGE" = true ]; then
    print_step "覆盖率报告"

    if [ -f "htmlcov/index.html" ]; then
        print_info "HTML报告: htmlcov/index.html"

        # 尝试在浏览器中打开
        if command -v xdg-open &> /dev/null; then
            xdg-open htmlcov/index.html 2>/dev/null || true
        elif command -v open &> /dev/null; then
            open htmlcov/index.html 2>/dev/null || true
        fi
    fi

    if [ -f "coverage.xml" ]; then
        print_info "XML报告: coverage.xml"
    fi
fi

print_step "测试完成"
print_success "所有测试执行完毕！"

exit 0
