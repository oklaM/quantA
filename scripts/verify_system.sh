#!/bin/bash
# quantA 系统快速验证脚本

echo "========================================"
echo "  quantA 系统快速验证"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 统计函数
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# 验证计数
total=0
passed=0
failed=0

echo "=== 1. 项目结构检查 ==="
echo ""

# 检查核心模块
modules=("agents" "backtest" "data" "live" "monitoring" "rl" "trading" "utils" "examples")
for mod in "${modules[@]}"; do
    if [ -d "$mod" ]; then
        check_pass "模块目录存在: $mod"
        ((passed++))
    else
        check_fail "模块目录缺失: $mod"
        ((failed++))
    fi
    ((total++))
done

echo ""
echo "=== 2. 测试文件检查 ==="
echo ""

test_count=$(find tests -name "*.py" 2>/dev/null | wc -l)
echo "测试文件数量: $test_count"

if [ $test_count -ge 20 ]; then
    check_pass "测试文件充足 (>=20)"
    ((passed++))
else
    check_warn "测试文件较少 (<20)"
fi
((total++))

# 统计测试用例
if command -v grep &> /dev/null; then
    test_cases=$(grep -r "def test_" tests/ --include="*.py" 2>/dev/null | wc -l)
    echo "测试用例数量: $test_cases"

    if [ $test_cases -ge 200 ]; then
        check_pass "测试用例充足 (>=200)"
        ((passed++))
    else
        check_warn "测试用例较少 (<200)"
    fi
    ((total++))
fi

echo ""
echo "=== 3. 代码质量检查 ==="
echo ""

# 统计代码行数
if command -v find &> /dev/null && command -v wc &> /dev/null; then
    code_lines=$(find . -name "*.py" ! -path "./tests/*" ! -path "./.claude/*" ! -path "./__pycache__/*" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    echo "生产代码行数: $code_lines"

    if [ $code_lines -gt 15000 ]; then
        check_pass "代码规模适中 (>15k行)"
        ((passed++))
    fi
    ((total++))
fi

echo ""
echo "=== 4. 配置文件检查 ==="
echo ""

config_files=("pytest.ini" "requirements.txt" "README.md" "Makefile")
for file in "${config_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "配置文件存在: $file"
        ((passed++))
    else
        check_warn "配置文件缺失: $file"
    fi
    ((total++))
done

echo ""
echo "=== 5. 核心模块完整性 ==="
echo ""

# 检查关键文件
key_files=(
    "agents/base/agent_base.py"
    "backtest/engine/engine.py"
    "backtest/engine/strategy.py"
    "data/market/data_manager.py"
    "rl/envs/a_share_trading_env.py"
    "rl/training/trainer.py"
    "backtest/optimization/optimizer.py"
    "backtest/visualization/performance_viz.py"
    "monitoring/alerts.py"
    "trading/risk/controls.py"
)

for file in "${key_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "关键文件存在: $file"
        ((passed++))
    else
        check_fail "关键文件缺失: $file"
        ((failed++))
    fi
    ((total++))
done

echo ""
echo "=== 6. 文档完整性 ==="
echo ""

doc_files=("README.md" "docs/TEST_PLAN.md" "docs/TEST_REPORT.md" "docs/rust_engine_architecture.md" "docs/xtp_integration_guide.md")
for file in "${doc_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "文档文件存在: $file"
        ((passed++))
    else
        check_warn "文档文件缺失: $file"
    fi
    ((total++))
done

echo ""
echo "=== 7. 示例文件检查 ==="
echo ""

example_count=$(find examples -name "*.py" 2>/dev/null | wc -l)
echo "示例文件数量: $example_count"

if [ $example_count -ge 10 ]; then
    check_pass "示例文件充足 (>=10)"
    ((passed++))
else
    check_warn "示例文件较少 (<10)"
fi
((total++))

echo ""
echo "========================================"
echo "  验证结果汇总"
echo "========================================"
echo ""
echo -e "${GREEN}通过:${NC} $passed"
echo -e "${RED}失败:${NC} $failed"
echo -e "${YELLOW}总计:${NC} $total"
echo ""

pass_rate=$((passed * 100 / total))
echo "通过率: $pass_rate%"

echo ""
if [ $pass_rate -ge 80 ]; then
    echo -e "${GREEN}系统状态: 优秀 ✅${NC}"
    echo "系统已准备好用于生产环境！"
elif [ $pass_rate -ge 60 ]; then
    echo -e "${YELLOW}系统状态: 良好 ⚠️${NC}"
    echo "系统基本可用，建议完善部分功能。"
else
    echo -e "${RED}系统状态: 需要改进 ❌${NC}"
    echo "系统存在较多问题，需要进一步完善。"
fi

echo ""
echo "========================================"
echo "  建议的后续步骤"
echo "========================================"
echo ""
echo "1. 安装依赖:"
echo "   pip install -r requirements.txt"
echo ""
echo "2. 运行测试:"
echo "   pytest tests/ -v"
echo ""
echo "3. 查看示例:"
echo "   python examples/strategies_example.py"
echo ""
echo "4. 阅读文档:"
echo "   cat README.md"
echo "   cat docs/TEST_REPORT.md"
echo ""

exit 0
