#!/bin/bash
# 快速修复安装 - 只安装核心必需的包

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}quantA 快速安装（核心依赖）${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 激活虚拟环境
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} 激活虚拟环境..."
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠${NC} 虚拟环境不存在，请先运行: bash scripts/install.sh"
    exit 1
fi

echo ""
echo -e "${BLUE}安装核心依赖...${NC}"
echo ""

# 分批安装，失败不中断
echo -e "[1/8] 安装数据处理核心..."
pip install pandas numpy scipy --quiet || echo "  ⚠ 部分包安装失败"

echo -e "[2/8] 安装数据源..."
pip install akshare --quiet || echo "  ⚠ akshare安装失败"

echo -e "[3/8] 安装技术指标..."
pip install ta --quiet || echo "  ⚠ ta安装失败"

echo -e "[4/8] 安装可视化..."
pip install matplotlib seaborn --quiet || echo "  ⚠ 可视化包安装失败"

echo -e "[5/8] 安装工具库..."
pip install requests loguru tqdm --quiet || echo "  ⚠ 工具库安装失败"

echo -e "[6/8] 安装配置管理..."
pip install pydantic python-dotenv --quiet || echo "  ⚠ 配置管理包安装失败"

echo -e "[7/8] 安装测试工具..."
pip install pytest pytest-cov --quiet || echo "  ⚠ 测试工具安装失败"

echo -e "[8/8] 安装代码质量工具..."
pip install black isort flake8 --quiet || echo "  ⚠ 代码质量工具安装失败"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}核心依赖安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "下一步:"
echo -e "  1. 激活虚拟环境: source venv/bin/activate"
echo -e "  2. 运行测试: pytest tests/ -v"
echo -e "  3. 运行示例: python examples/strategies_example.py"
echo ""
