#!/bin/bash
# quantA 文档自动生成脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
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
DOCS_DIR="$PROJECT_ROOT/docs"

cd "$PROJECT_ROOT"

print_step "quantA 文档自动生成"
print_info "项目目录: $PROJECT_ROOT"

# 检查Sphinx是否安装
if ! command -v sphinx-build &> /dev/null; then
    print_error "Sphinx 未安装！"
    print_info "安装命令: pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints"
    exit 1
fi

# 解析参数
BUILD_TYPE="html"
SERVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --html)
            BUILD_TYPE="html"
            shift
            ;;
        --pdf)
            BUILD_TYPE="pdf"
            shift
            ;;
        --all)
            BUILD_TYPE="all"
            shift
            ;;
        --serve)
            SERVE=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --html     构建HTML文档（默认）"
            echo "  --pdf      构建PDF文档"
            echo "  --all      构建所有格式"
            echo "  --serve    构建后启动本地服务器"
            echo ""
            echo "示例:"
            echo "  $0 --html --serve"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# 步骤1：清理旧文档
print_step "步骤 1/5: 清理旧文档"
print_info "删除 _build 目录..."
rm -rf "$DOCS_DIR/_build"
rm -rf "$DOCS_DIR/api"

print_success "清理完成"

# 步骤2：生成API文档
print_step "步骤 2/5: 生成API文档"

# 创建API文档目录
mkdir -p "$DOCS_DIR/api"

# 生成各个模块的API文档
modules=(
    "backtest:回测引擎"
    "agents:智能体模块"
    "data:数据模块"
    "live:实盘交易"
    "monitoring:监控告警"
    "rl:强化学习"
    "trading:交易执行"
    "utils:工具函数"
)

for module_info in "${modules[@]}"; do
    IFS=':' read -r module_name module_desc <<< "$module_info"
    api_file="$DOCS_DIR/api/$module_name.rst"

    print_info "生成 $module_desc API文档..."

    cat > "$api_file" << EOF
$module_name 模块
${"="====="}

.. automodule:: $module_name
   :members:
   :undoc-members:
   :show-inheritance:

Submodules
----------

.. toctree::
   :maxdepth: 4

EOF

    # 为每个子模块创建文档
    if [ -d "$PROJECT_ROOT/$module_name" ]; then
        find "$PROJECT_ROOT/$module_name" -name "*.py" -type f | while read file; do
            # 跳过__init__.py和测试文件
            if [[ "$file" == *"__init__"* ]] || [[ "$file" == *"test"* ]]; then
                continue
            fi

            # 计算相对路径和模块名
            rel_path="${file#$PROJECT_ROOT/}"
            module_path="${rel_path%.py}"
            module_path="${module_path//\//.}"

            # 添加到toctree
            echo "   $module_path" >> "$api_file"

            # 创建子模块文件
            submodule_file="$DOCS_DIR/api/$module_path.rst"
            mkdir -p "$(dirname "$submodule_file")"

            cat > "$submodule_file" << EOF
$(basename "$module_path")
${"="====="}

.. automodule:: $module_path
   :members:
   :undoc-members:
   :show-inheritance:
EOF
        done
    fi

    print_success "$module_desc API文档生成完成"
done

# 步骤3：生成索引
print_step "步骤 3/5: 生成文档索引"

cat > "$DOCS_DIR/api.rst" << EOF
API 参考文档
============

本节包含quantA所有模块的API文档。

.. toctree::
   :maxdepth: 2
   :caption: 模块列表:

EOF

for module_info in "${modules[@]}"; do
    IFS=':' read -r module_name module_desc <<< "$module_info"
    echo "   api/$module_name" >> "$DOCS_DIR/api.rst"
done

print_success "索引生成完成"

# 步骤4：构建文档
print_step "步骤 4/5: 构建文档"

case $BUILD_TYPE in
    html)
        print_info "构建HTML文档..."
        sphinx-build -b html "$DOCS_DIR" "$DOCS_DIR/_build/html"
        print_success "HTML文档构建完成"
        print_info "文档位置: $DOCS_DIR/_build/html/index.html"
        ;;
    pdf)
        print_info "构建PDF文档..."
        sphinx-build -b latex "$DOCS_DIR" "$DOCS_DIR/_build/latex"
        cd "$DOCS_DIR/_build/latex"
        make
        cd "$PROJECT_ROOT"
        print_success "PDF文档构建完成"
        print_info "文档位置: $DOCS_DIR/_build/latex/quanta.pdf"
        ;;
    all)
        print_info "构建所有格式..."
        sphinx-build -b html "$DOCS_DIR" "$DOCS_DIR/_build/html"
        sphinx-build -b latex "$DOCS_DIR" "$DOCS_DIR/_build/latex"
        cd "$DOCS_DIR/_build/latex"
        make
        cd "$PROJECT_ROOT"
        print_success "所有格式文档构建完成"
        ;;
esac

# 步骤5：启动本地服务器（可选）
if [ "$SERVE" = true ] && [ "$BUILD_TYPE" = "html" ]; then
    print_step "步骤 5/5: 启动本地文档服务器"
    print_info "服务器地址: http://localhost:8000"
    print_info "按 Ctrl+C 停止服务器"
    echo ""

    cd "$DOCS_DIR/_build/html"
    python3 -m http.server 8000
fi

# 步骤5：完成
print_step "文档生成完成"
print_success "所有文档生成完毕！"
print_info "提示: 使用 '$0 --serve' 在浏览器中查看文档"

exit 0
