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
