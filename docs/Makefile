# quantA 文档生成 Makefile

# 默认目标
.PHONY: help all html pdf clean serve watch

help:
	@echo "quantA 文档生成命令:"
	@echo ""
	@echo "  make html       - 生成HTML文档"
	@echo "  make pdf        - 生成PDF文档"
	@echo "  make all        - 生成所有格式"
	@echo "  make clean      - 清理生成的文件"
	@echo "  make serve      - 生成HTML并启动本地服务器"
	@echo "  make watch      - 监控文件变化并自动重新构建"
	@echo ""
	@echo "示例:"
	@echo "  make html"
	@echo "  make serve"

all: html pdf

html:
	@echo "生成HTML文档..."
	sphinx-build -b html . _build/html
	@echo "✓ HTML文档生成完成: _build/html/index.html"

pdf:
	@echo "生成PDF文档..."
	sphinx-build -b latex . _build/latex
	@echo "编译LaTeX..."
	$(MAKE) -C _build/latex
	@echo "✓ PDF文档生成完成: _build/latex/quanta.pdf"

clean:
	@echo "清理生成的文件..."
	rm -rf _build
	rm -rf api
	@echo "✓ 清理完成"

serve: html
	@echo "启动本地文档服务器..."
	@echo "访问地址: http://localhost:8000"
	@echo "按 Ctrl+C 停止服务器"
	@echo ""
	cd _build/html && python3 -m http.server 8000

watch:
	@echo "监控文件变化..."
	@command -v sphinx-autobuild > /dev/null || (echo "错误: sphinx-autobuild 未安装" && exit 1)
	sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000

# API文档生成
api:
	@echo "生成API文档..."
	bash ../scripts/generate_docs.sh --html

# 检查文档链接
linkcheck:
	@echo "检查文档链接..."
	sphinx-build -b linkcheck . _build/linkcheck

# 生成开发文档（包含private成员）
dev:
	@echo "生成开发版文档（包含private成员）"
	sphinx-build -b html -D autodoc_private_members=True . _build/html
