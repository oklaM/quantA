.PHONY: help install test test-cov test-unit test-integration lint format clean run

# 默认目标
help:
	@echo "quantA - A股量化AI交易系统"
	@echo ""
	@echo "可用命令:"
	@echo "  make install     - 安装依赖"
	@echo "  make test        - 运行所有测试"
	@echo "  make test-cov    - 运行测试并生成覆盖率报告"
	@echo "  make test-unit   - 只运行单元测试"
	@echo "  make test-integration - 只运行集成测试"
	@echo "  make lint        - 运行代码检查"
	@echo "  make format      - 格式化代码"
	@echo "  make clean       - 清理临时文件"
	@echo "  make run         - 运行示例程序"

# 安装依赖
install:
	@echo "安装依赖..."
	pip install -r requirements.txt
	@echo "依赖安装完成！"

# 创建虚拟环境
venv:
	@echo "创建虚拟环境..."
	python3 -m venv venv
	@echo "虚拟环境创建完成！"
	@echo "请运行 'source venv/bin/activate' 激活虚拟环境"

# 运行所有测试
test:
	@echo "运行所有测试..."
	pytest -v

# 运行测试并生成覆盖率报告
test-cov:
	@echo "运行测试并生成覆盖率报告..."
	pytest --cov=. --cov-report=html --cov-report=term
	@echo "覆盖率报告已生成: htmlcov/index.html"

# 只运行单元测试
test-unit:
	@echo "运行单元测试..."
	pytest -v -m "unit"

# 只运行集成测试
test-integration:
	@echo "运行集成测试..."
	pytest -v -m "integration"

# 只运行回测测试
test-backtest:
	@echo "运行回测测试..."
	pytest -v -m "backtest"

# 只运行Agent测试
test-agents:
	@echo "运行Agent测试..."
	pytest -v -m "agents"

# 只运行强化学习测试
test-rl:
	@echo "运行强化学习测试..."
	pytest -v -m "rl"

# 运行代码检查
lint:
	@echo "运行代码检查..."
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	mypy .

# 格式化代码
format:
	@echo "格式化代码..."
	black .
	isort .

# 检查代码格式
format-check:
	@echo "检查代码格式..."
	black --check .
	isort --check-only .

# 清理临时文件
clean:
	@echo "清理临时文件..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '.pytest_cache' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	find . -type d -name 'htmlcov' -exec rm -rf {} +
	find . -type f -name '.coverage' -delete
	find . -type f -name '*.log' -delete
	@echo "清理完成！"

# 运行回测示例
run:
	@echo "运行回测示例..."
	python3 examples/backtest_example.py

# 运行Agent示例
run-agent:
	@echo "运行Agent示例..."
	python3 examples/agent_example.py

# 安装开发依赖
install-dev:
	@echo "安装开发依赖..."
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy
	@echo "开发依赖安装完成！"

# 更新依赖
update-deps:
	@echo "更新依赖..."
	pip install --upgrade -r requirements.txt
	@echo "依赖更新完成！"

# 显示依赖树
deps-tree:
	@echo "显示依赖树..."
	pip install pipdeptree
	pipdeptree

# 数据库相关
db-init:
	@echo "初始化数据库..."
	python3 -m data.init_db

db-backup:
	@echo "备份数据库..."
	python3 -m data.backup_db

# 监控面板
monitor:
	@echo "启动监控面板..."
	streamlit run monitoring/dashboard.py

# API服务
api:
	@echo "启动API服务..."
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Docker相关
docker-build:
	@echo "构建Docker镜像..."
	docker build -t quanta:latest .

docker-run:
	@echo "运行Docker容器..."
	docker run -d --name quanta -p 8000:8000 quanta:latest

docker-stop:
	@echo "停止Docker容器..."
	docker stop quanta
	docker rm quanta
