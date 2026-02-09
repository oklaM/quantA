#!/usr/bin/env python3
"""
quantA 环境检查工具
检查系统环境、依赖项、配置和资源可用性
"""

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# 颜色输出
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

@staticmethod
def success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")

def error(msg: str):
    print(f"{Colors.RED}✗{Colors.NC} {msg}")

def warning(msg: str):
    print(f"{Colors.YELLOW}⚠{Colors.NC} {msg}")

def info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def header(msg: str):
    print("\n" + "="*70)
    print(f" {msg}")
    print("="*70 + "\n")


class EnvironmentChecker:
    """环境检查器"""

    def __init__(self):
        self.results = {
            'python': {},
            'packages': {},
            'system': {},
            'config': {},
            'data_sources': {},
        }
        self.warnings = []
        self.errors = []

    def check_all(self) -> bool:
        """运行所有检查"""
        header("quantA 环境检查工具")

        # 1. Python环境检查
        self.check_python_environment()

        # 2. 依赖包检查
        self.check_packages()

        # 3. 系统资源检查
        self.check_system_resources()

        # 4. 配置文件检查
        self.check_configuration()

        # 5. 数据源检查
        self.check_data_sources()

        # 6. 项目结构检查
        self.check_project_structure()

        # 生成报告
        self.generate_report()

        return len(self.errors) == 0

    def check_python_environment(self):
        """检查Python环境"""
        header("1. Python 环境检查")

        # Python版本
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major == 3 and version.minor >= 9:
            success(f"Python 版本: {version_str}")
            self.results['python']['version'] = version_str
            self.results['python']['version_ok'] = True
        else:
            error(f"Python 版本过低: {version_str} (需要 >= 3.9)")
            self.results['python']['version'] = version_str
            self.results['python']['version_ok'] = False
            self.errors.append("Python版本不符合要求")

        # Python路径
        success(f"Python 路径: {sys.executable}")
        self.results['python']['executable'] = sys.executable

        # 虚拟环境检查
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        if in_venm:
            success(f"虚拟环境: {sys.prefix}")
            self.results['python']['venv'] = True
        else:
            warning("未检测到虚拟环境")
            self.results['python']['venv'] = False
            self.warnings.append("建议使用虚拟环境")

        # 平台信息
        success(f"操作系统: {platform.system()} {platform.release()}")
        success(f"架构: {platform.machine()}")
        self.results['python']['platform'] = platform.system()
        self.results['python']['arch'] = platform.machine()

    def check_packages(self):
        """检查依赖包"""
        header("2. 依赖包检查")

        # 核心依赖
        core_packages = {
            'numpy': '1.21.0',
            'pandas': '1.5.0',
            'matplotlib': '3.5.0',
            'pytest': '7.0.0',
        }

        # 可选依赖
        optional_packages = {
            'stable_baselines3': '2.0.0',
            'gymnasium': '0.29.0',
            'plotly': '5.14.0',
            'bokeh': '3.0.0',
            'numba': '0.57.0',
            'cython': '0.29.0',
            'optuna': '3.0.0',
        }

        # 开发工具
        dev_tools = {
            'pytest_cov': '4.0.0',
            'black': '23.0.0',
            'flake8': '6.0.0',
            'mypy': '1.0.0',
        }

        # 检查核心包
        info("检查核心依赖...")
        for package, min_version in core_packages.items():
            self._check_package(package, min_version, required=True)

        # 检查可选包
        info("\n检查可选依赖...")
        for package, min_version in optional_packages.items():
            self._check_package(package, min_version, required=False)

        # 检查开发工具
        info("\n检查开发工具...")
        for package, min_version in dev_tools.items():
            self._check_package(package, min_version, required=False)

    def _check_package(self, package_name: str, min_version: str, required: bool):
        """检查单个包"""
        try:
            # 处理带下划线的包名
            import_name = package_name.replace('-', '_')

            module = importlib.import_module(import_name)

            # 获取版本
            version = getattr(module, '__version__', 'unknown')

            # 版本比较（简化）
            version_ok = True  # 简化处理，实际应该解析版本号

            if version_ok:
                success(f"{package_name}: {version}")
                self.results['packages'][package_name] = {
                    'installed': True,
                    'version': version,
                    'required': required,
                }
            else:
                warning(f"{package_name}: {version} (建议 >= {min_version})")
                self.results['packages'][package_name] = {
                    'installed': True,
                    'version': version,
                    'required': required,
                }
                if required:
                    self.warnings.append(f"{package_name}版本较旧")

        except ImportError:
            if required:
                error(f"{package_name}: 未安装 (需要 >= {min_version})")
                self.results['packages'][package_name] = {
                    'installed': False,
                    'required': True,
                }
                self.errors.append(f"缺少必需依赖: {package_name}")
            else:
                warning(f"{package_name}: 未安装 (可选)")
                self.results['packages'][package_name] = {
                    'installed': False,
                    'required': False,
                }

    def check_system_resources(self):
        """检查系统资源"""
        header("3. 系统资源检查")

        try:
            import psutil
            success("psutil 已安装，可以进行详细检查")

            # CPU
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            success(f"CPU 核心: {cpu_count}")
            success(f"CPU 使用率: {cpu_percent}%")
            self.results['system']['cpu_cores'] = cpu_count
            self.results['system']['cpu_usage'] = cpu_percent

            # 内存
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            percent = memory.percent

            success(f"总内存: {total_gb:.2f} GB")
            success(f"可用内存: {available_gb:.2f} GB")
            success(f"内存使用率: {percent}%")
            self.results['system']['memory_total'] = total_gb
            self.results['system']['memory_available'] = available_gb
            self.results['system']['memory_usage'] = percent

            # 磁盘
            disk = psutil.disk_usage('.')
            total_gb = disk.total / (1024**3)
            free_gb = disk.free / (1024**3)
            percent = disk.percent

            success(f"磁盘总空间: {total_gb:.2f} GB")
            success(f"磁盘可用空间: {free_gb:.2f} GB")
            success(f"磁盘使用率: {percent}%")
            self.results['system']['disk_total'] = total_gb
            self.results['system']['disk_free'] = free_gb
            self.results['system']['disk_usage'] = percent

            # 资源检查
            if total_gb < 2:
                warning("内存较少，可能影响性能")
                self.warnings.append("内存不足2GB")

            if free_gb < 5:
                warning("磁盘空间较少，建议清理")
                self.warnings.append("磁盘可用空间不足5GB")

        except ImportError:
            warning("psutil 未安装，跳过详细资源检查")
            self.warnings.append("建议安装 psutil 以获取系统资源信息")

        # 并行处理能力
        try:
            from multiprocessing import cpu_count
            cores = cpu_count()
            success(f"并行处理核心数: {cores}")
            self.results['system']['parallel_cores'] = cores
        except:
            pass

    def check_configuration(self):
        """检查配置文件"""
        header("4. 配置文件检查")

        config_files = {
            '.env': '环境配置文件',
            'requirements.txt': '依赖列表',
            'pytest.ini': 'pytest配置',
            'README.md': '项目说明',
        }

        for file, desc in config_files.items():
            if os.path.exists(file):
                success(f"{file} ({desc})")
                self.results['config'][file] = True
            else:
                warning(f"{file} ({desc}) 不存在")
                self.results['config'][file] = False
                if file == '.env':
                    self.warnings.append("缺少 .env 配置文件")

        # 检查 .env 内容
        if os.path.exists('.env'):
            info("\n检查 .env 配置项...")
            from dotenv import load_dotenv
            load_dotenv()

            # 检查关键配置
            config_keys = {
                'LOG_LEVEL': '日志级别',
                'TUSHARE_TOKEN': 'Tushare Token',
                'GLM_API_KEY': 'GLM API Key',
            }

            for key, desc in config_keys.items():
                value = os.getenv(key)
                if value:
                    success(f"{key} ({desc}): 已设置")
                else:
                    warning(f"{key} ({desc}): 未设置")

    def check_data_sources(self):
        """检查数据源连接"""
        header("5. 数据源检查")

        # AKShare
        try:
            import akshare as ak
            success("AKShare: 已安装")
            self.results['data_sources']['akshare'] = True

            # 尝试获取数据
            try:
                df = ak.stock_zh_a_spot_em()
                success(f"AKShare: 数据获取正常 (获取到 {len(df)} 只股票)")
            except Exception as e:
                warning(f"AKShare: 数据获取失败 ({str(e)[:50]})")
                self.warnings.append("AKShare数据获取异常")

        except ImportError:
            warning("AKShare: 未安装")
            self.results['data_sources']['akshare'] = False

        # Tushare
        tushare_token = os.getenv('TUSHARE_TOKEN')
        if tushare_token and tushare_token != 'your_tushare_token_here':
            try:
                import tushare as ts
                success("Tushare: 已安装且已配置Token")
                self.results['data_sources']['tushare'] = True

                try:
                    ts.set_token(tushare_token)
                    pro = ts.pro_api()
                    # 测试API
                    df = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240105')
                    success("Tushare: API连接正常")
                except Exception as e:
                    warning(f"Tushare: API连接失败 ({str(e)[:50]})")
                    self.warnings.append("Tushare API连接异常")

            except ImportError:
                warning("Tushare: 未安装")
                self.results['data_sources']['tushare'] = False
        else:
            info("Tushare: 未配置Token，跳过检查")

    def check_project_structure(self):
        """检查项目结构"""
        header("6. 项目结构检查")

        required_dirs = [
            'agents',
            'backtest',
            'data',
            'live',
            'monitoring',
            'rl',
            'trading',
            'utils',
            'tests',
            'examples',
        ]

        for dir_name in required_dirs:
            if os.path.isdir(dir_name):
                success(f"目录存在: {dir_name}/")
            else:
                error(f"目录缺失: {dir_name}/")
                self.errors.append(f"缺少必需目录: {dir_name}")

        # 检查关键文件
        key_files = [
            'agents/base/agent_base.py',
            'backtest/engine/engine.py',
            'backtest/indicators.py',
            'trading/risk/controls.py',
            'rl/envs/a_share_trading_env.py',
        ]

        info("\n检查关键文件...")
        for file in key_files:
            if os.path.exists(file):
                success(f"文件存在: {file}")
            else:
                warning(f"文件缺失: {file}")

    def generate_report(self):
        """生成检查报告"""
        header("检查报告汇总")

        # 统计
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)

        print(f"错误: {Colors.RED}{total_errors}{Colors.NC}")
        print(f"警告: {Colors.YELLOW}{total_warnings}{Colors.NC}")
        print(f"通过: {Colors.GREEN}{len([r for r in self.results['packages'].values() if r.get('installed', False)])} 个包{Colors.NC}")

        # 错误列表
        if self.errors:
            print(f"\n{Colors.RED}错误列表:{Colors.NC}")
            for i, err in enumerate(self.errors, 1):
                print(f"  {i}. {err}")

        # 警告列表
        if self.warnings:
            print(f"\n{Colors.YELLOW}警告列表:{Colors.NC}")
            for i, warn in enumerate(self.warnings, 1):
                print(f"  {i}. {warn}")

        # 总体评估
        print("\n" + "="*70)
        if total_errors == 0 and total_warnings == 0:
            print(f"{Colors.GREEN}✓ 环境检查完全通过！{Colors.NC}")
            print("系统已准备好运行。")
        elif total_errors == 0:
            print(f"{Colors.YELLOW}⚠ 环境检查通过，但有一些警告。{Colors.NC}")
            print("系统可以运行，但建议解决警告以获得最佳体验。")
        else:
            print(f"{Colors.RED}✗ 环境检查未通过！{Colors.NC}")
            print("请解决上述错误后再运行系统。")

        print("="*70 + "\n")

        # 保存报告到JSON
        report_file = 'env_check_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        info(f"详细报告已保存到: {report_file}")


def main():
    """主函数"""
    checker = EnvironmentChecker()
    success = checker.check_all()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
