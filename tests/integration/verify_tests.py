#!/usr/bin/env python
"""
集成测试验证脚本
验证所有新增的集成测试能够正确收集和运行
"""

import subprocess
import sys


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"测试: {description}")
    print(f"{'='*60}")
    print(f"命令: {cmd}")
    print()

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def main():
    """主函数"""
    print("="*60)
    print("quantA 集成测试验证")
    print("="*60)

    tests = [
        # 收集所有测试
        (
            "python -m pytest tests/integration/ --collect-only -q | tail -1",
            "收集所有集成测试"
        ),

        # 数据管道测试
        (
            "python -m pytest tests/integration/test_data_pipeline.py::TestDataAcquisitionIntegration::test_data_manager_initialization -v --no-cov",
            "数据管理器初始化测试"
        ),

        # Agent协作测试
        (
            "python -m pytest tests/integration/test_agent_collaboration.py::TestAgentRegistration::test_register_multiple_agents -v --no-cov",
            "Agent注册测试"
        ),

        # 回测流程测试
        (
            "python -m pytest tests/integration/test_backtest_workflow.py::TestBacktestInitialization::test_engine_initialization -v --no-cov",
            "回测引擎初始化测试"
        ),

        # RL训练测试（如果stable-baselines3可用）
        (
            "python -m pytest tests/integration/test_rl_training.py::TestEnvironmentCreation::test_env_initialization -v --no-cov",
            "RL环境初始化测试"
        ),

        # 跨模块测试
        (
            "python -m pytest tests/integration/test_cross_module_integration.py::TestDataToStrategyIntegration::test_data_indicators_strategy_pipeline -v --no-cov",
            "数据到策略集成测试"
        ),
    ]

    results = []
    for cmd, description in tests:
        success = run_command(cmd, description)
        results.append((description, success))

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for description, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {description}")

    total = len(results)
    passed = sum(1 for _, s in results if s)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n✓ 所有验证测试通过！")
        return 0
    else:
        print(f"\n✗ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
