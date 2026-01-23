"""
Rust回测引擎Python接口
集成Rust高性能回测引擎到Python系统
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class RustBacktestEngine:
    """
    Rust回测引擎的Python包装器

    注意：这是Python包装器，实际的Rust实现需要先编译
    """

    def __init__(
        self,
        initial_cash: float = 1000000.0,
        commission: float = 0.0003,
        slippage: float = 0.0001,
    ):
        """
        初始化Rust回测引擎

        Args:
            initial_cash: 初始资金
            commission: 手续费率
            slippage: 滑点率
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

        # 尝试导入Rust引擎
        try:
            import quanta_rust_engine

            self._rust_engine = quanta_rust_engine.BacktestEngine(
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage,
            )
            self._use_rust = True
        except ImportError:
            # 如果Rust引擎未编译，使用Python实现
            from backtest.engine.engine import BacktestEngine

            self._python_engine = BacktestEngine(
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage,
            )
            self._use_rust = False

    def load_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        加载市场数据

        Args:
            data: {symbol: DataFrame} 字典
        """
        if self._use_rust:
            # 使用Rust引擎
            rust_data = quanta_rust_engine.MarketData()

            for symbol, df in data.items():
                for _, row in df.iter():
                    timestamp = int(row["datetime"].timestamp())
                    rust_data.add_bar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=float(row["open"]),
                        high=float(row["high"]),
                        low=float(row["low"]),
                        close=float(row["close"]),
                        volume=int(row["volume"]),
                    )

            self._rust_engine.load_data(rust_data)
        else:
            # 使用Python引擎
            self._python_engine.data = data

    def run(self, strategy: Any = None) -> Dict[str, float]:
        """
        运行回测

        Args:
            strategy: 交易策略

        Returns:
            回测结果字典
        """
        if self._use_rust:
            # 使用Rust引擎
            results = self._rust_engine.run_backtest(
                self._rust_engine.data, strategy or self._default_strategy
            )
            return dict(results)
        else:
            # 使用Python引擎
            return self._python_engine.run(strategy)

    def _default_strategy(self, event):
        """默认策略（用于测试）"""
        pass

    @property
    def is_rust_enabled(self) -> bool:
        """检查是否启用了Rust引擎"""
        return self._use_rust


def check_rust_availability() -> bool:
    """
    检查Rust引擎是否可用

    Returns:
        True如果Rust引擎已编译并可导入
    """
    try:
        import quanta_rust_engine

        return True
    except ImportError:
        return False


def build_rust_engine() -> bool:
    """
    构建Rust引擎

    Returns:
        True如果构建成功
    """
    import subprocess
    import sys

    rust_dir = os.path.join(os.path.dirname(__file__), "..", "..", "rust_engine")

    if not os.path.exists(rust_dir):
        print(f"Rust引擎目录不存在: {rust_dir}")
        return False

    try:
        # 使用maturin构建Python扩展
        result = subprocess.run(
            [sys.executable, "-m", "maturin", "develop", "--release"],
            cwd=rust_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Rust引擎构建成功")
            return True
        else:
            print(f"Rust引擎构建失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"构建Rust引擎时出错: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    # 检查Rust引擎是否可用
    if check_rust_availability():
        print("✓ Rust引擎已安装")
    else:
        print("✗ Rust引擎未安装")
        print("正在构建Rust引擎...")
        if build_rust_engine():
            print("✓ Rust引擎构建成功")
        else:
            print("✗ Rust引擎构建失败，将使用Python引擎")

    # 创建引擎
    engine = RustBacktestEngine(initial_cash=1000000)

    print(f"\n引擎信息:")
    print(f"  使用Rust: {engine.is_rust_enabled}")
    print(f"  初始资金: {engine.initial_cash:,.2f}")
    print(f"  手续费率: {engine.commission:.4f}")
