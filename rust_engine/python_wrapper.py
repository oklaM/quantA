"""
Rust执行引擎Python包装器

提供Python接口调用Rust执行引擎
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

# 尝试导入编译好的Rust模块
try:
    # 这需要先编译Rust模块: cargo build --release
    # 然后将编译好的.so/.pyd文件放到合适的位置
    import quanta_rust_engine
    RUST_ENGINE_AVAILABLE = True
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    print("警告: Rust执行引擎未编译或不可用")


class RustOrderManager:
    """Rust订单管理器包装器"""

    def __init__(self):
        if not RUST_ENGINE_AVAILABLE:
            raise RuntimeError("Rust执行引擎不可用，请先编译: cd rust_engine && cargo build --release")

        self._inner = quanta_rust_engine.OrderManager()

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: int,
        price: Optional[float] = None,
    ) -> str:
        """
        创建订单

        Args:
            symbol: 股票代码
            side: 方向 ('buy' 或 'sell')
            order_type: 类型 ('market' 或 'limit')
            quantity: 数量
            price: 价格（限价单必需）

        Returns:
            订单ID
        """
        side_enum = quanta_rust_engine.OrderSide.Buy if side.lower() == 'buy' else quanta_rust_engine.OrderSide.Sell
        type_enum = quanta_rust_engine.OrderType.Limit if order_type.lower() == 'limit' else quanta_rust_engine.OrderType.Market

        return self._inner.create_order(symbol, side_enum, type_enum, quantity, price)

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单"""
        order = self._inner.get_order(order_id)
        if order is None:
            return None

        return {
            'id': order.id,
            'symbol': order.symbol,
            'side': 'buy' if order.side.name == 'Buy' else 'sell',
            'type': order.order_type.name.lower(),
            'quantity': order.quantity,
            'price': order.price,
            'filled_quantity': order.filled_quantity,
            'avg_fill_price': order.avg_fill_price,
            'status': order.status.name.lower(),
        }

    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        return self._inner.cancel_order(order_id)

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """获取活跃订单"""
        orders = self._inner.get_active_orders()
        return [
            {
                'id': order.id,
                'symbol': order.symbol,
                'side': 'buy' if order.side.name == 'Buy' else 'sell',
                'type': order.order_type.name.lower(),
                'quantity': order.quantity,
                'price': order.price,
                'filled_quantity': order.filled_quantity,
                'avg_fill_price': order.avg_fill_price,
                'status': order.status.name.lower(),
            }
            for order in orders
        ]

    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self._inner.get_stats()


class RustOrderBook:
    """Rust订单簿包装器"""

    def __init__(self, symbol: str):
        if not RUST_ENGINE_AVAILABLE:
            raise RuntimeError("Rust执行引擎不可用，请先编译: cd rust_engine && cargo build --release")

        self._inner = quanta_rust_engine.OrderBook(symbol)

    def get_depth(self, depth: int = 5) -> Dict[str, Any]:
        """
        获取订单簿深度

        Args:
            depth: 深度档位

        Returns:
            深度数据
        """
        return self._inner.get_depth(depth)

    def get_best_bid_ask(self) -> tuple[Optional[float], Optional[float]]:
        """获取最优买卖价"""
        return self._inner.get_best_bid_ask()


def compile_rust_engine(debug: bool = False) -> bool:
    """
    编译Rust执行引擎

    Args:
        debug: 是否为调试模式

    Returns:
        编译是否成功
    """
    import subprocess
    import sys

    rust_dir = Path(__file__).parent

    print(f"正在编译Rust执行引擎...")

    try:
        cmd = ["cargo", "build"]
        if not debug:
            cmd.append("--release")

        result = subprocess.run(
            cmd,
            cwd=rust_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"编译失败:")
            print(result.stdout)
            print(result.stderr)
            return False

        print("编译成功!")

        # 尝试复制编译好的文件到Python路径
        if debug:
            so_file = list(rust_dir.glob("target/debug/libquanta_rust_engine.*"))
        else:
            so_file = list(rust_dir.glob("target/release/libquanta_rust_engine.*"))

        if so_file:
            print(f"编译产物: {so_file[0]}")
            print("请确保该文件在Python导入路径中")

        return True

    except Exception as e:
        print(f"编译出错: {e}")
        return False


def check_rust_engine() -> bool:
    """检查Rust引擎是否可用"""
    return RUST_ENGINE_AVAILABLE


__all__ = [
    'RustOrderManager',
    'RustOrderBook',
    'compile_rust_engine',
    'check_rust_engine',
    'RUST_ENGINE_AVAILABLE',
]
