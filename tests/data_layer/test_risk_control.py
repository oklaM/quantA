#!/usr/bin/env python3
"""
风险控制测试脚本
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict

# 设置 Python 路径
sys.path.insert(0, os.path.abspath('.'))

# 导入风险控制模块
from trading.risk import ActionType, OrderRequest, RiskController


def test_risk_control():
    """测试风险控制功能"""
    print("="*70)
    print("风险控制测试")
    print("="*70)

    # 创建风险控制器配置
    config = {
        'min_available_cash': 1000000,  # 最少保留100万
        'max_single_order_amount': 500000,  # 单笔50万
        'max_single_position_ratio': 0.25,  # 单一持仓25%
        'max_daily_loss_ratio': 0.03,  # 日亏损3%
        'max_positions': 30,  # 最多持仓30只
        'max_daily_volume': 50000000,  # 日交易量5000万
    }

    # 创建风险控制器
    controller = RiskController(config=config)

    # 测试上下文
    context = {
        'account': {
            'total_asset': 10000000,  # 1000万
            'available_cash': 3000000,  # 300万
        },
        'portfolio': {
            'total_value': 10000000,
            'positions': {
                '600000.SH': 2000000,  # 已持仓200万
                '000001.SZ': 1500000,  # 已持仓150万
            }
        },
        'initial_cash': 10000000,
    }

    print("\n测试1: 正常订单检查")
    # 测试正常订单
    allowed, rejects = controller.validate_order(
        symbol='600036.SH',
        action='buy',
        quantity=10000,
        price=20.0,
        context=context
    )
    print(f"  正常订单: {'✓ 通过' if allowed else '✗ 拒绝'}")
    if not allowed:
        print(f"    拒绝原因: {rejects}")

    print("\n测试2: 单笔金额超限检查")
    # 测试单笔金额超限
    allowed, rejects = controller.validate_order(
        symbol='600036.SH',
        action='buy',
        quantity=50000,  # 1000万，超过单笔限制
        price=20.0,
        context=context
    )
    print(f"  大额订单: {'✓ 通过' if allowed else '✗ 拒绝'}")
    if not allowed:
        print(f"    拒绝原因: {rejects}")

    print("\n测试3: 持仓比例超限检查")
    # 测试持仓比例超限
    allowed, rejects = controller.validate_order(
        symbol='600000.SH',  # 已经持仓200万，再买100万，总计300万
        action='buy',
        quantity=50000,
        price=20.0,
        context=context
    )
    print(f"  增持订单: {'✓ 通过' if allowed else '✗ 拒绝'}")
    if not allowed:
        print(f"    拒绝原因: {rejects}")

    print("\n测试4: 当日亏损限制检查")
    # 模拟当日亏损
    daily_stats = {
        'daily_pnl': -400000,  # 亏损40万
        'initial_asset': 10000000,
        'traded_volume': 0,
    }
    context['daily_stats'] = daily_stats

    allowed, rejects = controller.validate_order(
        symbol='600036.SH',
        action='buy',
        quantity=10000,
        price=20.0,
        context=context
    )
    print(f" 亏损后的订单: {'✓ 通过' if allowed else '✗ 拒绝'}")
    if not allowed:
        print(f"    拒绝原因: {rejects}")

    # 获取风控统计
    stats = controller.get_statistics()
    print(f"\n风险控制统计:")
    print(f"  总检查次数: {stats['total_checks']}")
    print(f"  总拒绝次数: {stats['total_rejects']}")
    print(f"  拒绝率: {stats['reject_ratio']:.2%}")
    print(f"  活跃规则数: {stats['active_rules']}")

    # 测试股票黑名单
    print("\n测试5: 股票黑名单检查")
    config_with_blacklist = config.copy()
    config_with_blacklist['stock_blacklist'] = ['ST.*', '.*ST']
    controller_blacklist = RiskController(config=config_with_blacklist)

    allowed, rejects = controller_blacklist.validate_order(
        symbol='ST康美',
        action='buy',
        quantity=1000,
        price=5.0,
        context=context
    )
    print(f"  ST股票交易: {'✓ 通过' if allowed else '✗ 拒绝'}")
    if not allowed:
        print(f"    拒绝原因: {rejects}")

    print("\n" + "="*70)
    print("风险控制测试完成！")
    print("="*70)

    return True

def test_risk_control_edge_cases():
    """测试风险控制边界情况"""
    print("\n" + "="*70)
    print("风险控制边界情况测试")
    print("="*70)

    # 创建保守的风控设置
    config = {
        'min_available_cash': 100000,
        'max_single_order_amount': 100000,  # 单笔10万
        'max_single_position_ratio': 0.1,  # 10%单股持仓限制
        'max_daily_loss_ratio': 0.01,  # 1%日亏损限制
    }

    controller = RiskController(config=config)

    # 测试边界情况
    contexts = [
        {
            'name': '资金不足',
            'context': {
                'account': {'total_asset': 100000, 'available_cash': 5000},
                'portfolio': {'total_value': 100000, 'positions': {}},
                'initial_cash': 100000,
            },
            'order': OrderRequest(
                symbol='600000.SH',
                action=ActionType.BUY,
                quantity=10000,
                price=10.0,  # 10万
            )
        },
        {
            'name': '零金额订单',
            'context': {
                'account': {'total_asset': 100000, 'available_cash': 100000},
                'portfolio': {'total_value': 100000, 'positions': {}},
                'initial_cash': 100000,
            },
            'order': OrderRequest(
                symbol='600000.SH',
                action=ActionType.BUY,
                quantity=0,
                price=10.0,
            )
        },
        {
            'name': '无限价格',
            'context': {
                'account': {'total_asset': 100000, 'available_cash': 100000},
                'portfolio': {'total_value': 100000, 'positions': {}},
                'initial_cash': 100000,
            },
            'order': OrderRequest(
                symbol='600000.SH',
                action=ActionType.BUY,
                quantity=1000,
                price=100000.0,  # 100万
            )
        }
    ]

    for test_case in contexts:
        print(f"\n测试: {test_case['name']}")
        allowed, rejects = controller.validate_order(
            symbol=test_case['order'].symbol,
            action=test_case['order'].action.value,
            quantity=test_case['order'].quantity,
            price=test_case['order'].price,
            context=test_case['context']
        )
        print(f"  结果: {'✓ 通过' if allowed else '✗ 拒绝'}")
        if not allowed:
            print(f"    拒绝原因: {rejects}")

    return True

def main():
    """主测试函数"""
    try:
        # 运行基础测试
        test_risk_control()

        # 运行边界情况测试
        test_risk_control_edge_cases()

        print("\n🎉 所有风险控制测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)