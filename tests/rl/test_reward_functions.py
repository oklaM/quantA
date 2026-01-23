"""
奖励函数测试
"""

import numpy as np
import pytest

from rl.rewards.reward_functions import (
    AsymmetricReward,
    CalmarRatioReward,
    CompositeReward,
    MaxDrawdownReward,
    RiskAdjustedReward,
    SharpeRatioReward,
    SimpleProfitReward,
    SortinoRatioReward,
    TransactionCostAwareReward,
    create_reward_function,
)


@pytest.mark.rl
class TestSimpleProfitReward:
    """测试简单利润奖励函数"""

    def test_profit_calculation(self):
        """测试利润计算"""
        reward_func = SimpleProfitReward()

        result = reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert result.reward == 50000.0
        assert result.components["profit"] == 50000.0

    def test_loss_calculation(self):
        """测试亏损计算"""
        reward_func = SimpleProfitReward()

        result = reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=950000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert result.reward == -50000.0


@pytest.mark.rl
class TestSharpeRatioReward:
    """测试夏普比率奖励函数"""

    def test_sharpe_calculation(self):
        """测试夏普比率计算"""
        reward_func = SharpeRatioReward(window_size=10)

        # 模拟多次计算
        for i in range(20):
            reward_func.calculate(
                action=0,
                current_price=100.0,
                portfolio_value=1000000 + i * 1000,
                previous_value=1000000 + (i - 1) * 1000 if i > 0 else 1000000,
                position_size=1000,
            )

        # 最终奖励应该是夏普比率
        result = reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=1020000.0,
            previous_value=1015000.0,
            position_size=1000,
        )

        assert "sharpe_ratio" in result.components
        assert isinstance(result.reward, float)


@pytest.mark.rl
class TestRiskAdjustedReward:
    """测试风险调整奖励函数"""

    def test_risk_adjusted_calculation(self):
        """测试风险调整计算"""
        reward_func = RiskAdjustedReward(
            profit_weight=1.0, volatility_penalty=0.5, drawdown_penalty=0.3
        )

        # 有利润的情况
        result = reward_func.calculate(
            action=0,
            current_price=105.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        # 检查各组件
        assert "profit" in result.components
        assert "volatility_penalty" in result.components
        assert "drawdown_penalty" in result.components

        # 总奖励应该小于纯利润
        assert result.reward < 50000.0  # 纯利润


@pytest.mark.rl
class TestMaxDrawdownReward:
    """测试最大回撤奖励函数"""

    def test_drawdown_calculation(self):
        """测试回撤计算"""
        reward_func = MaxDrawdownReward(drawdown_penalty_weight=10.0)

        # 第一步：上涨，创建新峰值
        result1 = reward_func.calculate(
            action=1,
            current_price=105.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert result1.info["peak_value"] == 1050000.0

        # 第二步：下跌，产生回撤
        result2 = reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=1000000.0,
            previous_value=1050000.0,
            position_size=1000,
        )

        assert result2.info["peak_value"] == 1050000.0
        assert result2.components["drawdown_penalty"] < 0  # 应该有惩罚
        assert "drawdown" in result2.info


@pytest.mark.rl
class TestTransactionCostAwareReward:
    """测试交易成本感知奖励函数"""

    def test_transaction_penalty(self):
        """测试交易惩罚"""
        reward_func = TransactionCostAwareReward(
            base_reward_weight=1.0, transaction_cost=100.0, holding_reward=10.0
        )

        # 买入：应该有交易成本惩罚
        result_buy = reward_func.calculate(
            action=1,
            current_price=100.0,
            portfolio_value=1000000.0,
            previous_value=1000000.0,
            position_size=0,
        )

        assert result_buy.components["transaction_penalty"] == -100.0

        # 持有：应该有持仓奖励
        result_hold = reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=1000000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert result_hold.components["holding_bonus"] == 10.0
        assert result_hold.info["holding_duration"] == 1


@pytest.mark.rl
class TestAsymmetricReward:
    """测试不对称奖励函数"""

    def test_profit_reward(self):
        """测试盈利奖励"""
        reward_func = AsymmetricReward(profit_weight=1.0, loss_weight=2.0)

        result = reward_func.calculate(
            action=0,
            current_price=105.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert result.reward == 50000.0
        assert result.info["is_profitable"] is True

    def test_loss_penalty(self):
        """测试亏损惩罚（应该更大）"""
        reward_func = AsymmetricReward(profit_weight=1.0, loss_weight=2.0)

        result = reward_func.calculate(
            action=0,
            current_price=95.0,
            portfolio_value=950000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        # 损失50000，权重为2，所以奖励应该是-100000
        assert result.reward == -100000.0
        assert result.info["is_profitable"] is False


@pytest.mark.rl
class TestSortinoRatioReward:
    """测试Sortino比率奖励函数"""

    def test_sortino_calculation(self):
        """测试Sortino比率计算"""
        reward_func = SortinoRatioReward(target_return=0.0, window_size=10)

        # 添加一些收益历史
        for i in range(15):
            reward_func.calculate(
                action=0,
                current_price=100.0,
                portfolio_value=1000000 + i * 10000 * (1 if i % 2 == 0 else -1),
                previous_value=1000000,
                position_size=1000,
            )

        # Sortino比率应该在合理范围内
        result = reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=1010000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert "sortino_ratio" in result.components
        assert isinstance(result.reward, float)


@pytest.mark.rl
class TestCalmarRatioReward:
    """测试Calmar比率奖励函数"""

    def test_calmar_calculation(self):
        """测试Calmar比率计算"""
        reward_func = CalmarRatioReward()

        # 模拟多个时间步
        for i in range(10):
            reward_func.calculate(
                action=0,
                current_price=100.0 + i * 0.5,
                portfolio_value=1000000 + i * 50000,
                previous_value=1000000 + (i - 1) * 50000 if i > 0 else 1000000,
                position_size=1000,
            )

        # 检查结果
        result = reward_func.calculate(
            action=0,
            current_price=105.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert "calmar_ratio" in result.components
        assert "annualized_return" in result.components
        assert "max_drawdown" in result.components


@pytest.mark.rl
class TestCompositeReward:
    """测试组合奖励函数"""

    def test_composite_reward(self):
        """测试组合奖励"""
        # 创建多个奖励函数
        profit_reward = SimpleProfitReward()
        drawdown_reward = MaxDrawdownReward(drawdown_penalty_weight=5.0)

        # 组合奖励
        composite = CompositeReward(
            [
                (profit_reward, 0.7),  # 利润权重70%
                (drawdown_reward, 0.3),  # 回撤权重30%
            ]
        )

        result = composite.calculate(
            action=0,
            current_price=105.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        # 检查组合结果
        assert "SimpleProfitReward_profit" in result.components
        assert "MaxDrawdownReward_drawdown_penalty" in result.components
        assert "total_reward" in result.components


@pytest.mark.rl
class TestRewardFunctionFactory:
    """测试奖励函数工厂"""

    def test_create_simple_reward(self):
        """测试创建简单奖励"""
        reward_func = create_reward_function("simple")

        assert isinstance(reward_func, SimpleProfitReward)

    def test_create_sharpe_reward(self):
        """测试创建夏普比率奖励"""
        reward_func = create_reward_function("sharpe", window_size=20)

        assert isinstance(reward_func, SharpeRatioReward)

    def test_create_invalid_reward(self):
        """测试创建无效奖励"""
        with pytest.raises(ValueError):
            create_reward_function("invalid_type")

    def test_create_with_params(self):
        """测试带参数创建"""
        reward_func = create_reward_function(
            "transaction_aware", transaction_cost=200.0, holding_reward=20.0
        )

        assert isinstance(reward_func, TransactionCostAwareReward)


@pytest.mark.rl
class TestRewardReset:
    """测试奖励函数重置"""

    def test_simple_reward_reset(self):
        """测试简单奖励重置"""
        reward_func = SimpleProfitReward()

        # 添加一些历史
        reward_func.calculate(
            action=0,
            current_price=100.0,
            portfolio_value=1050000.0,
            previous_value=1000000.0,
            position_size=1000,
        )

        assert len(reward_func.history) == 1

        # 重置
        reward_func.reset()

        assert len(reward_func.history) == 0

    def test_sharpe_reward_reset(self):
        """测试夏普比率奖励重置"""
        reward_func = SharpeRatioReward()

        # 添加一些历史
        for i in range(10):
            reward_func.calculate(
                action=0,
                current_price=100.0,
                portfolio_value=1000000 + i * 1000,
                previous_value=1000000 + (i - 1) * 1000 if i > 0 else 1000000,
                position_size=1000,
            )

        assert len(reward_func.returns_history) == 10

        # 重置
        reward_func.reset()

        assert len(reward_func.returns_history) == 0
