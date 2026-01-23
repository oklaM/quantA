"""
强化学习奖励函数模块
提供多种奖励函数设计和实现
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RewardResult:
    """奖励计算结果"""

    reward: float
    components: Dict[str, float]
    info: Dict[str, Any]


class BaseRewardFunction(ABC):
    """奖励函数基类"""

    def __init__(self, **kwargs):
        self.config = kwargs
        self.history: List[RewardResult] = []

    @abstractmethod
    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """
        计算奖励

        Args:
            action: 执行的动作 (0=持有, 1=买入, 2=卖出)
            current_price: 当前价格
            portfolio_value: 当前组合价值
            previous_value: 上一步组合价值
            position_size: 持仓数量
            **kwargs: 其他参数

        Returns:
            RewardResult
        """
        pass

    def reset(self):
        """重置状态"""
        self.history.clear()


class SimpleProfitReward(BaseRewardFunction):
    """
    简单利润奖励函数

    奖励 = 当前组合价值 - 上一步组合价值
    """

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        profit = portfolio_value - previous_value

        components = {
            "profit": profit,
        }

        info = {
            "action": action,
            "portfolio_value": portfolio_value,
            "profit": profit,
        }

        result = RewardResult(reward=profit, components=components, info=info)
        self.history.append(result)
        return result


class SharpeRatioReward(BaseRewardFunction):
    """
    基于夏普比率的奖励函数

    考虑收益和风险的平衡
    """

    def __init__(self, window_size: int = 20, risk_free_rate: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.returns_history: List[float] = []

    def reset(self):
        """重置状态"""
        super().reset()
        self.returns_history.clear()

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        # 计算收益率
        if previous_value > 0:
            ret = (portfolio_value - previous_value) / previous_value
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # 保持历史窗口
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # 计算夏普比率
        if len(self.returns_history) >= 2:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return > 0:
                sharpe = (mean_return * 252 - self.risk_free_rate) / (std_return * np.sqrt(252))
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        components = {
            "return": ret,
            "sharpe_ratio": sharpe,
        }

        info = {
            "action": action,
            "mean_return": np.mean(self.returns_history) if self.returns_history else 0,
            "std_return": np.std(self.returns_history) if len(self.returns_history) >= 2 else 0,
        }

        result = RewardResult(reward=sharpe, components=components, info=info)
        self.history.append(result)
        return result


class RiskAdjustedReward(BaseRewardFunction):
    """
    风险调整奖励函数

    考虑收益、波动率和最大回撤
    """

    def __init__(
        self,
        profit_weight: float = 1.0,
        volatility_penalty: float = 0.5,
        drawdown_penalty: float = 0.3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.profit_weight = profit_weight
        self.volatility_penalty = volatility_penalty
        self.drawdown_penalty = drawdown_penalty

        self.returns_history: List[float] = []
        self.peak_value: float = 0.0

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        # 计算收益率
        if previous_value > 0:
            ret = (portfolio_value - previous_value) / previous_value
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # 计算回撤
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            drawdown = 0.0

        # 计算波动率
        if len(self.returns_history) >= 20:
            volatility = np.std(self.returns_history[-20:])
        else:
            volatility = 0.0

        # 计算风险调整后的奖励
        profit_component = self.profit_weight * ret * 100  # 转换为百分比
        volatility_component = -self.volatility_penalty * volatility * 100
        drawdown_component = -self.drawdown_penalty * drawdown * 100

        total_reward = profit_component + volatility_component + drawdown_component

        components = {
            "profit": profit_component,
            "volatility_penalty": volatility_component,
            "drawdown_penalty": drawdown_component,
        }

        info = {
            "return": ret,
            "volatility": volatility,
            "drawdown": drawdown,
            "peak_value": self.peak_value,
        }

        return RewardResult(reward=total_reward, components=components, info=info)


class MaxDrawdownReward(BaseRewardFunction):
    """
    最大回撤奖励函数

    惩罚最大回撤，鼓励稳定增长
    """

    def __init__(self, drawdown_penalty_weight: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.peak_value: float = 0.0

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # 计算回撤
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            drawdown = 0.0

        # 奖励：利润减去回撤惩罚
        profit = portfolio_value - previous_value
        reward = profit - self.drawdown_penalty_weight * drawdown * portfolio_value

        components = {
            "profit": profit,
            "drawdown_penalty": -self.drawdown_penalty_weight * drawdown * portfolio_value,
        }

        info = {
            "drawdown": drawdown,
            "peak_value": self.peak_value,
            "portfolio_value": portfolio_value,
        }

        return RewardResult(reward=reward, components=components, info=info)


class TransactionCostAwareReward(BaseRewardFunction):
    """
    考虑交易成本的奖励函数

    惩罚频繁交易，鼓励长期持有
    """

    def __init__(
        self,
        base_reward_weight: float = 1.0,
        transaction_cost: float = 100.0,
        holding_reward: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_reward_weight = base_reward_weight
        self.transaction_cost = transaction_cost
        self.holding_reward = holding_reward

        self.last_action: Optional[int] = None
        self.holding_duration: int = 0

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        # 基础利润奖励
        profit = portfolio_value - previous_value
        base_reward = self.base_reward_weight * profit

        # 交易成本惩罚
        transaction_penalty = 0.0
        holding_bonus = 0.0

        if action in [1, 2]:  # 买入或卖出
            transaction_penalty = -self.transaction_cost
            self.holding_duration = 0
        elif action == 0 and position_size > 0:  # 持有且有持仓
            self.holding_duration += 1
            holding_bonus = self.holding_reward * self.holding_duration

        total_reward = base_reward + transaction_penalty + holding_bonus

        components = {
            "base_reward": base_reward,
            "transaction_penalty": transaction_penalty,
            "holding_bonus": holding_bonus,
        }

        info = {
            "action": action,
            "holding_duration": self.holding_duration,
            "position_size": position_size,
        }

        self.last_action = action

        return RewardResult(reward=total_reward, components=components, info=info)


class AsymmetricReward(BaseRewardFunction):
    """
    不对称奖励函数

    惩罚损失的权重大于奖励盈利的权重（损失厌恶）
    """

    def __init__(self, profit_weight: float = 1.0, loss_weight: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.profit_weight = profit_weight
        self.loss_weight = loss_weight

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        profit = portfolio_value - previous_value

        # 不对称奖励
        if profit > 0:
            reward = self.profit_weight * profit
        else:
            reward = self.loss_weight * profit  # profit为负，乘以大于1的权重增加惩罚

        components = {
            "raw_profit": profit,
            "weighted_reward": reward,
        }

        info = {
            "action": action,
            "profit": profit,
            "is_profitable": profit > 0,
        }

        return RewardResult(reward=reward, components=components, info=info)


class SortinoRatioReward(BaseRewardFunction):
    """
    基于Sortino比率的奖励函数

    类似夏普比率，但只考虑下行波动率
    """

    def __init__(self, target_return: float = 0.0, window_size: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.target_return = target_return
        self.window_size = window_size
        self.returns_history: List[float] = []

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        # 计算收益率
        if previous_value > 0:
            ret = (portfolio_value - previous_value) / previous_value
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # 保持历史窗口
        if len(self.returns_history) > self.window_size:
            self.returns_history.pop(0)

        # 计算下行偏差
        if len(self.returns_history) >= 2:
            excess_returns = np.array(self.returns_history) - self.target_return
            downside_returns = excess_returns[excess_returns < 0]
            downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.001

            # Sortino比率
            mean_excess_return = np.mean(excess_returns)
            sortino = mean_excess_return / downside_deviation
        else:
            sortino = 0.0

        components = {
            "return": ret,
            "sortino_ratio": sortino,
        }

        info = {
            "mean_excess_return": (
                np.mean(np.array(self.returns_history) - self.target_return)
                if self.returns_history
                else 0
            ),
            "downside_deviation": (
                np.std([r for r in self.returns_history if r < self.target_return])
                if self.returns_history
                else 0
            ),
        }

        return RewardResult(reward=sortino, components=components, info=info)


class CalmarRatioReward(BaseRewardFunction):
    """
    基于Calmar比率的奖励函数

    Calmar比率 = 年化收益 / 最大回撤
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.returns_history: List[float] = []
        self.peak_value: float = 0.0
        self.start_value: float = 0.0

    def reset(self):
        """重置状态"""
        super().reset()
        self.returns_history.clear()
        self.peak_value = 0.0
        self.start_value = 0.0

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        # 初始化起始值
        if self.start_value == 0:
            self.start_value = previous_value

        # 计算收益率
        if previous_value > 0:
            ret = (portfolio_value - previous_value) / previous_value
        else:
            ret = 0.0

        self.returns_history.append(ret)

        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # 计算年化收益
        days = len(self.returns_history)
        if days > 0 and self.start_value > 0:
            total_return = (portfolio_value - self.start_value) / self.start_value
            annualized_return = (1 + total_return) ** (252 / days) - 1 if days >= 1 else 0
        else:
            annualized_return = 0.0

        # 计算最大回撤
        if self.peak_value > 0:
            max_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        else:
            max_drawdown = 0.0

        # Calmar比率
        if max_drawdown > 0:
            calmar = annualized_return / max_drawdown
        else:
            calmar = 0.0

        components = {
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
        }

        info = {
            "days": days,
            "total_return": (
                (portfolio_value - self.start_value) / self.start_value
                if self.start_value > 0
                else 0
            ),
        }

        return RewardResult(reward=calmar, components=components, info=info)


class CompositeReward(BaseRewardFunction):
    """
    组合奖励函数

    可以组合多个奖励函数
    """

    def __init__(self, reward_functions: List[tuple], **kwargs):
        """
        Args:
            reward_functions: [(reward_func, weight), ...] 奖励函数和权重列表
        """
        super().__init__(**kwargs)
        self.reward_functions = reward_functions

    def calculate(
        self,
        action: int,
        current_price: float,
        portfolio_value: float,
        previous_value: float,
        position_size: int,
        **kwargs,
    ) -> RewardResult:
        """计算奖励"""
        total_reward = 0.0
        all_components = {}
        all_info = {}

        for reward_func, weight in self.reward_functions:
            result = reward_func.calculate(
                action=action,
                current_price=current_price,
                portfolio_value=portfolio_value,
                previous_value=previous_value,
                position_size=position_size,
                **kwargs,
            )

            total_reward += weight * result.reward

            # 合并组件和信息
            for key, value in result.components.items():
                all_components[f"{reward_func.__class__.__name__}_{key}"] = value

            for key, value in result.info.items():
                all_info[f"{reward_func.__class__.__name__}_{key}"] = value

        all_components["total_reward"] = total_reward

        return RewardResult(reward=total_reward, components=all_components, info=all_info)


def create_reward_function(reward_type: str, **kwargs) -> BaseRewardFunction:
    """
    创建奖励函数的工厂函数

    Args:
        reward_type: 奖励函数类型
        **kwargs: 奖励函数参数

    Returns:
        BaseRewardFunction实例
    """
    reward_classes = {
        "simple": SimpleProfitReward,
        "sharpe": SharpeRatioReward,
        "risk_adjusted": RiskAdjustedReward,
        "max_drawdown": MaxDrawdownReward,
        "transaction_aware": TransactionCostAwareReward,
        "asymmetric": AsymmetricReward,
        "sortino": SortinoRatioReward,
        "calmar": CalmarRatioReward,
    }

    reward_class = reward_classes.get(reward_type.lower())
    if reward_class is None:
        raise ValueError(f"未知的奖励函数类型: {reward_type}")

    return reward_class(**kwargs)


__all__ = [
    "BaseRewardFunction",
    "RewardResult",
    "SimpleProfitReward",
    "SharpeRatioReward",
    "RiskAdjustedReward",
    "MaxDrawdownReward",
    "TransactionCostAwareReward",
    "AsymmetricReward",
    "SortinoRatioReward",
    "CalmarRatioReward",
    "CompositeReward",
    "create_reward_function",
]
