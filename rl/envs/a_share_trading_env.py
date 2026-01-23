"""
A股交易环境
实现符合Gymnasium接口的强化学习交易环境
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None

from backtest.engine.a_share_rules import AShareRulesEngine, Order, Position
from backtest.engine.indicators import TechnicalIndicators
from utils.logging import get_logger

logger = get_logger(__name__)


class ASharesTradingEnv(gym.Env):
    """
    A股交易环境

    状态空间: 技术指标 + 账户状态
    动作空间: {买入、卖出、持有}
    奖励函数: 考虑收益、成本、风险
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
        window_size: int = 60,
    ):
        """
        Args:
            data: 历史K线数据
            initial_cash: 初始资金
            commission_rate: 佣金率
            window_size: 观察窗口大小
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium未安装，请运行: pip install gymnasium")

        super().__init__()

        # 数据
        self.data = data.reset_index(drop=True)
        self.max_steps = len(data) - window_size - 1
        self.window_size = window_size

        # 账户状态
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position: Optional[Position] = None
        self.position_size = 0  # 持仓数量

        # 交易规则
        self.commission_rate = commission_rate
        self.rules_engine = AShareRulesEngine()

        # 技术指标计算器
        self.indicators = TechnicalIndicators()

        # 当前步数
        self.current_step = 0

        # 定义动作空间: 0=持有, 1=买入, 2=卖出
        self.action_space = spaces.Discrete(3)

        # 定义状态空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_observation_shape(),),
            dtype=np.float32,
        )

        # 状态记录
        self.trade_history = []
        self.total_reward = 0.0

        logger.info("A股交易环境初始化完成")

    def _get_observation_shape(self) -> int:
        """计算观察空间维度"""
        # 价格特征: OHLCV (5)
        # 技术指标: MA(2), RSI(1), MACD(3), 布林带(3), 趋势(1), 收益率(3) = 13
        # 账户状态: 持仓比例(1), 现金比例(1) = 2
        return 5 + 13 + 2  # 20维

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        super().reset(seed=seed)

        self.current_step = 0
        self.cash = self.initial_cash
        self.position_size = 0
        self.position = None
        self.trade_history = []
        self.total_reward = 0.0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步

        Args:
            action: 动作 (0=持有, 1=买入, 2=卖出)

        Returns:
            (观察, 奖励, 终止, 截断, 信息)
        """
        # 获取当前价格
        current_idx = self.current_step + self.window_size
        current_price = self.data.iloc[current_idx]["close"]

        reward = 0.0
        trade_executed = False

        # 执行动作
        if action == 1:  # 买入
            trade_executed = self._execute_buy(current_price)
        elif action == 2:  # 卖出
            trade_executed = self._execute_sell(current_price)
        else:  # 持有
            reward = self._calculate_hold_reward(current_price)

        # 计算奖励
        if not trade_executed and action == 0:
            pass  # 已经在上面计算了

        # 检查是否终止
        done = self.current_step >= self.max_steps - 1

        # 更新步数
        self.current_step += 1

        # 获取新的观察
        observation = self._get_observation()

        # 构建信息字典
        info = {
            "total_value": self._get_total_value(current_price),
            "position_size": self.position_size,
            "cash": self.cash,
            "current_step": self.current_step,
        }

        return observation, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """获取观察（状态）"""
        current_idx = self.current_step + self.window_size

        # 确保索引有效
        if current_idx >= len(self.data):
            current_idx = len(self.data) - 1

        # 获取窗口数据
        window_data = self.data.iloc[current_idx - self.window_size + 1 : current_idx + 1]

        # 1. 价格特征 (OHLCV)
        current_bar = window_data.iloc[-1]
        price_features = [
            current_bar["open"] / current_bar["close"] - 1,
            current_bar["high"] / current_bar["close"] - 1,
            current_bar["low"] / current_bar["close"] - 1,
            current_bar["volume"] / window_data["volume"].mean(),
            window_data["close"].pct_change().iloc[-1],  # 收益率
        ]

        # 2. 技术指标
        close_prices = window_data["close"].values

        ma5 = self.indicators.sma(pd.Series(close_prices), 5).iloc[-1]
        ma20 = (
            self.indicators.sma(pd.Series(close_prices), 20).iloc[-1]
            if len(window_data) >= 20
            else ma5
        )
        rsi = self.indicators.rsi(pd.Series(close_prices)).iloc[-1]

        macd_line, signal_line, histogram = self.indicators.macd(pd.Series(close_prices))

        upper, middle, lower = self.indicators.bollinger_bands(
            pd.Series(close_prices),
            period=20 if len(window_data) >= 20 else len(window_data),
        )

        current_price = close_prices[-1]

        indicator_features = [
            (current_price / ma5 - 1) if ma5 > 0 else 0,
            (current_price / ma20 - 1) if ma20 > 0 else 0,
            (rsi - 50) / 50,  # 归一化到[-1, 1]
            macd_line.iloc[-1] / current_price if not pd.isna(macd_line.iloc[-1]) else 0,
            signal_line.iloc[-1] / current_price if not pd.isna(signal_line.iloc[-1]) else 0,
            histogram.iloc[-1] / current_price if not pd.isna(histogram.iloc[-1]) else 0,
            (current_price - upper.iloc[-1]) / current_price if not pd.isna(upper.iloc[-1]) else 0,
            (
                (current_price - middle.iloc[-1]) / current_price
                if not pd.isna(middle.iloc[-1])
                else 0
            ),
            (current_price - lower.iloc[-1]) / current_price if not pd.isna(lower.iloc[-1]) else 0,
            # 趋势
            1 if ma5 > ma20 else 0,
            (close_prices[-1] / close_prices[-5] - 1) if len(close_prices) >= 5 else 0,
            (close_prices[-1] / close_prices[-10] - 1) if len(close_prices) >= 10 else 0,
            (close_prices[-1] / close_prices[-20] - 1) if len(close_prices) >= 20 else 0,
        ]

        # 3. 账户状态
        total_value = self._get_total_value(current_price)
        position_ratio = (
            (self.position_size * current_price) / total_value if total_value > 0 else 0
        )
        cash_ratio = self.cash / total_value if total_value > 0 else 0

        account_features = [position_ratio, cash_ratio]

        # 合并所有特征
        observation = np.array(
            price_features + indicator_features + account_features, dtype=np.float32
        )

        return observation

    def _execute_buy(self, price: float) -> bool:
        """执行买入"""
        # 计算可买入数量
        max_value = self.cash * 0.95  # 保留5%现金
        max_quantity = int(max_value / price / 100) * 100

        if max_quantity < 100:
            return False

        # 计算手续费
        commission = max(max_quantity * price * self.commission_rate, 5)

        # 执行买入
        self.cash -= max_quantity * price + commission
        self.position_size += max_quantity

        self.trade_history.append(
            {
                "step": self.current_step,
                "action": "buy",
                "price": price,
                "quantity": max_quantity,
                "commission": commission,
            }
        )

        return True

    def _execute_sell(self, price: float) -> bool:
        """执行卖出"""
        if self.position_size < 100:
            return False

        # T+1检查（简化版：假设可以卖出）
        # 实际应用中需要记录买入时间

        # 计算手续费和印花税
        commission = max(self.position_size * price * self.commission_rate, 5)
        stamp_duty = self.position_size * price * 0.001

        # 执行卖出
        self.cash += self.position_size * price - commission - stamp_duty

        # 计算收益
        if self.position:
            profit = (price - self.position.avg_price) * self.position_size
        else:
            profit = 0

        self.position_size = 0
        self.position = None

        self.trade_history.append(
            {
                "step": self.current_step,
                "action": "sell",
                "price": price,
                "quantity": self.position_size,
                "commission": commission,
                "profit": profit,
            }
        )

        return True

    def _calculate_hold_reward(self, current_price: float) -> float:
        """计算持有时的奖励"""
        # 基于价格变化计算奖励
        if self.current_step > 0:
            prev_idx = self.current_step + self.window_size - 1
            prev_price = self.data.iloc[prev_idx]["close"]
            price_change = (current_price - prev_price) / prev_price

            # 如果持仓，价格变化直接影响奖励
            if self.position_size > 0:
                return (
                    price_change
                    * self.position_size
                    * current_price
                    / self._get_total_value(current_price)
                )
            else:
                return 0.0
        return 0.0

    def _get_total_value(self, price: float) -> float:
        """计算总资产"""
        return self.cash + self.position_size * price

    def render(self, mode="human"):
        """渲染环境"""
        if mode == "human":
            current_idx = self.current_step + self.window_size
            current_price = self.data.iloc[current_idx]["close"]
            total_value = self._get_total_value(current_price)

            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Cash: {self.cash:.2f}")
            print(f"Position: {self.position_size}")
            print(f"Total Value: {total_value:.2f}")
            print("-" * 40)


__all__ = ["ASharesTradingEnv"]
