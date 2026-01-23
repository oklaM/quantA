"""
模型评估和对比框架
提供完整的模型评估、对比和分析功能
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from stable_baselines3 import BaseAlgorithm

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseAlgorithm = None

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """评估结果"""

    model_name: str
    algorithm: str
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    median_reward: float
    mean_length: float
    n_episodes: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": self.min_reward,
            "max_reward": self.max_reward,
            "median_reward": self.median_reward,
            "mean_length": self.mean_length,
            "n_episodes": self.n_episodes,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
        }


class ModelEvaluator:
    """
    模型评估器

    评估单个模型的性能
    """

    def __init__(self, env, n_episodes: int = 100):
        """
        Args:
            env: 评估环境
            n_episodes: 评估回合数
        """
        self.env = env
        self.n_episodes = n_episodes

    def evaluate(
        self, model: BaseAlgorithm, model_name: str, deterministic: bool = True, **kwargs
    ) -> EvaluationResult:
        """
        评估模型

        Args:
            model: SB3模型
            model_name: 模型名称
            deterministic: 是否使用确定性策略
            **kwargs: 其他参数

        Returns:
            EvaluationResult
        """
        episode_rewards = []
        episode_lengths = []
        episode_details = []

        for episode in range(self.n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            episode_actions = []

            while True:
                action, _states = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_length += 1
                episode_actions.append(int(action))

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_details.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "actions": episode_actions,
                }
            )

        # 计算统计指标
        rewards_array = np.array(episode_rewards)
        lengths_array = np.array(episode_lengths)

        result = EvaluationResult(
            model_name=model_name,
            algorithm=model.__class__.__name__,
            mean_reward=np.mean(rewards_array),
            std_reward=np.std(rewards_array),
            min_reward=np.min(rewards_array),
            max_reward=np.max(rewards_array),
            median_reward=np.median(rewards_array),
            mean_length=np.mean(lengths_array),
            n_episodes=self.n_episodes,
            metrics=self._calculate_metrics(rewards_array, lengths_array),
            details=episode_details,
        )

        logger.info(f"评估完成: {model_name}")
        logger.info(f"  平均奖励: {result.mean_reward:.2f} +/- {result.std_reward:.2f}")

        return result

    def _calculate_metrics(self, rewards: np.ndarray, lengths: np.ndarray) -> Dict[str, float]:
        """计算额外指标"""
        metrics = {}

        # 夏普比率
        if len(rewards) > 1:
            metrics["sharpe_ratio"] = np.mean(rewards) / (np.std(rewards) + 1e-8)

        # 胜率（盈利回合占比）
        metrics["win_rate"] = np.sum(rewards > 0) / len(rewards)

        # 风险指标
        metrics["reward_std"] = np.std(rewards)
        metrics["reward_range"] = np.max(rewards) - np.min(rewards)

        # 稳定性（变异系数）
        if np.mean(rewards) > 0:
            metrics["coefficient_of_variation"] = np.std(rewards) / np.abs(np.mean(rewards))
        else:
            metrics["coefficient_of_variation"] = 0.0

        return metrics


class ModelComparator:
    """
    模型对比器

    对比多个模型的性能
    """

    def __init__(self, env, n_episodes: int = 100):
        """
        Args:
            env: 评估环境
            n_episodes: 评估回合数
        """
        self.env = env
        self.n_episodes = n_episodes
        self.evaluator = ModelEvaluator(env, n_episodes)

        self.results: List[EvaluationResult] = []

    def compare(
        self,
        models: Dict[str, BaseAlgorithm],
        deterministic: bool = True,
    ) -> pd.DataFrame:
        """
        对比多个模型

        Args:
            models: {模型名称: 模型实例} 字典
            deterministic: 是否使用确定性策略

        Returns:
            对比结果DataFrame
        """
        self.results = []

        for model_name, model in models.items():
            logger.info(f"评估模型: {model_name}")
            result = self.evaluator.evaluate(
                model=model,
                model_name=model_name,
                deterministic=deterministic,
            )
            self.results.append(result)

        # 创建对比表
        comparison_df = pd.DataFrame([r.to_dict() for r in self.results])

        logger.info(f"\n模型对比结果:")
        logger.info(f"\n{comparison_df.to_string()}")

        return comparison_df

    def get_best_model(self, metric: str = "mean_reward") -> EvaluationResult:
        """
        获取最佳模型

        Args:
            metric: 评估指标

        Returns:
            最佳模型的评估结果
        """
        if not self.results:
            raise ValueError("没有评估结果，请先调用compare()方法")

        sorted_results = sorted(self.results, key=lambda r: getattr(r, metric), reverse=True)

        best = sorted_results[0]
        logger.info(f"最佳模型: {best.model_name} ({metric}={getattr(best, metric):.2f})")

        return best

    def plot_comparison(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None,
    ):
        """
        绘制对比图表

        Args:
            metrics: 要绘制的指标列表
            save_path: 保存路径
        """
        if not self.results:
            raise ValueError("没有评估结果，请先调用compare()方法")

        if metrics is None:
            metrics = ["mean_reward", "std_reward", "win_rate", "sharpe_ratio"]

        # 准备数据
        model_names = [r.model_name for r in self.results]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("模型性能对比", fontsize=16)

        # 1. 平均奖励对比
        ax = axes[0, 0]
        mean_rewards = [r.mean_reward for r in self.results]
        std_rewards = [r.std_reward for r in self.results]
        ax.bar(model_names, mean_rewards, yerr=std_rewards, alpha=0.7, capsize=5)
        ax.set_ylabel("Mean Reward")
        ax.set_title("Average Reward Comparison")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # 2. 奖励分布（箱线图）
        ax = axes[0, 1]
        all_rewards = []
        for r in self.results:
            all_rewards.append([d["reward"] for d in r.details])
        ax.boxplot(all_rewards, labels=model_names)
        ax.set_ylabel("Reward")
        ax.set_title("Reward Distribution")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # 3. 风险调整指标
        ax = axes[1, 0]
        sharpe_ratios = [r.metrics.get("sharpe_ratio", 0) for r in self.results]
        win_rates = [r.metrics.get("win_rate", 0) for r in self.results]

        x = np.arange(len(model_names))
        width = 0.35

        ax.bar(x - width / 2, sharpe_ratios, width, label="Sharpe Ratio", alpha=0.7)
        ax.bar(x + width / 2, win_rates, width, label="Win Rate", alpha=0.7)
        ax.set_ylabel("Value")
        ax.set_title("Risk-Adjusted Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # 4. 综合评分雷达图
        ax = axes[1, 1]
        categories = ["Mean Reward", "Stability", "Win Rate", "Sharpe Ratio"]

        # 归一化指标
        normalized_scores = []
        for r in self.results:
            scores = [
                (
                    r.mean_reward / max([rr.mean_reward for rr in self.results])
                    if r.mean_reward > 0
                    else 0
                ),
                1 / (1 + r.metrics.get("coefficient_of_variation", 1)),
                r.metrics.get("win_rate", 0),
                min(r.metrics.get("sharpe_ratio", 0) / 2, 1),  # 限制最大值
            ]
            normalized_scores.append(scores)

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        normalized_scores += normalized_scores[:1]  # 闭合
        angles += angles[:1]

        ax = plt.subplot(2, 2, 4, projection="polar")
        for i, (name, scores) in enumerate(zip(model_names, normalized_scores)):
            ax.plot(angles, scores, "o-", linewidth=2, label=name)
            ax.fill(angles, scores, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title("Comprehensive Scores")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"对比图表已保存到: {save_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, path: str):
        """
        保存对比结果

        Args:
            path: 保存路径
        """
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "n_episodes": self.n_episodes,
            "results": [r.to_dict() for r in self.results],
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"对比结果已保存到: {path}")


class CrossValidator:
    """
    交叉验证器

    进行K折交叉验证
    """

    def __init__(
        self,
        env_builder: callable,
        n_folds: int = 5,
        n_episodes: int = 50,
    ):
        """
        Args:
            env_builder: 环境构建函数
            n_folds: 折数
            n_episodes: 每折的回合数
        """
        self.env_builder = env_builder
        self.n_folds = n_folds
        self.n_episodes = n_episodes

    def cross_validate(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        model_name: str,
    ) -> Dict[str, Any]:
        """
        执行交叉验证

        Args:
            model_class: 模型类
            model_params: 模型参数
            model_name: 模型名称

        Returns:
            交叉验证结果
        """
        fold_results = []

        for fold in range(self.n_folds):
            logger.info(f"交叉验证 - Fold {fold + 1}/{self.n_folds}")

            # 创建环境
            env = self.env_builder()

            # 创建模型
            model = model_class("MlpPolicy", env, **model_params)

            # 短暂训练
            model.learn(total_timesteps=5000, verbose=0)

            # 评估
            evaluator = ModelEvaluator(env, self.n_episodes)
            result = evaluator.evaluate(model, f"{model_name}_fold{fold}")

            fold_results.append(result.to_dict())

        # 汇总结果
        mean_rewards = [r["mean_reward"] for r in fold_results]
        std_rewards = [r["std_reward"] for r in fold_results]

        cv_results = {
            "model_name": model_name,
            "mean_reward": np.mean(mean_rewards),
            "std_reward": np.std(mean_rewards),
            "fold_results": fold_results,
            "n_folds": self.n_folds,
        }

        logger.info(f"交叉验证完成: {model_name}")
        logger.info(
            f"  平均奖励: {cv_results['mean_reward']:.2f} +/- {cv_results['std_reward']:.2f}"
        )

        return cv_results


def create_comparison_report(
    results: List[EvaluationResult],
    output_path: str = "comparison_report.html",
) -> str:
    """
    创建HTML对比报告

    Args:
        results: 评估结果列表
        output_path: 输出路径

    Returns:
        HTML报告内容
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>模型对比报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #ddd; }
            .metric-card { display: inline-block; margin: 10px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>模型性能对比报告</h1>
        <p>生成时间: {timestamp}</p>

        <h2>对比表</h2>
        <table>
            <tr>
                <th>模型名称</th>
                <th>算法</th>
                <th>平均奖励</th>
                <th>标准差</th>
                <th>最小值</th>
                <th>最大值</th>
                <th>中位数</th>
                <th>回合数</th>
            </tr>
            {rows}
        </table>

        <h2>性能指标</h2>
        <div class="metrics">
            {metric_cards}
        </div>

        <h2>详细分析</h2>
        <p>{analysis}</p>
    </body>
    </html>
    """

    # 生成表格行
    rows = ""
    for r in results:
        rows += f"""
            <tr>
                <td>{r.model_name}</td>
                <td>{r.algorithm}</td>
                <td>{r.mean_reward:.2f}</td>
                <td>{r.std_reward:.2f}</td>
                <td>{r.min_reward:.2f}</td>
                <td>{r.max_reward:.2f}</td>
                <td>{r.median_reward:.2f}</td>
                <td>{r.n_episodes}</td>
            </tr>
        """

    # 生成指标卡片
    metric_cards = ""
    best_model = max(results, key=lambda r: r.mean_reward)
    metric_cards += f"""
        <div class="metric-card">
            <h3>最佳模型</h3>
            <p>{best_model.model_name}</p>
            <p>平均奖励: {best_model.mean_reward:.2f}</p>
        </div>
    """

    # 简单分析
    analysis = (
        f"共对比了 {len(results)} 个模型。最佳模型是 {best_model.model_name}，"
        f"平均奖励为 {best_model.mean_reward:.2f}。"
    )

    html = html.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        rows=rows,
        metric_cards=metric_cards,
        analysis=analysis,
    )

    # 保存文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"对比报告已保存到: {output_path}")

    return html


__all__ = [
    "EvaluationResult",
    "ModelEvaluator",
    "ModelComparator",
    "CrossValidator",
    "create_comparison_report",
]
