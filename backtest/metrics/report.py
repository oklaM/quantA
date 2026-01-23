"""
回测报告生成器
生成HTML格式的回测报告
"""

import base64
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.metrics.performance import PerformanceMetrics, create_performance_report
from utils.logging import get_logger

logger = get_logger(__name__)


class BacktestReport:
    """回测报告生成器"""

    def __init__(self):
        self.title = "量化交易回测报告"
        self.css_style = self._get_css_style()

    def _get_css_style(self) -> str:
        """获取CSS样式"""
        return """
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                background-color: #f5f5f5;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 28px;
            }
            h2 {
                color: #2c3e50;
                border-left: 4px solid #3498db;
                padding-left: 10px;
                margin-top: 30px;
                margin-bottom: 15px;
                font-size: 20px;
            }
            .summary {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }
            .metric-card.positive {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }
            .metric-card.negative {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }
            .metric-label {
                font-size: 14px;
                opacity: 0.9;
                margin-bottom: 5px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f8f9fa;
                font-weight: 600;
                color: #495057;
            }
            tr:hover {
                background-color: #f8f9fa;
            }
            .positive {
                color: #e74c3c;
            }
            .negative {
                color: #27ae60;
            }
            .info {
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
                border-left: 4px solid #3498db;
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                color: #7f8c8d;
                font-size: 12px;
            }
        </style>
        """

    def generate(
        self,
        results: Dict[str, Any],
        strategy_name: str = "策略",
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成HTML报告

        Args:
            results: 回测结果
            strategy_name: 策略名称
            output_path: 输出文件路径

        Returns:
            HTML报告字符串
        """
        html_parts = []

        # HTML头部
        html_parts.append(
            """
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>回测报告</title>
        """
        )

        # CSS样式
        html_parts.append(self.css_style)

        html_parts.append("</head><body>")

        # 容器
        html_parts.append('<div class="container">')

        # 标题
        html_parts.append(f"<h1>{self.title}</h1>")

        # 策略信息
        html_parts.append(f'<div class="info">')
        html_parts.append(f"<strong>策略名称:</strong> {strategy_name}<br>")
        html_parts.append(
            f'<strong>生成时间:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )
        html_parts.append("</div>")

        # 账户概览
        html_parts.extend(self._generate_account_summary(results.get("account", {})))

        # 绩效指标
        html_parts.extend(self._generate_performance_metrics(results.get("performance", {})))

        # 持仓详情
        html_parts.extend(self._generate_positions_table(results.get("positions", [])))

        # 净值曲线
        html_parts.extend(self._generate_equity_curve(results.get("equity_curve", pd.DataFrame())))

        # 页脚
        html_parts.append('<div class="footer">')
        html_parts.append("本报告由quantA量化交易系统生成<br>")
        html_parts.append("投资有风险，入市需谨慎")
        html_parts.append("</div>")

        html_parts.append("</div>")  # container
        html_parts.append("</body></html>")

        html_content = "\n".join(html_parts)

        # 保存到文件
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"报告已保存到: {output_path}")

        return html_content

    def _generate_account_summary(self, account: Dict[str, Any]) -> List[str]:
        """生成账户概览"""
        html = ["<h2>账户概览</h2>", '<div class="summary">']

        cards = [
            ("总资产", f"¥{account.get('total_value', 0):,.2f}", ""),
            ("初始资金", f"¥{account.get('initial_cash', 0):,.2f}", ""),
            ("可用资金", f"¥{account.get('cash', 0):,.2f}", ""),
            ("持仓市值", f"¥{account.get('position_value', 0):,.2f}", ""),
        ]

        total_return = account.get("total_return_pct", 0)
        return_class = "positive" if total_return > 0 else "negative"
        cards.append(("总收益率", f"{total_return:.2f}%", return_class))

        for label, value, card_class in cards:
            if card_class:
                class_attr = f'class="metric-card {card_class}"'
            else:
                class_attr = 'class="metric-card"'
            html.append(f"<div {class_attr}>")
            html.append(f'<div class="metric-label">{label}</div>')
            html.append(f'<div class="metric-value">{value}</div>')
            html.append("</div>")

        html.append("</div>")
        return html

    def _generate_performance_metrics(self, performance: Dict[str, Any]) -> List[str]:
        """生成绩效指标"""
        html = ["<h2>绩效指标</h2>", "<table>"]
        html.append("<tr><th>指标</th><th>数值</th></tr>")

        metrics = [
            ("夏普比率", performance.get("sharpe_ratio", 0)),
            ("最大回撤", f"{performance.get('max_drawdown', 0) * 100:.2f}%"),
            ("年化波动率", f"{performance.get('volatility', 0) * 100:.2f}%"),
        ]

        for name, value in metrics:
            html.append(f"<tr><td>{name}</td><td>{value}</td></tr>")

        html.append("</table>")
        return html

    def _generate_positions_table(self, positions: List[Dict[str, Any]]) -> List[str]:
        """生成持仓表格"""
        html = ["<h2>当前持仓</h2>"]

        if not positions:
            html.append("<p>无持仓</p>")
            return html

        html.append("<table>")
        html.append(
            "<tr><th>股票代码</th><th>持仓数量</th><th>成本价</th><th>现价</th><th>市值</th><th>盈亏</th><th>盈亏比例</th></tr>"
        )

        for pos in positions:
            pnl = pos.get("pnl", 0)
            pnl_class = "positive" if pnl > 0 else "negative"

            html.append(f"<tr>")
            html.append(f'<td>{pos.get("symbol", "")}</td>')
            html.append(f'<td>{pos.get("quantity", 0)}</td>')
            html.append(f'<td>¥{pos.get("avg_price", 0):.2f}</td>')
            html.append(f'<td>¥{pos.get("current_price", 0):.2f}</td>')
            html.append(f'<td>¥{pos.get("market_value", 0):,.2f}</td>')
            html.append(f'<td class="{pnl_class}">¥{pnl:,.2f}</td>')
            html.append(f'<td class="{pnl_class}">{pos.get("pnl_pct", 0):.2f}%</td>')
            html.append(f"</tr>")

        html.append("</table>")
        return html

    def _generate_equity_curve(self, equity_curve: pd.DataFrame) -> List[str]:
        """生成净值曲线描述"""
        html = ["<h2>净值曲线</h2>"]

        if equity_curve.empty:
            html.append("<p>无净值曲线数据</p>")
        else:
            initial_value = equity_curve["total_value"].iloc[0]
            final_value = equity_curve["total_value"].iloc[-1]
            total_return = (final_value - initial_value) / initial_value * 100

            html.append("<table>")
            html.append("<tr><th>指标</th><th>数值</th></tr>")
            html.append(f"<tr><td>初始净值</td><td>¥{initial_value:,.2f}</td></tr>")
            html.append(f"<tr><td>最终净值</td><td>¥{final_value:,.2f}</td></tr>")
            html.append(f"<tr><td>收益率</td><td>{total_return:.2f}%</td></tr>")
            html.append("</table>")

        return html


def generate_report(
    results: Dict[str, Any],
    strategy_name: str = "策略",
    output_path: Optional[str] = None,
) -> str:
    """
    生成回测报告（便捷函数）

    Args:
        results: 回测结果
        strategy_name: 策略名称
        output_path: 输出文件路径

    Returns:
        HTML报告字符串
    """
    generator = BacktestReport()
    return generator.generate(results, strategy_name, output_path)


__all__ = [
    "BacktestReport",
    "generate_report",
]
