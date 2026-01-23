"""
实时监控Web界面

使用Streamlit实现实时交易监控仪表板
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any

# 配置页面
st.set_page_config(
    page_title="quantA 监控面板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .status-running {
        color: #00cc00;
        font-weight: bold;
    }
    .status-stopped {
        color: #cc0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """渲染页面标题"""
    st.markdown('<h1 class="main-header">📊 quantA 实时交易监控面板</h1>', unsafe_allow_html=True)
    st.markdown("---")


def render_system_status(status: Dict[str, Any]):
    """渲染系统状态"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_class = "status-running" if status.get("status") == "running" else "status-stopped"
        st.markdown(f"""
        <div class="metric-card">
            <h3>系统状态</h3>
            <p class="{status_class}">{status.get("status", "unknown").upper()}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        uptime = status.get("uptime_seconds", 0) / 3600
        st.metric("运行时间", f"{uptime:.1f} 小时")

    with col3:
        cpu = status.get("cpu_percent", 0)
        st.metric("CPU使用率", f"{cpu:.1f}%")

    with col4:
        memory = status.get("memory_percent", 0)
        st.metric("内存使用率", f"{memory:.1f}%")


def render_trading_status(status: Dict[str, Any]):
    """渲染交易状态"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("总订单数", status.get("total_orders", 0))

    with col2:
        filled = status.get("filled_orders", 0)
        total = status.get("total_orders", 1)
        fill_rate = status.get("fill_rate", 0) * 100
        st.metric("成交订单", f"{filled} ({fill_rate:.1f}%)")

    with col3:
        rejected = status.get("rejected_orders", 0)
        st.metric("拒单数", rejected)


def render_performance_status(status: Dict[str, Any]):
    """渲染绩效状态"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_value = status.get("total_value", 0)
        st.metric("总资产", f"¥{total_value:,.2f}")

    with col2:
        daily_pnl = status.get("daily_pnl", 0)
        daily_pnl_ratio = status.get("daily_pnl_ratio", 0)
        delta_color = "normal" if daily_pnl >= 0 else "inverse"
        st.metric("今日盈亏", f"¥{daily_pnl:,.2f} ({daily_pnl_ratio:.2%})", delta_color=delta_color)

    with col3:
        total_pnl_ratio = status.get("total_pnl_ratio", 0)
        delta_color = "normal" if total_pnl_ratio >= 0 else "inverse"
        st.metric("总收益率", f"{total_pnl_ratio:.2%}", delta_color=delta_color)

    with col4:
        sharpe = status.get("sharpe_ratio", 0)
        st.metric("夏普比率", f"{sharpe:.2f}")


def render_pnl_chart(pnl_history: pd.DataFrame):
    """渲染盈亏曲线"""
    if pnl_history.empty:
        st.info("暂无盈亏数据")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pnl_history['timestamp'],
        y=pnl_history['total_value'],
        mode='lines',
        name='总资产',
        line=dict(color='#1f77b4', width=2),
    ))

    fig.add_trace(go.Scatter(
        x=pnl_history['timestamp'],
        y=pnl_history['initial_value'],
        mode='lines',
        name='初始资金',
        line=dict(color='gray', width=1, dash='dash'),
    ))

    fig.update_layout(
        title="资产曲线",
        xaxis_title="时间",
        yaxis_title="资产 (¥)",
        hovermode='x unified',
        height=300,
        margin=dict(l=0, r=0, t=30, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(drawdown_history: pd.DataFrame):
    """渲染回撤图"""
    if drawdown_history.empty:
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown_history['timestamp'],
        y=drawdown_history['drawdown'] * 100,
        fill='tozeroy',
        mode='lines',
        name='回撤',
        line=dict(color='red'),
    ))

    fig.update_layout(
        title="回撤曲线",
        xaxis_title="时间",
        yaxis_title="回撤 (%)",
        height=200,
        margin=dict(l=0, r=0, t=30, b=30),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_position_table(positions: pd.DataFrame):
    """渲染持仓表格"""
    if positions.empty:
        st.info("暂无持仓")
        return

    # 格式化数据
    display_df = positions.copy()
    display_df['market_value'] = display_df['market_value'].apply(lambda x: f"¥{x:,.2f}")
    display_df['unrealized_pnl'] = display_df['unrealized_pnl'].apply(lambda x: f"¥{x:,.2f}")
    display_df['pnl_ratio'] = display_df['pnl_ratio'].apply(lambda x: f"{x:.2%}")

    st.dataframe(
        display_df,
        column_config={
            "symbol": "股票代码",
            "quantity": "数量",
            "avg_cost": "成本价",
            "current_price": "现价",
            "market_value": "市值",
            "unrealized_pnl": "浮动盈亏",
            "pnl_ratio": "盈亏比例",
        },
        hide_index=True,
        use_container_width=True,
    )


def render_alerts(alerts: list):
    """渲染告警列表"""
    if not alerts:
        st.success("🎉 无活跃告警")
        return

    for alert in alerts[:10]:  # 显示最近10条
        severity = alert.get("severity", "info")
        icon = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }.get(severity, "📌")

        st.markdown(f"""
        **{icon} [{alert.get('severity', 'info').upper()}] {alert.get('title', '未知告警')}**

        {alert.get('message', '')}

        *时间: {alert.get('timestamp', '')}*

        ---
        """)


def render_sidebar():
    """渲染侧边栏"""
    st.sidebar.header("⚙️ 控制面板")

    # 系统控制
    st.sidebar.subheader("系统控制")
    if st.sidebar.button("▶️ 启动系统", key="start"):
        st.session_state.running = True
        st.sidebar.success("系统已启动")

    if st.sidebar.button("⏸️ 停止系统", key="stop"):
        st.session_state.running = False
        st.sidebar.warning("系统已停止")

    # 刷新间隔
    st.sidebar.subheader("设置")
    refresh_interval = st.sidebar.slider(
        "刷新间隔 (秒)",
        min_value=1,
        max_value=60,
        value=5,
        key="refresh_interval"
    )

    # 数据过滤
    st.sidebar.subheader("数据过滤")
    time_range = st.sidebar.selectbox(
        "时间范围",
        ["今日", "近3天", "近7天", "近30天"],
        index=0,
        key="time_range"
    )

    # 显示统计
    st.sidebar.subheader("统计信息")
    st.sidebar.metric("数据点", "1,234")
    st.sidebar.metric("告警数", "5")


def main():
    """主函数"""
    # 初始化session state
    if "running" not in st.session_state:
        st.session_state.running = False

    # 渲染侧边栏
    render_sidebar()

    # 渲染标题
    render_header()

    # 模拟数据（实际应用中从监控系统获取）
    system_status = {
        "status": "running" if st.session_state.running else "stopped",
        "uptime_seconds": 3600 * 2.5,
        "cpu_percent": 35.2,
        "memory_percent": 62.8,
    }

    trading_status = {
        "is_trading": st.session_state.running,
        "total_orders": 125,
        "filled_orders": 98,
        "rejected_orders": 3,
        "pending_orders": 5,
        "fill_rate": 0.784,
    }

    performance_status = {
        "total_value": 1052340.50,
        "initial_capital": 1000000.0,
        "daily_pnl": 52340.50,
        "daily_pnl_ratio": 0.05234,
        "total_pnl": 52340.50,
        "total_pnl_ratio": 0.05234,
        "max_drawdown": -0.0234,
        "sharpe_ratio": 1.85,
        "win_rate": 0.65,
    }

    alerts = [
        {
            "type": "warning",
            "severity": "warning",
            "title": "持仓集中度告警",
            "message": "单一持仓比例过高: 000001.SZ 占比32%",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        {
            "type": "info",
            "severity": "info",
            "title": "数据更新",
            "message": "市场数据已更新",
            "timestamp": (datetime.now() - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
        },
    ]

    # 模拟持仓数据
    positions_data = pd.DataFrame([
        {"symbol": "000001.SZ", "quantity": 1000, "avg_cost": 10.5, "current_price": 11.2,
         "market_value": 11200, "unrealized_pnl": 700, "pnl_ratio": 0.0667},
        {"symbol": "000002.SZ", "quantity": 500, "avg_cost": 25.3, "current_price": 24.8,
         "market_value": 12400, "unrealized_pnl": -250, "pnl_ratio": -0.0198},
    ])

    # 模拟盈亏历史数据
    pnl_history = pd.DataFrame([
        {"timestamp": datetime.now() - timedelta(hours=i), "total_value": 1000000 + i * 1000,
         "initial_value": 1000000}
        for i in range(24, 0, -1)
    ])

    # 渲染各个部分
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 系统与交易状态")
        render_system_status(system_status)
        st.markdown("---")
        render_trading_status(trading_status)

    with col2:
        st.subheader("💰 绩效概览")
        render_performance_status(performance_status)

    st.markdown("---")

    # 图表区域
    col1, col2 = st.columns(2)

    with col1:
        render_pnl_chart(pnl_history)

    with col2:
        st.subheader("📉 持仓明细")
        render_position_table(positions_data)

    st.markdown("---")

    # 告警区域
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🔔 告警中心")
        render_alerts(alerts)

    with col2:
        st.subheader("📝 系统日志")
        st.text("""
[10:30:25] 系统启动
[10:30:26] 数据连接成功
[10:30:27] 策略加载完成
[10:30:28] 开始交易
[10:35:12] 买入订单: 000001.SZ 1000股
[10:35:15] 订单成交
        """)

    # 自动刷新
    if st.session_state.running:
        st.rerun()


if __name__ == "__main__":
    main()
