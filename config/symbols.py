"""
股票池配置
定义可交易股票范围和筛选条件
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set


class MarketType(Enum):
    """市场类型"""

    MAIN = "main"  # 主板
    SME = "sme"  # 中小板
    ChiNext = "chinext"  # 创业板
    STAR = "star"  # 科创板
    BEIJING = "beijing"  # 北交所


class IndexType(Enum):
    """指数类型"""

    SH000001 = "sh.000001"  # 上证指数
    SZ399001 = "sz.399001"  # 深证成指
    SZ399006 = "sz.399006"  # 创业板指
    SH000688 = "sh.000688"  # 科创50
    SH000300 = "sh.000300"  # 沪深300
    SH000905 = "sh.000905"  # 中证500
    SH000852 = "sh.000852"  # 中证1000


@dataclass
class SymbolFilter:
    """股票筛选条件"""

    # 市值范围（亿元）
    MIN_MARKET_CAP = 50  # 最小市值
    MAX_MARKET_CAP = 2000  # 最大市值

    # 流动性过滤
    MIN_AVG_VOLUME = 10000000  # 最小日均成交额（万元）
    MIN_TURNOVER = 0.01  # 最小换手率

    # 技术指标过滤
    MIN_PRICE = 5.0  # 最低股价
    MAX_PRICE = 100.0  # 最高股价

    # ST股票过滤
    EXCLUDE_ST = True  # 排除ST股票
    EXCLUDE_SUSPEND = True  # 排除停牌股票

    # 新股过滤
    MIN_LISTING_DAYS = 180  # 最小上市天数（6个月）

    # 行业过滤
    EXCLUDE_INDUSTRIES: Set[str] = field(
        default_factory=lambda: {
            "银行",  # 排除银行股（受监管影响大）
        }
    )

    # 可交易时间
    TRADABLE_START_DATE = "2020-01-01"


# 预设股票池
PRESET_UNIVERSES: Dict[str, List[str]] = {
    # 核心宽基指数成分股
    "hs300": [],  # 沪深300
    "csi500": [],  # 中证500
    "csi1000": [],  # 中证1000
    "cyb50": [],  # 创业板50
    "st50": [],  # 科创50
    # 行业主题
    "technology": [],  # 科技
    "consumer": [],  # 消费
    "healthcare": [],  # 医疗
    "finance": [],  # 金融
    "energy": [],  # 能源
    # 自定义股票池（需要手动维护）
    "custom": [
        # 示例
        # "600519.SH",  # 贵州茅台
        # "300750.SZ",  # 宁德时代
        # "601318.SH",  # 中国平安
    ],
}


# 行业分类（申万一级）
SW_INDUSTRY_LEVEL_1 = [
    "农林牧渔",
    "采掘",
    "化工",
    "钢铁",
    "有色金属",
    "电子",
    "汽车",
    "家用电器",
    "食品饮料",
    "纺织服装",
    "轻工制造",
    "医药生物",
    "公用事业",
    "交通运输",
    "房地产",
    "商业贸易",
    "休闲服务",
    "综合",
    "建筑材料",
    "建筑装饰",
    "电气设备",
    "国防军工",
    "计算机",
    "传媒",
    "通信",
    "银行",
    "非银金融",
    "机械设备",
]


# 概念板块（示例）
CONCEPT_SECTORS = {
    "ai": "人工智能",
    "new_energy": "新能源",
    "chip": "半导体芯片",
    "biomedicine": "生物医药",
    "military": "军工",
    "consumer_upgrade": "消费升级",
}


# 板块映射
STOCK_BOARD_MAP = {
    # 主板
    "600": MarketType.MAIN,
    "601": MarketType.MAIN,
    "603": MarketType.MAIN,
    "605": MarketType.MAIN,
    # 科创板
    "688": MarketType.STAR,
    # 深证主板
    "000": MarketType.MAIN,
    "001": MarketType.MAIN,
    # 中小板
    "002": MarketType.SME,
    # 创业板
    "300": MarketType.ChiNext,
    # 北交所
    "8": MarketType.BEIJING,
    "4": MarketType.BEIJING,
}


def get_market_type(symbol: str) -> MarketType:
    """根据股票代码判断市场类型"""
    # 移除后缀（如 .SH, .SZ）
    code = symbol.split(".")[0]

    for prefix, market in STOCK_BOARD_MAP.items():
        if code.startswith(prefix):
            return market

    return MarketType.MAIN


def get_limit_percent(symbol: str) -> float:
    """获取股票涨跌停限制比例"""
    market = get_market_type(symbol)

    if market in [MarketType.ChiNext, MarketType.STAR]:
        return 0.20  # 20%
    else:
        return 0.10  # 10%


def is_valid_symbol(symbol: str) -> bool:
    """验证股票代码是否有效"""
    code = symbol.split(".")[0]

    # A股代码长度通常为6位
    if len(code) != 6:
        return False

    # 检查是否为数字
    if not code.isdigit():
        return False

    # 检查是否在已知前缀中
    return get_market_type(symbol) in [
        MarketType.MAIN,
        MarketType.SME,
        MarketType.ChiNext,
        MarketType.STAR,
    ]


# 默认筛选器
default_filter = SymbolFilter()


# 导出配置
__all__ = [
    "MarketType",
    "IndexType",
    "SymbolFilter",
    "PRESET_UNIVERSES",
    "SW_INDUSTRY_LEVEL_1",
    "CONCEPT_SECTORS",
    "get_market_type",
    "get_limit_percent",
    "is_valid_symbol",
    "default_filter",
]
