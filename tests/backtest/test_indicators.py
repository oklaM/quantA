"""
测试技术指标计算模块
"""

import numpy as np
import pandas as pd
import pytest

from backtest.engine.indicators import (
    TechnicalIndicators,
    add_indicators,
)


@pytest.mark.unit
class TestSMA:
    """测试简单移动平均线"""

    def test_sma_basic(self):
        """测试基本SMA计算"""
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = TechnicalIndicators.sma(prices, period=3)

        assert len(result) == len(prices)
        # 验证第一个值（min_periods=1）
        assert result.iloc[0] == 1.0
        # 验证第三个值
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        # 验证倒数第二个值 (7+8+9)/3 = 8.0
        assert result.iloc[-2] == 8.0
        # 验证最后一个值
        assert result.iloc[-1] == 9.0  # (8+9+10)/3

    def test_sma_period(self):
        """测试不同周期"""
        prices = pd.Series([10, 20, 30, 40, 50])

        sma5 = TechnicalIndicators.sma(prices, period=5)
        assert sma5.iloc[-1] == 30.0  # (10+20+30+40+50)/5

        sma3 = TechnicalIndicators.sma(prices, period=3)
        assert sma3.iloc[-1] == 40.0  # (30+40+50)/3


@pytest.mark.unit
class TestEMA:
    """测试指数移动平均线"""

    def test_ema_basic(self):
        """测试基本EMA计算"""
        prices = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = TechnicalIndicators.ema(prices, period=3)

        assert len(result) == len(prices)
        # EMA的第一个值等于第一个价格
        assert result.iloc[0] == prices.iloc[0]
        # EMA应该比SMA更平滑
        sma = TechnicalIndicators.sma(prices, period=3)
        assert result.iloc[-1] != sma.iloc[-1]


@pytest.mark.unit
class TestMACD:
    """测试MACD指标"""

    def test_macd_basic(self):
        """测试基本MACD计算"""
        prices = pd.Series(range(1, 101))

        macd, signal, histogram = TechnicalIndicators.macd(prices)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)

        # MACD = EMA(fast) - EMA(slow)
        # 在上升趋势中，MACD应该为正
        assert macd.iloc[-1] > 0

        # Histogram = MACD - Signal
        assert histogram.iloc[-1] == macd.iloc[-1] - signal.iloc[-1]

    def test_macd_custom_periods(self):
        """测试自定义周期"""
        prices = pd.Series(range(1, 51))

        macd, signal, histogram = TechnicalIndicators.macd(prices, fast=5, slow=10, signal=3)

        assert len(macd) == len(prices)
        assert len(signal) == len(prices)


@pytest.mark.unit
class TestRSI:
    """测试RSI指标"""

    def test_rsi_basic(self):
        """测试基本RSI计算"""
        # 创建一个上涨序列
        prices = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

        rsi = TechnicalIndicators.rsi(prices, period=14)

        assert len(rsi) == len(prices)
        # 纯上涨的RSI应该接近100
        assert rsi.iloc[-1] > 80

    def test_rsi_range(self):
        """测试RSI范围"""
        prices = pd.Series(np.random.randn(100) + 100)

        rsi = TechnicalIndicators.rsi(prices, period=14)

        # RSI应该在0-100之间
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_rsi_falling_prices(self):
        """测试下跌价格的RSI"""
        # 创建一个下跌序列
        prices = pd.Series([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])

        rsi = TechnicalIndicators.rsi(prices, period=14)

        # 纯下跌的RSI应该接近0
        assert rsi.iloc[-1] < 20


@pytest.mark.unit
class TestKDJ:
    """测试KDJ指标"""

    def test_kdj_basic(self):
        """测试基本KDJ计算"""
        high = pd.Series([105, 110, 115, 120, 125])
        low = pd.Series([95, 100, 105, 110, 115])
        close = pd.Series([100, 105, 110, 115, 120])

        k, d, j = TechnicalIndicators.kdj(high, low, close)

        assert len(k) == len(close)
        assert len(d) == len(close)
        assert len(j) == len(close)

        # K、D应该在0-100之间
        assert k.min() >= 0 and k.max() <= 100
        assert d.min() >= 0 and d.max() <= 100

    def test_kdj_relationship(self):
        """测试KDJ之间的关系"""
        high = pd.Series(range(100, 200))
        low = pd.Series(range(0, 100))
        close = pd.Series(range(50, 150))

        k, d, j = TechnicalIndicators.kdj(high, low, close)

        # J = 3K - 2D
        assert np.allclose(j, 3 * k - 2 * d, equal_nan=True)


@pytest.mark.unit
class TestBollingerBands:
    """测试布林带"""

    def test_bollinger_bands_basic(self):
        """测试基本布林带计算"""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])

        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, period=5, std_dev=2)

        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

        # 上轨应该高于中轨
        assert upper.iloc[-1] > middle.iloc[-1]
        # 下轨应该低于中轨
        assert lower.iloc[-1] < middle.iloc[-1]

        # 上轨 = 中轨 + 2*标准差
        expected_std = prices.tail(5).std()
        assert np.isclose(upper.iloc[-1], middle.iloc[-1] + 2 * expected_std)

    def test_bollinger_bands_width(self):
        """测试布林带宽度"""
        prices = pd.Series(np.random.randn(100) + 100)

        upper, middle, lower = TechnicalIndicators.bollinger_bands(prices)

        # 计算布林带宽度
        width = upper - lower

        # 宽度应该为正（忽略NaN值，因为第一个值的std为NaN）
        assert (width.dropna() > 0).all()


@pytest.mark.unit
class TestATR:
    """测试ATR指标"""

    def test_atr_basic(self):
        """测试基本ATR计算"""
        high = pd.Series([105, 110, 115, 120, 125])
        low = pd.Series([95, 100, 105, 110, 115])
        close = pd.Series([100, 105, 110, 115, 120])

        atr = TechnicalIndicators.atr(high, low, close, period=14)

        assert len(atr) == len(close)
        # ATR应该为正
        assert (atr >= 0).all()

    def test_atr_with_gaps(self):
        """测试有跳空时的ATR"""
        # 创建有跳空的数据
        high = pd.Series([100, 110, 120, 130, 140])
        low = pd.Series([90, 100, 110, 120, 130])
        close = pd.Series([95, 105, 115, 125, 135])

        atr = TechnicalIndicators.atr(high, low, close, period=3)

        # ATR应该反映跳空
        assert atr.iloc[-1] > 0


@pytest.mark.unit
class TestOBV:
    """测试OBV指标"""

    def test_obv_basic(self):
        """测试基本OBV计算"""
        close = pd.Series([100, 101, 100, 102, 103])
        volume = pd.Series([1000, 2000, 1500, 3000, 2500])

        obv = TechnicalIndicators.obv(close, volume)

        assert len(obv) == len(close)
        # 第一个值应该等于第一个volume
        assert obv.iloc[0] == volume.iloc[0]

        # 上涨日：OBV增加
        # close[1] > close[0]，所以OBV应该增加
        assert obv.iloc[1] > obv.iloc[0]

    def test_obv_cumulative(self):
        """测试OBV累积性质"""
        close = pd.Series([100, 101, 102, 103, 104])
        volume = pd.Series([100, 100, 100, 100, 100])

        obv = TechnicalIndicators.obv(close, volume)

        # 全部上涨，OBV应该单调递增
        assert (obv.diff().dropna() > 0).all()


@pytest.mark.unit
class TestWVAD:
    """测试WVAD指标"""

    def test_wvad_basic(self):
        """测试基本WVAD计算"""
        open_prices = pd.Series([100, 101, 102, 103, 104])
        high = pd.Series([105, 106, 107, 108, 109])
        low = pd.Series([95, 96, 97, 98, 99])
        close = pd.Series([102, 103, 104, 105, 106])
        volume = pd.Series([1000, 2000, 1500, 3000, 2500])

        wvad = TechnicalIndicators.wvad(open_prices, high, low, close, volume)

        assert len(wvad) == len(close)
        # WVAD是累积值
        assert wvad.iloc[-1] != wvad.iloc[0]


@pytest.mark.unit
class TestCCI:
    """测试CCI指标"""

    def test_cci_basic(self):
        """测试基本CCI计算"""
        high = pd.Series(range(100, 200))
        low = pd.Series(range(0, 100))
        close = pd.Series(range(50, 150))

        cci = TechnicalIndicators.cci(high, low, close, period=20)

        assert len(cci) == len(close)
        # CCI可以超出-100到100的范围
        # 验证没有NaN值（除了前面几个值可能为NaN）
        assert cci.iloc[-1] != 0


@pytest.mark.unit
class TestAO:
    """测试AO指标"""

    def test_ao_basic(self):
        """测试基本AO计算"""
        high = pd.Series(range(100, 200))
        low = pd.Series(range(0, 100))

        ao = TechnicalIndicators.ao(high, low, fast=5, slow=34)

        assert len(ao) == len(high)
        # AO = SMA(fast) - SMA(slow)
        assert ao.iloc[-1] != 0


@pytest.mark.unit
class TestMomentum:
    """测试动量指标"""

    def test_momentum_basic(self):
        """测试基本动量计算"""
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

        momentum = TechnicalIndicators.momentum(prices, period=5)

        assert len(momentum) == len(prices)
        # 动量 = 当前价格 - N期前的价格
        # 最后一个值应该是 110 - 105 = 5
        assert momentum.iloc[-1] == 5

    def test_momentum_rising_prices(self):
        """测试上涨价格的动量"""
        prices = pd.Series(range(100, 200))

        momentum = TechnicalIndicators.momentum(prices, period=10)

        # 上涨趋势，动量应该为正（除了前面几个值）
        assert momentum.iloc[-1] > 0


@pytest.mark.unit
class TestROC:
    """测试ROC指标"""

    def test_roc_basic(self):
        """测试基本ROC计算"""
        prices = pd.Series([100, 105, 110, 115, 120, 125])

        roc = TechnicalIndicators.roc(prices, period=5)

        assert len(roc) == len(prices)
        # 最后一个值: (125 - 100) / 100 * 100 = 25%
        assert np.isclose(roc.iloc[-1], 25.0)


@pytest.mark.unit
class TestWilliamsR:
    """测试威廉指标"""

    def test_williams_r_basic(self):
        """测试基本威廉指标计算"""
        high = pd.Series([110, 115, 120, 125, 130])
        low = pd.Series([90, 95, 100, 105, 110])
        close = pd.Series([100, 105, 110, 115, 120])

        willr = TechnicalIndicators.williams_r(high, low, close, period=14)

        assert len(willr) == len(close)
        # 威廉指标应该在-100到0之间
        assert willr.min() >= -100
        assert willr.max() <= 0

    def test_williams_r_range(self):
        """测试威廉指标范围"""
        high = pd.Series(np.random.rand(100) + 100)
        low = pd.Series(np.random.rand(100) + 90)
        close = pd.Series(np.random.rand(100) + 95)

        willr = TechnicalIndicators.williams_r(high, low, close)

        # 所有值应该在-100到0之间
        assert (willr >= -100).all()
        assert (willr <= 0).all()


@pytest.mark.unit
class TestAddIndicators:
    """测试批量添加指标"""

    def test_add_indicators_basic(self, sample_price_data):
        """测试基本添加指标"""
        df = sample_price_data.copy()

        result = add_indicators(df)

        # 验证返回DataFrame
        assert isinstance(result, pd.DataFrame)
        # 原始列应该保留
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

        # 验证添加的指标列
        expected_indicators = [
            "ma5",
            "ma10",
            "ma20",
            "ma60",
            "ma120",
            "ema12",
            "ema26",
            "macd",
            "macd_signal",
            "macd_hist",
            "rsi",
            "k",
            "d",
            "j",
            "boll_upper",
            "boll_middle",
            "boll_lower",
            "atr",
            "obv",
            "cci",
            "ao",
            "roc",
            "willr",
        ]

        for indicator in expected_indicators:
            assert indicator in result.columns

    def test_add_indicators_missing_columns(self):
        """测试缺少必要列的情况"""
        df = pd.DataFrame({"close": [1, 2, 3]})

        with pytest.raises(ValueError, match="缺少必要列"):
            add_indicators(df)

    def test_add_indicators_values(self, sample_price_data):
        """测试添加指标的计算值"""
        df = sample_price_data.copy()

        result = add_indicators(df)

        # 验证MA5计算
        expected_ma5 = result["close"].rolling(window=5, min_periods=1).mean()
        assert np.allclose(result["ma5"], expected_ma5, equal_nan=True)

        # 验证RSI在0-100之间
        assert result["rsi"].min() >= 0
        assert result["rsi"].max() <= 100

        # 验证布林带关系（忽略NaN值）
        assert (result["boll_upper"] >= result["boll_middle"]).all() or result[
            "boll_upper"
        ].isna().any()
        assert (result["boll_middle"] >= result["boll_lower"]).all() or result[
            "boll_lower"
        ].isna().any()
        # 更严格的检查：对于非NaN的值，应该满足关系
        valid_mask = ~(
            result["boll_upper"].isna() | result["boll_middle"].isna() | result["boll_lower"].isna()
        )
        assert (result.loc[valid_mask, "boll_upper"] >= result.loc[valid_mask, "boll_middle"]).all()
        assert (result.loc[valid_mask, "boll_middle"] >= result.loc[valid_mask, "boll_lower"]).all()
