"""
数据处理工具函数
提供数据预处理、清洗、特征工程等功能
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def preprocess_data(
    df: pd.DataFrame,
    fill_method: str = "ffill",
    drop_na: bool = True,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    数据预处理

    Args:
        df: 原始数据
        fill_method: 填充方法 ('ffill', 'bfill', 'mean', 'median')
        drop_na: 是否删除缺失值
        normalize: 是否标准化

    Returns:
        处理后的数据
    """
    data = df.copy()

    # 填充缺失值
    if fill_method == "ffill":
        data = data.fillna(method="ffill")
    elif fill_method == "bfill":
        data = data.fillna(method="bfill")
    elif fill_method == "mean":
        data = data.fillna(data.mean())
    elif fill_method == "median":
        data = data.fillna(data.median())

    # 删除缺失值
    if drop_na:
        data = data.dropna()

    # 标准化
    if normalize:
        data = normalize_data(data, method="zscore")

    return data


def normalize_data(df: pd.DataFrame, method: str = "zscore", columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    数据标准化

    Args:
        df: 数据
        method: 标准化方法 ('zscore', 'minmax')
        columns: 要标准化的列，None表示所有数值列

    Returns:
        标准化后的数据
    """
    data = df.copy()

    # 确定要标准化的列
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    # 标准化
    if method == "zscore":
        for col in columns:
            if col in data.columns:
                mean = data[col].mean()
                std = data[col].std()
                if std > 0:
                    data[col] = (data[col] - mean) / std
    elif method == "minmax":
        for col in columns:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    data[col] = (data[col] - min_val) / (max_val - min_val)

    return data


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    method: str = "time",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    数据分割

    Args:
        df: 数据
        train_ratio: 训练集比例
        method: 分割方法 ('time', 'random')

    Returns:
        (训练集, 测试集)
    """
    if method == "time":
        split_idx = int(len(df) * train_ratio)
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
    else:  # random
        train_data = df.sample(frac=train_ratio, random_state=42)
        test_data = df.drop(train_data.index)

    return train_data, test_data


def serialize_dataframe(df: pd.DataFrame) -> Dict:
    """
    序列化DataFrame

    Args:
        df: DataFrame

    Returns:
        序列化后的字典
    """
    return {
        "data": df.values.tolist(),
        "columns": df.columns.tolist(),
        "index": df.index.tolist(),
    }


def deserialize_dataframe(data: Dict) -> pd.DataFrame:
    """
    反序列化DataFrame

    Args:
        data: 序列化数据

    Returns:
        DataFrame
    """
    return pd.DataFrame(
        data=data["data"],
        columns=data["columns"],
        index=data["index"],
    )


def export_to_csv(df: pd.DataFrame, file_path: str, **kwargs):
    """
    导出数据到CSV

    Args:
        df: DataFrame
        file_path: 文件路径
        **kwargs: pandas to_csv的额外参数
    """
    df.to_csv(file_path, **kwargs)


def export_to_parquet(df: pd.DataFrame, file_path: str, **kwargs):
    """
    导出数据到Parquet

    Args:
        df: DataFrame
        file_path: 文件路径
        **kwargs: pandas to_parquet的额外参数
    """
    df.to_parquet(file_path, **kwargs)


def validate_data(df: pd.DataFrame) -> List[str]:
    """
    验证数据质量

    Args:
        df: DataFrame

    Returns:
        问题列表
    """
    issues = []

    # 检查缺失值
    if df.isnull().sum().sum() > 0:
        issues.append(f"发现缺失值: {df.isnull().sum().sum()}个")

    # 检查重复值
    if df.duplicated().sum() > 0:
        issues.append(f"发现重复值: {df.duplicated().sum()}个")

    # 检查负值（对于价格和成交量）
    for col in ["close", "volume"]:
        if col in df.columns and (df[col] < 0).sum() > 0:
            issues.append(f"{col}列存在负值")

    # 检查异常值（使用3σ原则）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            outliers = ((df[col] - mean).abs() > 3 * std).sum()
            if outliers > 0:
                issues.append(f"{col}列发现{outliers}个异常值")

    return issues


def detect_outliers(df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, List[int]]:
    """
    检测异常值

    Args:
        df: DataFrame
        threshold: 异常值阈值（标准差倍数）

    Returns:
        {列名: [异常值索引列表]}
    """
    outliers = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        if std > 0:
            outlier_mask = (df[col] - mean).abs() > threshold * std
            outlier_indices = df[outlier_mask].index.tolist()

            if outlier_indices:
                outliers[col] = outlier_indices

    return outliers


def check_data_consistency(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    检查多股票数据一致性

    Args:
        data_dict: {股票代码: DataFrame}

    Returns:
        {股票代码: [问题列表]}
    """
    issues = {}

    # 获取日期范围
    date_ranges = {}
    for symbol, df in data_dict.items():
        if "date" in df.columns:
            date_ranges[symbol] = (df["date"].min(), df["date"].max())

    # 检查日期范围是否一致
    if len(date_ranges) > 1:
        reference_range = list(date_ranges.values())[0]

        for symbol, date_range in date_ranges.items():
            symbol_issues = []

            if date_range != reference_range:
                symbol_issues.append(f"日期范围不一致: {date_range}")

            if symbol_issues:
                issues[symbol] = symbol_issues

    # 检查列一致性
    reference_columns = None
    for symbol, df in data_dict.items():
        if reference_columns is None:
            reference_columns = set(df.columns)
        else:
            if set(df.columns) != reference_columns:
                if symbol not in issues:
                    issues[symbol] = []
                issues[symbol].append(f"列不一致: {set(df.columns) - reference_columns}")

    return issues


__all__ = [
    "preprocess_data",
    "normalize_data",
    "split_data",
    "serialize_dataframe",
    "deserialize_dataframe",
    "export_to_csv",
    "export_to_parquet",
    "validate_data",
    "detect_outliers",
    "check_data_consistency",
]
