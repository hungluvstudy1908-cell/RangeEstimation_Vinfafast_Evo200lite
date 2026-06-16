"""
Chuẩn hóa dữ liệu (Normalization) cho pipeline huấn luyện CNN1D.

Module này cung cấp hai phương pháp chuẩn hóa:
- Min-Max normalization: đưa giá trị về khoảng [0, 1].
- Z-score normalization: chuẩn hóa theo phân phối chuẩn (mean=0, std=1).

Dùng chung cho cả training (notebook) và runtime inference.
"""

import numpy as np
import pandas as pd


def normalize_minmax(data: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa Min-Max: đưa mỗi cột về khoảng [0, 1].

    Công thức: x_norm = (x - x_min) / (x_max - x_min)

    Phù hợp khi cần giữ nguyên phân phối tương đối của dữ liệu
    và các cột có đơn vị rất khác nhau (V, A, °C, km/h).

    Args:
        data: DataFrame với các cột số cần chuẩn hóa.

    Returns:
        DataFrame cùng kích thước, giá trị trong [0, 1].
    """
    return (data - data.min()) / (data.max() - data.min())


def normalize_zscore(data: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa Z-score: đưa mỗi cột về phân phối chuẩn (mean=0, std=1).

    Công thức: x_norm = (x - mean) / std

    Phù hợp khi dữ liệu gần phân phối chuẩn hoặc khi cần
    xử lý outlier tốt hơn Min-Max.

    Args:
        data: DataFrame với các cột số cần chuẩn hóa.

    Returns:
        DataFrame cùng kích thước, mean ≈ 0, std ≈ 1.
    """
    return (data - data.mean()) / data.std()


def normalize(data: pd.DataFrame, use_minmax: bool = True) -> pd.DataFrame:
    """
    Hàm chuẩn hóa tổng quát, chọn phương pháp qua tham số.

    Args:
        data: DataFrame với các cột số cần chuẩn hóa.
        use_minmax: True → dùng Min-Max; False → dùng Z-score.

    Returns:
        DataFrame đã chuẩn hóa.
    """
    if use_minmax:
        return normalize_minmax(data)
    return normalize_zscore(data)


def fit_minmax(data: pd.DataFrame) -> dict:
    """
    Tính min/max từng cột, trả về dict để lưu vào checkpoint.

    Args:
        data: DataFrame features (chưa normalize).

    Returns:
        Dict dạng {col: {"min": float, "max": float}}.
    """
    return {col: {"min": float(data[col].min()), "max": float(data[col].max())}
            for col in data.columns}


def apply_minmax(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Chuẩn hóa min-max dùng params đã fit từ fit_minmax().

    Map theo tên cột, không theo vị trí. Guard rng==0 → trả 0.0.

    Args:
        data: DataFrame cần normalize.
        params: Dict từ fit_minmax(), {col: {"min": float, "max": float}}.

    Returns:
        DataFrame đã normalize, cùng shape và tên cột.
    """
    result = data.copy()
    for col in data.columns:
        mn, mx = params[col]["min"], params[col]["max"]
        rng = mx - mn
        result[col] = 0.0 if rng == 0.0 else (data[col] - mn) / rng
    return result
