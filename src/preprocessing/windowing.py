"""
Tạo dataset dạng cửa sổ trượt (sliding window) cho mô hình CNN1D.

Khác biệt so với create_lstm_dataset() gốc (KeiLongW/battery-state-estimation):
  - Sliding window có chồng lấp (overlapping), không phải chunk rời nhau.
    → Tận dụng tối đa dữ liệu, model thấy nhiều ngữ cảnh hơn.
  - Target (y) = soc_bms tại timestep CUỐI của mỗi cửa sổ.
    → "Dự đoán SoC hiện tại dựa trên lịch sử window_size giây gần nhất."
  - Feature (x) = 4 tín hiệu vật lý: V, I, T, tốc độ (không dùng Power).

Dữ liệu đầu vào phải đã qua load_evo200_csv() và normalize().
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Danh sách feature đầu vào cho CNN1D
# Thứ tự này phải khớp với thứ tự kênh khi train và khi inference runtime.
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "pack_voltage_v",   # Điện áp pack (V)
    "pack_current_a",   # Dòng điện (A), I > 0 khi xả
    "temp_c",           # Nhiệt độ pin (°C)
    "speed_kmh",        # Tốc độ xe (km/h)
]

# Cột nhãn (ground truth SoC từ BMS, dùng để train)
TARGET_COL = "soc_bms"


def create_cnn1d_dataset(
    df: pd.DataFrame,
    window_size: int,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tạo dataset dạng sliding window từ DataFrame Evo200 đã chuẩn hóa.

    Với mỗi vị trí i từ 0 đến (N - window_size):
      - x[i] = ma trận tín hiệu từ bước i đến i+window_size-1, shape (window_size, 4)
      - y[i] = soc_bms tại bước i+window_size-1 (timestep cuối của window)

    Ví dụ: window_size=60 → mỗi mẫu là 60 giây lịch sử, dự đoán SoC giây thứ 60.

    Args:
        df: DataFrame từ load_evo200_csv(), đã normalize().
            Phải có đủ các cột trong FEATURE_COLS và TARGET_COL.
        window_size: Số timestep trong mỗi cửa sổ (ví dụ: 60 = 60s ở 1Hz).
        step: Bước nhảy giữa các cửa sổ liên tiếp.
              step=1 → hoàn toàn chồng lấp (dataset lớn nhất).
              step=window_size → không chồng lấp (tương đương utils.py gốc).

    Returns:
        x: shape (n_windows, window_size, n_features) — float64.
        y: shape (n_windows,) — SoC (%) tại cuối mỗi cửa sổ.

    Raises:
        ValueError: Nếu DataFrame thiếu cột hoặc quá ngắn.
    """
    _validate_dataframe(df)

    n_samples = len(df)
    if n_samples < window_size:
        raise ValueError(
            f"DataFrame có {n_samples} dòng, ngắn hơn window_size={window_size}. "
            "Cần nhiều dữ liệu hơn."
        )

    features = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    targets  = df[TARGET_COL].to_numpy(dtype=np.float64)

    # Tạo danh sách các điểm bắt đầu của mỗi cửa sổ
    start_indices = range(0, n_samples - window_size + 1, step)

    x_list = []
    y_list = []

    for start in start_indices:
        end = start + window_size
        x_list.append(features[start:end])        # (window_size, n_features)
        y_list.append(targets[end - 1])            # SoC tại timestep cuối

    x = np.array(x_list, dtype=np.float64)        # (n_windows, window_size, n_features)
    y = np.array(y_list, dtype=np.float64)         # (n_windows,)

    logger.info(
        "Tạo dataset: %d cửa sổ, window_size=%d, step=%d, features=%d",
        len(x), window_size, step, len(FEATURE_COLS)
    )
    return x, y


def keep_only_y_end(y: np.ndarray, step: int) -> np.ndarray:
    """
    Lấy mẫu thưa trên mảng nhãn y bằng cách chọn mỗi 'step' phần tử.

    Hàm này giữ nguyên từ utils.py gốc (KeiLongW). Dùng khi y là 2D
    (n_sequences, sequence_length) và muốn giữ lại mỗi step-th nhãn,
    ví dụ chỉ lấy nhãn cuối mỗi window để so sánh với output LSTM nhiều bước.

    Với CNN1D output đơn (y shape (n,)), gọi keep_only_y_end(y.reshape(-1,1), step).

    Args:
        y: Mảng nhãn 2D, shape (n_sequences, seq_len).
        step: Lấy mỗi step-th phần tử theo trục cột.

    Returns:
        Mảng nhãn đã lấy mẫu thưa, shape (n_sequences, seq_len // step).
    """
    return y[:, ::step]


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _validate_dataframe(df: pd.DataFrame) -> None:
    """Kiểm tra DataFrame có đủ cột feature và target không."""
    required = set(FEATURE_COLS) | {TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame thiếu cột: {missing}. "
            f"Các cột hiện có: {list(df.columns)}"
        )
