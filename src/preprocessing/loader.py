"""
Loader dữ liệu cho dataset VinFast Evo 200 Lite.

Module này là **nguồn sự thật duy nhất** cho việc đọc file CSV training.
Mọi notebook và runtime đều import từ đây để đảm bảo:
  - Tên cột nhất quán (snake_case tiếng Anh).
  - Quy ước dấu dòng điện nhất quán (I > 0 khi xả).
  - Xử lý timestamp, sensor startup đồng nhất.

KHÔNG đọc file CSV ở chỗ nào khác — luôn dùng load_evo200_csv().
"""

import glob
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Đọc ngưỡng startup từ config — fallback về giá trị mặc định nếu thiếu file
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "preprocessing.yaml"

def _load_startup_thresholds() -> tuple[int, int]:
    """Đọc ngưỡng sensor startup từ configs/preprocessing.yaml."""
    if yaml is None:
        logger.warning("PyYAML not available, using default startup thresholds.")
        return 5, 10

    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        startup = cfg.get("startup", {})
        return (
            startup.get("min_soc_bms_pct", 5),
            startup.get("min_temp_c", 10),
        )
    except FileNotFoundError:
        logger.warning("Không tìm thấy configs/preprocessing.yaml, dùng ngưỡng mặc định.")
        return 5, 10

_STARTUP_MIN_SOC_BMS, _STARTUP_MIN_TEMP_C = _load_startup_thresholds()

# ---------------------------------------------------------------------------
# Hằng số — ánh xạ tên cột gốc tiếng Việt sang tên chuẩn của dự án
# ---------------------------------------------------------------------------

# Header gốc trong file CSV (có khoảng trắng và ký tự tiếng Việt)
_COL_TIME      = "Thoi Gian"
_COL_VOLTAGE   = "Dien Ap (V)"
_COL_CURRENT   = "Dong Dien (A)"
_COL_SOC       = "SOC (%)"
_COL_TEMP      = "Nhiet Do (C)"
_COL_SPEED     = "Van toc (km/h)"
_COL_ODO       = "ODO (km)"

# Tên chuẩn sau khi rename (snake_case + đơn vị rõ ràng)
_RENAME_MAP = {
    _COL_VOLTAGE : "pack_voltage_v",
    _COL_CURRENT : "pack_current_a",   # sẽ đảo dấu sau khi rename
    _COL_SOC     : "soc_bms",
    _COL_TEMP    : "temp_c",
    _COL_SPEED   : "speed_kmh",
    _COL_ODO     : "odo_km",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_evo200_files(data_dir: str) -> list[str]:
    """
    Liệt kê tất cả file CSV dataset Evo200 trong thư mục chỉ định.

    Tìm theo pattern 'Evo200_Mixed*.csv' (không phân biệt hoa thường
    trên Windows).

    Args:
        data_dir: Đường dẫn thư mục chứa file CSV, ví dụ 'data/raw/'.

    Returns:
        Danh sách đường dẫn file, sắp xếp theo tên (Evo200_Mixed1, ..., 15).

    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {data_dir}")

    pattern = os.path.join(data_dir, "Evo200_Mixed*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        logger.warning("Không tìm thấy file Evo200_Mixed*.csv trong: %s", data_dir)

    logger.info("Tìm thấy %d file dataset trong %s", len(files), data_dir)
    return files


def load_evo200_csv(path: str) -> pd.DataFrame:
    """
    Đọc một file CSV dataset Evo200 và trả về DataFrame đã chuẩn hóa.

    Các bước xử lý thực hiện theo thứ tự:
      1. Đọc CSV, rename cột sang snake_case tiếng Anh.
      2. Đảo dấu dòng điện: CSV gốc có I < 0 khi xả; project dùng I > 0 khi xả.
      3. Parse cột 'Thoi Gian' (HH:MM:SS, không có ngày) thành datetime,
         xử lý trường hợp phiên đo qua nửa đêm.
      4. Skip các sample đầu khi BMS/sensor chưa init (SOC = 0, Temp = 0).

    Args:
        path: Đường dẫn file CSV, ví dụ 'data/raw/Evo200_Mixed1.csv'.

    Returns:
        DataFrame với các cột:
          timestamp (datetime), pack_voltage_v (float), pack_current_a (float),
          soc_bms (float), temp_c (float), speed_kmh (float), odo_km (float).
        Index được reset về 0, 1, 2, ...

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file thiếu cột bắt buộc.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    logger.info("Đang đọc: %s", path)

    # --- Bước 1: Đọc và rename cột ---
    df = pd.read_csv(path)
    _validate_columns(df, path)

    df = df.rename(columns=_RENAME_MAP)

    # --- Bước 2: Đảo dấu dòng điện ---
    # CSV gốc: I < 0 khi xả (xe đang chạy), I > 0 khi sạc.
    # Quy ước dự án: I > 0 khi xả. Đảo dấu MỘT LẦN DUY NHẤT tại đây.
    # Mọi module downstream (Coulomb counter, model, ...) giả định I > 0 khi xả.
    df["pack_current_a"] = -df["pack_current_a"]

    # --- Bước 3: Parse timestamp ---
    df["timestamp"] = _parse_timestamp(df[_COL_TIME])
    df = df.drop(columns=[_COL_TIME])

    # --- Bước 4: Skip sensor startup ---
    df = _skip_sensor_startup(df)

    df = df.reset_index(drop=True)

    logger.info(
        "Đã load %d sample từ %s (sau khi skip startup)",
        len(df), Path(path).name
    )
    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, path: str) -> None:
    """Kiểm tra file có đủ các cột bắt buộc không."""
    required = {_COL_TIME, _COL_VOLTAGE, _COL_CURRENT,
                _COL_SOC, _COL_TEMP, _COL_SPEED, _COL_ODO}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"File {path} thiếu cột: {missing}. "
            f"Các cột hiện có: {list(df.columns)}"
        )


def _parse_timestamp(time_series: pd.Series) -> pd.Series:
    """
    Parse cột 'Thoi Gian' dạng 'HH:MM:SS' thành datetime.

    Vì cột không có ngày, dùng ngày cơ sở cố định (2000-01-01).
    Khi phát hiện thời gian đi lùi (phiên đo qua nửa đêm), cộng thêm 1 ngày.

    Args:
        time_series: Series dạng string 'HH:MM:SS'.

    Returns:
        Series kiểu datetime64, tăng đơn điệu.
    """
    base_date = datetime(2000, 1, 1)

    timestamps = []
    day_offset = timedelta(0)
    prev_seconds = None

    for time_str in time_series:
        t = datetime.strptime(str(time_str).strip(), "%H:%M:%S")
        current_seconds = t.hour * 3600 + t.minute * 60 + t.second

        # Phát hiện qua nửa đêm: thời gian đột ngột nhảy về gần 0
        if prev_seconds is not None and current_seconds < prev_seconds - 3600:
            day_offset += timedelta(days=1)

        dt = base_date + day_offset + timedelta(seconds=current_seconds)
        timestamps.append(dt)
        prev_seconds = current_seconds

    return pd.Series(timestamps, name="timestamp")


def _skip_sensor_startup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bỏ qua các sample đầu khi BMS và cảm biến nhiệt chưa khởi động xong.

    Triệu chứng: SOC = 0 và Temp = 0 trong vài giây đầu do BMS chưa init.
    Nếu khởi tạo Coulomb Counter từ soc_bms[0] = 0 → sai ngay từ đầu.

    Giữ từ row đầu tiên thỏa đồng thời:
      - soc_bms >= _STARTUP_MIN_SOC_BMS
      - temp_c  >= _STARTUP_MIN_TEMP_C

    Args:
        df: DataFrame đã rename cột, chưa reset index.

    Returns:
        DataFrame đã cắt bỏ các sample startup không tin cậy.
    """
    valid_mask = (df["soc_bms"] >= _STARTUP_MIN_SOC_BMS) & \
                 (df["temp_c"]  >= _STARTUP_MIN_TEMP_C)

    first_valid = valid_mask.idxmax()

    if first_valid > 0:
        logger.debug(
            "Skip %d sample startup (SOC hoặc Temp chưa ổn định)", first_valid
        )

    return df.loc[first_valid:]
