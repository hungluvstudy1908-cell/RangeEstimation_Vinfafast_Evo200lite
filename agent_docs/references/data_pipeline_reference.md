# Data Pipeline Reference

> **ĐÂY LÀ TÀI LIỆU THAM KHẢO**, KHÔNG phải spec để copy nguyên xi.
>
> Mô tả pipeline xử lý data CNN gốc viết cho dataset **LG 18650-HG2** (cell-level,
> McMaster University). Dùng làm **kiến trúc tham khảo** khi viết
> `src/preprocessing/` cho dự án Evo 200 Lite (pack-level).
>
> Đọc file này khi: viết hoặc sửa `src/preprocessing/`, `notebooks/lib/` —
> đặc biệt 4 stage Discovery → Preprocess → Scale → Window.

## Khác biệt khi adapt cho dataset Evo 200 Lite

Pipeline gốc thiết kế cho LG 18650-HG2 (cell-level). Khi áp dụng cho dataset
VinFast Evo 200 Lite (pack-level, xem `database_schema.md`), giữ **kiến trúc
4 stage** nhưng **adapt các điểm sau**:

| Điểm | LG 18650-HG2 (gốc) | Evo 200 Lite (dự án) |
|---|---|---|
| **Cấp độ** | Cell (1 cell 18650) | Pack (toàn bộ pin xe) |
| **Schema CSV** | Header đa cấp, 28 dòng metadata | Header tiếng Việt, không metadata |
| **Quy ước dấu I** | Theo dataset gốc | `I < 0` khi xả → **phải đảo** (xem `database_schema.md` §1.4) |
| **Sample rate** | Đều, resample 1Hz dễ | ~7Hz median, KHÔNG đều → cần resample cẩn thận |
| **Phân chia file** | Folder theo nhiệt độ + loại chu kỳ | Tất cả 15 file Mixed*.csv cùng folder |
| **Train/test split** | Theo loại chu kỳ (Mixed train, HWFET/UDDS/US06/LA92 test) | Theo file: **10 train / 2 val / 3 test** |
| **Sensor startup** | Không có | SoC=0, NhietDo=0 ở vài sample đầu → phải skip |
| **`Status` column** | Có (CHA/DCH/TABLE) | KHÔNG có → bỏ filter này |
| **`window_size`** | 200 (cho LG với 1Hz) | Cân nhắc lại theo sample rate sau resample |

**Bỏ hẳn:**
- Logic filter `Status ∈ {CHA, DCH, TABLE}`.
- Logic phân biệt charge vs discharge cycle (Evo dataset có cả 2 trộn lẫn,
  không cần tách).
- Folder discovery theo nhiệt độ.

**Giữ nguyên triết lý:**
- 4 stage rõ ràng (Discovery → Preprocess → Scale → Window).
- Fit scaler **CHỈ trên train**, apply cho val + test (chống data leakage).
- Lưu scaler kèm checkpoint.
- Cắt window **per-file** rồi mới concat (chống cross-file contamination).

---

# Pipeline gốc (LG 18650-HG2) — bản gốc của tài liệu

## 1. Tổng quan

Mục tiêu của pipeline: biến **các file CSV thô** (mỗi file là một chu kỳ lái/sạc ở một nhiệt độ) thành **các mảng numpy dạng cửa sổ trượt** `X: (N, W, 7)` và `y: (N,)` để nạp thẳng vào CNN.

**Ví dụ trực giác:** hãy tưởng tượng dữ liệu pin như một cuốn nhật ký rất dài ghi từng giây (điện áp, dòng, nhiệt độ...). CNN không đọc cả cuốn một lúc — nó đọc từng **"đoạn 200 giây"** rồi đoán xem pin còn bao nhiêu phần trăm (SOC) ở **giây cuối** của đoạn đó. Pipeline này chính là cái máy cắt cuốn nhật ký thành các đoạn đó.

## 2. Feature & Target

Khớp đúng với `CNN1D_MODEL(in_channels=7)`:

```python
FEATURE_COLUMNS = ["Voltage", "Current", "Temperature", "Power",
                   "Voltage Average", "Current Average", "Power Average"]   # 7 features
TARGET          = "Capacity"   # đã chuẩn hóa về [0,1] → đóng vai trò SOC
```

| Nhóm | Cột | Nguồn gốc |
|------|-----|-----------|
| Đo trực tiếp | `Voltage`, `Current`, `Temperature` | Đọc từ CSV |
| Tính toán | `Power` | `Voltage × Current` |
| Làm mượt | `Voltage Average`, `Current Average`, `Power Average` | Rolling mean của 3 cột trên |
| **Nhãn** | `Capacity` | Chuẩn hóa về `[0,1]` theo chu kỳ sạc/xả |

## 3. Sơ đồ luồng (đúng thứ tự chạy)

```
 STAGE 1            STAGE 2                STAGE 3              STAGE 4
 DISCOVERY          PREPROCESS             SCALE                WINDOW
 ─────────          ──────────             ─────                ──────
 lg_get_file()  →   lg_create_dataset() →  fit_feature_scaler() → build_sequences_from_cycles()
   └ sort theo        └ _prepare_lg_         normalization()        └ create_seq_single_file()
     timestamp          cycle()              └ apply_feature_         (cắt per-file)
   (_read_lg_first      (resample, Power,      scaler()
    _timestamp)         rolling, norm Cap)

 list[str]      →   list[DataFrame]    →   list[DataFrame]    →   X:(N,W,7)  y:(N,)  file_lengths
 file paths         cycles                 cycles (normalized)
```

## 4. Chi tiết từng Stage

### STAGE 1 — File Discovery

Quét thư mục dataset, lọc file theo **nhiệt độ** và **loại chu kỳ**, rồi **sắp xếp theo thời gian**.

| Hàm | Chữ ký | Vai trò |
|-----|--------|---------|
| `_read_lg_first_timestamp` | `(csv_file) → Timestamp` | Đọc đúng **1 dòng đầu** của CSV để lấy mốc thời gian làm khóa sort |
| `lg_get_file` | `(data_path, drive_cycle, charge_cycle, temperatures) → list[str]` | Trả về danh sách path **đã sort theo timestamp** |

> **Tại sao phải sort theo timestamp?** Để giữ đúng trật tự thời gian giữa các file — quan trọng khi đánh giá khả năng tổng quát hóa và khi vẽ chuỗi SOC theo thời gian.

### STAGE 2 — Per-cycle Preprocessing

Mỗi file CSV → một `DataFrame` đã làm sạch.

**Các bước bên trong `_prepare_lg_cycle`:**

1. **Đọc CSV** — bỏ 28 dòng metadata đầu, header đa cấp 2 tầng, parse index thành `datetime`.
2. **Resample 1Hz** (nếu bật):
   - Chu kỳ **xả**: `resample("1s").first()`
   - Chu kỳ **sạc**: bỏ timestamp trùng → `resample("1s").ffill()`
3. **Chuẩn hóa `Capacity` về [0,1]** (chính là nhãn SOC):

   | Chu kỳ | Công thức |
   |--------|-----------|
   | Xả | `Capacity = (Capacity + |min|) / (|min| + ε)` |
   | Sạc | `Capacity = Capacity / (|max| + ε)` |

4. **Lọc trạng thái** hợp lệ: `Status ∈ {CHA, DCH, TABLE}`.
5. **Tính `Power = Voltage × Current`**.
6. **Rolling average** (làm mượt) cho `Voltage / Current / Power`:
   - Cửa sổ mặc định `5000`; nếu đã resample 1Hz → `÷10 = 500`.
7. **`dropna()`** (bỏ NaN do rolling) + reset index.

**`lg_create_dataset`** thêm một lớp lọc: file có **< 500 dòng** sau xử lý sẽ bị bỏ và đưa vào `invalid_paths` (cần để khớp metadata khi vẽ biểu đồ sau này).

> **Ví dụ trực giác cho rolling average:** giống như xem nhiệt độ trung bình **5 phút gần nhất** thay vì con số nhảy múa từng giây — đường sẽ mượt hơn, giúp model "nhìn" được xu hướng thay vì nhiễu.

### STAGE 3 — Feature Scaling

Đưa các feature về cùng thang đo. **Fit một lần trên train**, rồi **apply cho cả train và test**.

| Hàm | Chữ ký | Vai trò |
|-----|--------|---------|
| `fit_feature_scaler` | `(dataset, feature_columns, minmax_norm=True) → scaler(dict)` | Học `offset` + `scale` từ dữ liệu |
| `apply_feature_scaler` | `(dataset, scaler) → DataFrame` | `(x - offset) / scale` |
| `normalization` | `(dataset, scaler, eps=EPSILON) → DataFrame` | Wrapper tiện gọi cho từng cycle |

**Hai chế độ scaler:**

| `minmax_norm` | Công thức | `offset` | `scale` |
|---------------|-----------|----------|---------|
| `True` (mặc định) | Min-Max → `[0,1]` | `min` | `max - min` |
| `False` | Standardize → mean 0, std 1 | `mean` | `std` |

> **Tại sao fit chỉ trên train?** Nếu fit trên cả test → model "nhìn lén" được phân phối của test → **data leakage**, kết quả ảo cao. Pipeline này tránh điều đó bằng cách `pd.concat(train_cycles_raw)` rồi fit, sau đó áp cùng scaler cho test.

### STAGE 4 — Sequence Windowing

Cắt mỗi cycle thành các cửa sổ trượt độ dài `W`. **Target = giá trị tại timestep CUỐI của cửa sổ.**

| Hàm | Chữ ký | Vai trò |
|-----|--------|---------|
| `create_seq_single_file` | `(cycle_df, window_size, feature_cols, target_col) → (X, y)` | Cắt window cho **1 file** |
| `build_sequences_from_cycles` | `(cycles_norm, window_size, feature_cols, target_col) → (X_all, y_all, file_lengths)` | Cắt **per-file** rồi `concatenate` |

**Kích thước đầu ra:**

| Mảng | Shape | Ý nghĩa |
|------|-------|---------|
| `X` (1 file) | `(N, W, F)` với `N = len(cycle) - W` | Các cửa sổ |
| `y` (1 file) | `(N,)` | SOC tại timestep cuối mỗi cửa sổ |
| `file_lengths` | `list[int]` | Số sequence của từng file (cần cho metadata plot) |

> **Vì sao cắt PER-FILE rồi mới concat?** Để tránh **cross-file contamination** — một cửa sổ không bao giờ trộn dữ liệu của 2 file/2 chu kỳ khác nhau.

## 5. Cách dùng (khớp notebook gốc)

```python
import Data_Processing_CNN as dp
import pandas as pd

FEATURE_COLUMNS = ["Voltage", "Current", "Temperature", "Power",
                   "Voltage Average", "Current Average", "Power Average"]
TARGET = ["Capacity"]
window_size = 200

# STAGE 1 — lấy file
lg_train_file = dp.lg_get_file(lg_file_path, ['Mixed'], ['Charge'], TEMPS)
lg_test_file  = dp.lg_get_file(lg_file_path, ['HWFET','UDDS','US06','LA92'], ['Charge'], TEMPS)

# STAGE 2 — load + preprocess
train_cycles_raw, train_invalid = dp.lg_create_dataset(lg_train_file, ['Mixed'], ['Charge'])
test_cycles_raw,  test_invalid  = dp.lg_create_dataset(lg_test_file, ['HWFET','UDDS','US06','LA92'], ['Charge'])

# STAGE 3 — fit scaler trên train, apply cho cả 2
scaler = dp.fit_feature_scaler(pd.concat(train_cycles_raw, ignore_index=True), FEATURE_COLUMNS)
train_cycles_norm = [dp.normalization(c, scaler) for c in train_cycles_raw]
test_cycles_norm  = [dp.normalization(c, scaler) for c in test_cycles_raw]

# STAGE 4 — cắt window
X_train, y_train, train_lengths = dp.build_sequences_from_cycles(
    train_cycles_norm, window_size, FEATURE_COLUMNS, TARGET)
X_test, y_test, test_lengths = dp.build_sequences_from_cycles(
    test_cycles_norm, window_size, FEATURE_COLUMNS, TARGET)
```

## 6. Những điểm cần lưu ý (gotchas)

| # | Vấn đề | Chi tiết |
|---|--------|----------|
| 1 | **`invalid_paths` rất quan trọng** | File bị skip phải được loại khỏi danh sách file test trước khi build metadata plot |
| 2 | **Data leakage** | Luôn `fit_feature_scaler` **chỉ trên train**, rồi `normalization` cho test bằng cùng scaler |
| 3 | **`window_size` ảnh hưởng số sample** | `N = len(cycle) - W`. Tăng `W` → ngữ cảnh nhiều hơn nhưng tổng sample giảm |
| 4 | **Capacity max không tròn 1.0** | Do `|min| + ε` ở mẫu số, max ≈ `1/(1+ε)`. Vô hại |
