# Database Schema

> Đọc file này khi: thêm/sửa code đọc-ghi file CSV (training dataset hoặc
> runtime log), debug data pipeline, hoặc thêm signal mới từ CAN.
>
> Dự án này không dùng SQL database. "Schema" ở đây là format các file CSV
> trong `data/`.

## Tổng quan

Dự án có **hai loại CSV** với schema khác nhau:

| Loại | Vai trò | Đường dẫn | Sample rate |
|---|---|---|---|
| **Training dataset** | Input cho training (offline) | `data/raw/Evo200_*.csv` | ~7Hz median, không đều |
| **Runtime log** | Output của `logger/` trên Pi, dùng để retrain | `data/processed/runtime_*.csv` | 1Hz (mỗi giây 1 row) |

Hai loại có **schema khác nhau** vì runtime log có thêm 3 nguồn SoC, SoH,
range (xem `service_architecture.md` §5). Code đọc CSV phải biết đang đọc loại
nào.

## 1. Training dataset CSV

### Nguồn

Decode từ USB-CAN adapter trên xe **VinFast Evo 200 Lite**, log thành CSV.
Mỗi file = một phiên đo.

### Vị trí

- Bộ gốc: `D:\DoAn\dataset\EVO200_Dataset\file csv\` (Windows, máy dev).
- Trong repo: copy/symlink vào `data/raw/`.
- 15 file: `Evo200_Mixed1.csv` … `Evo200_Mixed15.csv`, **cùng schema**.

### Cấu trúc cột

| Cột (header gốc) | Kiểu | Đơn vị | Range quan sát | Mô tả |
|---|---|---|---|---|
| `Thoi Gian` | str | `HH:MM:SS` | — | Giờ phút giây. **KHÔNG có ngày.** |
| `Dien Ap (V)` | float | V | 70.0 – 73.6 | Pack voltage (hệ 72V). |
| `Dong Dien (A)` | float | A | −21.0 – +2.1 | Dòng pin. **`I<0` khi xả**, `I>0` khi sạc. |
| `SOC (%)` | int | % | 0 – 100 | SoC từ BMS. |
| `Nhiet Do (C)` | int | °C | 0 – 44 | Nhiệt độ pin. |
| `Van toc (km/h)` | int/float | km/h | 0 – 53 | Tốc độ xe. |
| `ODO (km)` | float | km | tăng dần | Đồng hồ km tích lũy. |

### Đặc điểm

- **Sample rate ~7Hz median**, dao động 1–10 sample/giây (không đều).
- Một file điển hình: ~3 đến 4 giờ chạy, ~45.000 row.

### Cạm bẫy quan trọng — phải xử lý ở `src/preprocessing/`

**1. Quy ước dấu dòng điện NGƯỢC với convention dự án.**

- File CSV: `I < 0` khi xả, `I > 0` khi sạc.
- Convention dự án (`code_conventions.md` §4, `service_architecture.md`):
  `I > 0` khi xả.

→ Ngay khi load CSV, **đảo dấu**: `df["current_a"] = -df["Dong Dien (A)"]`.
KHÔNG đảo ở chỗ khác. Coulomb counter, model, mọi thứ downstream đều giả
định convention dự án.

**2. Sensor startup — vài sample đầu không tin cậy.**

Quan sát thấy:
- `SOC (%) = 0` ở vài sample đầu rồi nhảy lên giá trị thật (BMS chưa init).
- `Nhiet Do (C) = 0` ở vài sample đầu (sensor chưa ready).

→ Trong preprocessing, **skip rows đầu cho tới khi SoC ≥ 5% và Nhiet Do ≥ 10°C**
(hoặc ngưỡng tương đương). Tham số ngưỡng đặt trong `configs/`.

**3. `Thoi Gian` không có ngày.**

→ Nếu một phiên đo qua nửa đêm, time string sẽ "reset" từ `23:59:59` về
`00:00:00`, **phá thứ tự**. Khi parse, phát hiện step lùi → cộng thêm 1 ngày.

→ Khi cần timestamp absolute (vd để join với ground truth), lấy ngày từ tên
file hoặc metadata, không từ cột `Thoi Gian`.

**4. Sample rate không đều.**

→ Nếu model cần input rate đều (vd CNN1D với window 60 sample), preprocessing
phải resample (interpolate hoặc downsample) về rate cố định trước khi tạo
window.

### Cột chuẩn hóa sau preprocessing

Khi load vào pipeline, đổi tên sang snake_case + đơn vị rõ ràng (theo
`code_conventions.md` §2):

| Header gốc | Tên chuẩn hóa |
|---|---|
| `Thoi Gian` | `timestamp` (đã parse, có ngày) |
| `Dien Ap (V)` | `pack_voltage_v` |
| `Dong Dien (A)` | `pack_current_a` (đã đảo dấu!) |
| `SOC (%)` | `soc_bms` |
| `Nhiet Do (C)` | `temp_c` |
| `Van toc (km/h)` | `speed_kmh` |
| `ODO (km)` | `odo_km` |

Đặt phép map này ở **một chỗ duy nhất** trong `src/can_reader/loader.py`
(hoặc `src/preprocessing/loader.py`). Notebook training import từ đây — đừng
viết lại logic load mỗi notebook.

## 2. Runtime log CSV

### Nguồn

Output của `src/logger/` trên Pi, ghi mỗi 1 giây trong vòng lặp chính (xem
`service_architecture.md` §4).

### Vị trí

- Trên Pi: `data/processed/runtime_YYYY-MM-DD_HHMMSS.csv`.
- Mỗi lần khởi động Pi → file mới (tránh ghi đè, giới hạn kích thước).

### Cấu trúc cột (dự kiến)

| Cột | Kiểu | Đơn vị | Mô tả |
|---|---|---|---|
| `timestamp` | str (ISO 8601) | — | `YYYY-MM-DDTHH:MM:SS` (có ngày). |
| `pack_voltage_v` | float | V | Pack voltage. |
| `pack_current_a` | float | A | Dòng. **Quy ước dự án: `I>0` khi xả.** |
| `temp_c` | float | °C | Nhiệt độ pin. |
| `speed_kmh` | float | km/h | Tốc độ. |
| `odo_km` | float | km | Odometer. |
| `soc_bms` | float | % | SoC #1 từ BMS (passthrough). |
| `soc_cc` | float | % | SoC #2 từ Coulomb Counter. |
| `soc_model` | float | % | SoC #3 từ CNN1D model. |
| `soh` | float | % | SoH từ Coulomb Counter (offline, cached). |
| `range_km` | float | km | Quãng đường ước lượng. |
| `wh_per_km` | float | Wh/km | Energy consumption baseline (EWMA). |
| `pack_power_w` | float | W | Công suất tức thời = Σcell_v × \|I\|. |
| `cell_01_v` .. `cell_22_v` | float | V | Điện áp 22 cell pin (LFP, ~3.0–3.65V/cell). Từ CAN 0x311–0x31B. Không dùng làm CNN feature nhưng hiển thị trên trang /bms. |

### Đặc điểm

- 1 row mỗi giây.
- Cùng convention với code (snake_case, `I>0` khi xả, SoC/SoH 0–100).
- File này là **đầu vào để retrain** sau này → schema phải ổn định.

## 3. Mapping training ↔ runtime

Khi retrain với log mới, các cột mapping như sau:

| Runtime log | Training dataset (tương đương) |
|---|---|
| `pack_voltage_v` | `Dien Ap (V)` |
| `pack_current_a` | `−1 × Dong Dien (A)` (đảo dấu) |
| `soc_bms` | `SOC (%)` (đây là ground truth cho training) |
| `temp_c` | `Nhiet Do (C)` |
| `speed_kmh` | `Van toc (km/h)` |
| `odo_km` | `ODO (km)` |
| `soc_cc`, `soc_model`, `soh`, `range_km`, `wh_per_km` | (không có ở training) |

**Ground truth cho training:** `soc_bms` (cột `SOC (%)` ở training dataset).
Model học prediction để khớp với BMS.

## 4. Checklist khi thêm signal mới từ CAN

1. Thêm signal vào `configs/can_ids.yaml`.
2. Thêm decode trong `src/can_reader/decoder.py`.
3. Thêm cột vào schema runtime log (mục 2 ở trên) — và update file này.
4. Thêm cột vào logger writer.
5. Nếu cần dùng làm feature: thêm vào preprocessing và update tài liệu training.

## 4b. Ghi chú — 22 cell voltages

- Lưu trong `SharedState.cell_data: list[float]` (22 phần tử, index 0–21).
- Cập nhật mỗi khi `decode()` trả về key `cell_01_v`..`cell_22_v` (từ CAN 0x311–0x31B).
- Emit qua SocketIO event `update_bms` cùng với voltage tổng, current, power.
- **Không dùng làm CNN1D feature** — training dataset không có cột này.
- Hiển thị trên `/bms` page để diagnostic.

## 5. Cạm bẫy chung — kiểm tra trước khi merge

- **Đơn vị**: V, A, Ah, %, km, km/h, °C. KHÔNG có `0–1` cho SoC/SoH.
- **Quy ước dấu I**: `I > 0` khi xả ở mọi nơi NGOẠI TRỪ raw training CSV.
- **Tên cột**: `snake_case` ở mọi nơi NGOẠI TRỪ raw training CSV (header
  tiếng Việt có khoảng trắng).
- **NaN**: training data có thể có NaN sau khi resample/merge. Preprocessing
  phải xử lý (drop hoặc interpolate) trước khi đưa vào model.
- **Timestamp parsing**: training có `HH:MM:SS` không ngày, runtime có ISO
  8601 đầy đủ. Code đọc CSV phải biết loại nào.
