# Debugging Notes

> Đọc file này khi: gặp số liệu nghi ngờ, debug pipeline, hoặc trước khi đoán
> nguyên nhân một lỗi đã có hồ sơ.
>
> Triết lý: ghi lại bug + cách phân tích để **không lặp lại debug**. Mỗi mục
> là một "vụ án" — triệu chứng → giả thuyết → cách kiểm chứng → kết luận.

## 1. Coulomb Counter — bias hệ thống +3.59%

### Triệu chứng

Chạy Coulomb counter trên 15 file dataset `Evo200_Mixed*.csv`:

```
Mean MAE  : 3.76%
Mean RMSE : 4.92%
Mean Bias : +3.59%     ← bất thường
Mean Corr : 0.9907
Files analyzed: 15
Samples total : 135093
```

Correlation 0.99 nghĩa là **shape khớp**, nhưng bias +3.59% là sai số **có hệ
thống** (không phải random noise). Coulomb counter dự đoán **cao hơn** SoC từ
BMS một khoảng cố định.

### Vì sao đáng debug

Random error → smoothing/filter xử lý được. Bias hệ thống → KHÔNG xử lý được
bằng filter. Phải tìm gốc rễ.

### Giả thuyết theo thứ tự ưu tiên

#### Giả thuyết 1 — Sai sample rate / dt

Coulomb counter: `soc -= I * dt / (capacity_ah * 3600) * 100`.

Dataset Evo200 sample rate **~7Hz median, KHÔNG ĐỀU** (xem `database_schema.md`
§1). Nếu code giả định `dt = 1.0` (1Hz) hoặc `dt = 0.1` (10Hz fixed), tích
phân sẽ sai theo hệ số.

**Cách kiểm chứng:**

```python
# Tính dt thực từ timestamp, không hardcode
dt_series = df["timestamp"].diff().dt.total_seconds()
print(dt_series.describe())
# Nếu median != hằng số code đang dùng → đây là nguyên nhân
```

**Hệ quả:** nếu code đang dùng `dt = 0.1` (giả định 10Hz) nhưng thực tế
median ~0.143s (7Hz), tích phân sẽ **lớn hơn** ~43% — tương đương bias dương.
Đây là giả thuyết ưu tiên #1.

#### Giả thuyết 2 — Quy ước dấu dòng điện

`database_schema.md` §1.4 đã ghi: dataset CSV có `I < 0` khi xả; convention
dự án `I > 0` khi xả.

Nếu code Coulomb counter đảo dấu **không nhất quán** (đảo ở loader rồi lại
đảo trong update, hoặc quên đảo), kết quả sẽ lệch hệ thống.

**Cách kiểm chứng:**

```python
# Sau khi load, in vài sample lúc xe đang chạy:
print(df[df["speed_kmh"] > 10].head()[["pack_current_a", "soc_bms"]])
# Nếu speed > 10 km/h mà current âm → chưa đảo dấu, hoặc đảo 2 lần
```

#### Giả thuyết 3 — Capacity_Ah dùng nameplate, không phải measured

Code công thức: `soc -= I * dt / (capacity_ah * 3600) * 100`. Nếu
`capacity_ah` lấy giá trị nameplate (vd 32Ah) nhưng pin thực có capacity
khác (vd 30Ah do tuổi pin / nhiệt độ), tích phân sẽ chia cho mẫu sai → bias.

**Cách kiểm chứng:** đo capacity thật từ một cycle xả 100%→0%:

```python
full_discharge = df[ (df["soc_bms_start"] >= 95) & (df["soc_bms_end"] <= 5) ]
measured_ah = abs((full_discharge["pack_current_a"] * dt).sum()) / 3600
print(f"Measured: {measured_ah:.2f} Ah, nameplate: 32 Ah")
```

Nếu measured khác nameplate >5%, đây là nguyên nhân (hoặc một phần).

#### Giả thuyết 4 — Sensor startup chưa skip

Vài sample đầu file có `SOC = 0`, `Nhiet Do = 0` (BMS chưa init — xem
`database_schema.md` §1.4). Nếu code initialize `soc_cc` từ
`df["soc_bms"].iloc[0]` mà không skip startup, init sai → cộng dồn bias.

**Cách kiểm chứng:** in 20 row đầu sau khi load, xem có row nào `soc_bms = 0`
trong khi voltage normal không.

### Cách approach khi sửa

1. **Kiểm chứng từng giả thuyết theo thứ tự** (#1 → #4). Đừng sửa nhiều thứ
   cùng lúc — không biết cái nào fix bug.
2. Sau mỗi sửa: chạy lại trên 15 file, in lại stats. Nếu bias giảm rõ → đúng
   hướng.
3. Target: `|Mean Bias| < 1%`. Vẫn còn bias nhỏ là chấp nhận được (sensor
   thực tế luôn có bias).
4. **Đừng "cân chỉnh" bằng cách trừ một hằng số.** Cách đó che bug, không
   sửa bug.

## 2. Quy ước dấu dòng điện — cạm bẫy nhập 2 lần

### Triệu chứng

SoC Coulomb counter **đi ngược chiều SoC BMS** (BMS giảm, CC tăng), hoặc
giảm gấp đôi tốc độ bình thường.

### Nguyên nhân

Quy ước dấu được đảo **HAI** chỗ thay vì một:
- Đảo lần 1 ở `src/preprocessing/loader.py` khi load training CSV.
- Đảo lần 2 ở `src/coulomb_counter/` vì lập trình viên nhớ "ở đây phải đảo".

Hai lần đảo = không đảo, lại sai theo hướng khác.

### Quy tắc

Đảo dấu **DUY NHẤT một lần** ở chỗ load CSV (xem `database_schema.md` §1.4).
Mọi module downstream giả định convention dự án `I > 0` khi xả — không đảo
nữa.

### Cách kiểm chứng nhanh

```python
# Khi xe đang chạy (speed > 0, không sạc):
sample = df[df["speed_kmh"] > 10].iloc[100]
assert sample["pack_current_a"] > 0, "Sau preprocessing, xả phải có I > 0"
```

## 3. Sensor startup — vài sample đầu không tin cậy

### Triệu chứng

- Plot SoC theo thời gian: vài giây đầu SoC = 0 rồi nhảy lên giá trị thật.
- Plot nhiệt độ: vài giây đầu T = 0°C rồi nhảy lên ~25°C.
- Coulomb counter initialize sai vì `soc_cc_init = soc_bms[0] = 0`.

### Nguyên nhân

BMS và sensor nhiệt độ chưa init xong trong ~1-2 giây đầu. Dataset ghi luôn
giá trị 0 thay vì NaN.

### Cách xử lý

Skip rows đầu trong `src/preprocessing/loader.py` cho tới khi đồng thời:
- `soc_bms >= 5` (BMS đã có giá trị)
- `temp_c >= 10` (sensor nhiệt đã ready)

Ngưỡng đặt trong `configs/preprocessing.yaml`.

```python
def skip_sensor_startup(df: pd.DataFrame) -> pd.DataFrame:
    valid = (df["soc_bms"] >= 5) & (df["temp_c"] >= 10)
    first_valid_idx = valid.idxmax()  # row đầu tiên True
    return df.loc[first_valid_idx:].reset_index(drop=True)
```

## 4. Thoi Gian — parse không có ngày

### Triệu chứng

Khi sort/join theo timestamp, một số file có data "lùi thời gian" — sort sai.

### Nguyên nhân

Cột `Thoi Gian` chỉ có `HH:MM:SS`, không có ngày (xem `database_schema.md`
§1.4). Phiên đo qua nửa đêm → `23:59:59` → `00:00:00` → parse thành cùng
ngày, thứ tự sai.

### Cách xử lý

Khi parse, phát hiện step lùi (current < previous) → cộng thêm 1 ngày.
Hoặc lấy ngày từ tên file / metadata.

## 6. CoulombCountingEngine — sai quy ước dấu trong formula

### Triệu chứng

Khi chạy `CoulombCountingEngine` từ `bms_realtime_display.py` trên dữ liệu
training dataset:
- Dòng điện nhập vào là `current_a < 0` khi xe xả (convention CSV gốc).
- SoC tính toán **tăng lên** khi xe đang chạy (và điện áp giảm, SoC BMS cũng giảm).
- Kết quả vô lý: SoC Coulomb counter vượt quá 100%.

### Nguyên nhân

Formula trong `CoulombCountingEngine`:
```python
if current_a >= 0:
    dq = (current_a * self.eta_c) * (dt / 3600.0)
else:
    dq = (current_a / self.eta_d) * (dt / 3600.0)

self.cumulative_q += dq
self.soc = 100.0 - (self.cumulative_q / self.Q_EFF) * 100.0
```

**Vấn đề:** Formula giả định `I > 0` khi xả (project convention), nhưng
dữ liệu đầu vào là `I < 0` khi xả (CSV convention).

Khi xả:
- `current_a < 0` (ví dụ -15A)
- `dq = (-15 / eta_d) * (dt/3600) = âm`
- `cumulative_q` trở thành âm
- `soc = 100 - (âm/Q) * 100 = 100 + dương > 100%` ← **SAI**

### Cách kiểm chứng

```python
# Khi xe đang xả: speed > 10 km/h, soc_bms giảm
# SoC Coulomb Counter phải GIẢM, không được tăng
sample = df[df["speed_kmh"] > 10]
assert (sample["soc_cc"].diff() <= 0).all(), "Xả phải làm SoC giảm"
```

### Cách fix

**Tùy chọn 1 (Khuyên dùng):** Dùng `CoulombCounter` từ `src/coulomb_counter/counter.py`
- Đã được viết với quy ước `I > 0 = xả` từ đầu.
- Formula đúng: `ΔSoC = -(I × Δt / 3600) / Q_ah × 100`.
- Nhận `dt` thực từ timestamp (không hardcode).

**Tùy chọn 2:** Fix formula trong `CoulombCountingEngine`:
```python
# Đảo dấu current đầu vào trước khi tính
current_flipped = -current_a  # I > 0 = xả
# Rồi dùng formula cũ
```

**Quy tắc:** Đảo dấu **DUY NHẤT một lần**, tại decoder.py (runtime CAN)
hoặc loader.py (training CSV). Module downstream không đảo lại.

### Dòng thời gian phát hiện

- **Phát hiện:** Khi refactor training code (`utils.py`) vào runtime (`src/`)
  và tạo `CoulombCounter` mới, so sánh hành vi với
  `bms_realtime_display.py::CoulombCountingEngine`.
- **Kết luận:** Bug sign convention chỉ ảnh hưởng khi `current_a < 0`,
  tức dataset training hoặc nếu CAN decoder quên đảo dấu.

## 8. Mock replay — MOCK_CSV_START_ROW phải >= 10000 để smoke test có ý nghĩa

### Triệu chứng

Chạy `MOCK_CAN=1 python -m src.main`, CC và model output **không thay đổi đáng kể**
trong vài phút đầu: CC kẹt ~100%, model output gần bằng giá trị mặc định/mean.

### Nguyên nhân

`Evo200_Mixed1.csv` (file mock mặc định) có cấu trúc:

- **Rows 0–100 (0.2 phút replay):** BMS=100%, I≈0.03A (idle sau sạc đầy).
  `should_reset` kích liên tục → CC anchored ở 100%.
- **Row 101 (data gap):** BMS nhảy thẳng 100%→78% trong 1 row duy nhất
  — xe đã chạy mà không log.
- **Rows 101–5000 (~8 phút replay):** BMS=78%, I≈0.03–0.13A (vẫn idle).
  CC chỉ drift ~0.4% sau 8 phút — dashboard hiện 100%.
- **Row 513+:** I tăng lên >2A (xe bắt đầu chạy), nhưng BMS đã ở 78%
  và current vẫn thấp trước row ~10000.
- **Row 10000+ (~16 phút replay):** driving thật, I trung bình ~6–9A,
  BMS giảm từng %, CC drift rõ (~5%/10 phút).

### Cách chạy đúng để thấy behavior thật

```bash
# Windows PowerShell
$env:MOCK_CAN="1"
$env:MOCK_CSV_START_ROW="10000"
python -m src.main
```

```bash
# Linux / bash
MOCK_CAN=1 MOCK_CSV_START_ROW=10000 python -m src.main
```

Kỳ vọng khi chạy từ row 10000:

- CC khởi đầu ≈ 68% (BMS tại row 10000), drift xuống theo discharge thật.
- Model output dao động theo input thật, chênh BMS ~5–6%.
- Range biến động theo SoC và speed.

### Lý do KHÔNG sửa code

Row offset là vấn đề **test setup**, không phải logic. Dataset có data gap là
đặc điểm của dữ liệu thực, không phải bug. Dùng `MOCK_CSV_START_ROW` để nhảy
vào đoạn CSV có driving data.

## 7. Quy trình debug chung — đọc trước khi đoán

Khi gặp số liệu nghi ngờ:

1. **Đọc file này trước.** Có thể bug đã có hồ sơ.
2. **Đo, đừng đoán.** In `df.describe()`, plot histogram, sanity check
   range giá trị.
3. **Một thay đổi một lần.** Sửa nhiều thứ cùng lúc → không xác định được
   thay đổi nào fix.
4. **Lưu kết quả "trước/sau".** Stats trước sửa và sau sửa lưu lại trong
   git commit message hoặc PR description.
5. **Nếu fix một bug mới, thêm mục vào file này.** Người sau (hoặc Claude
   Code lần sau) không phải debug lại.
