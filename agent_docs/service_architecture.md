# Service Architecture

> Đọc file này khi: cần hiểu kiến trúc tổng thể, thêm module mới, sửa luồng dữ
> liệu giữa các thành phần, hoặc **gom code từ hai phần (training + on-device)
> thành một repo**.
>
> Triết lý: tách biệt rõ trách nhiệm nhưng **không rời rạc** — training và
> runtime chia sẻ thư viện chung trong `src/` để decode/cleaning đồng nhất.
> Single-thread main loop, không over-engineering.

## 1. Mục tiêu hệ thống

Hệ thống có hai mục tiêu chính:

1. **Training (offline, trên máy dev)** — từ dữ liệu CAN logs, train mô hình
   **CNN1D** để dự đoán SoC, rồi export sang `.tflite`.
2. **Runtime (online, trên Pi 4)** — đọc CAN ở **~10Hz** (match dataset
   training ~7Hz median, xem `database_schema.md`), cleaning, resample về
   1Hz, tính **3 nguồn SoC song song** (BMS / Coulomb Counting / CNN1D), ước
   lượng quãng đường, hiển thị lên màn hình OBD + web dashboard, log CSV để
   retrain.

## 2. Kiến trúc tổng

```
                ┌──────────────────────────────────────────┐
                │   NOTEBOOKS (training, máy dev)          │
                │   import từ src/ để decode/cleaning      │
                │   đồng nhất với runtime                  │
                └────────────────┬─────────────────────────┘
                                 │ produces
                                 ▼
                       models/soc_cnn1d.tflite
                       configs/*.yaml
                                 │ consumed by
 ┌───────────────────────────────┼───────────────────────────────────────┐
 │           RUNTIME (Pi 4, single-thread main loop)                     │
 │                                                                       │
 │   CAN bus (~10Hz)                                                     │
 │      │                                                                │
 │      ▼                                                                │
 │   can_reader: decode frame  ◀── shared với notebooks                  │
 │      │                                                                │
 │      ▼                                                                │
 │   preprocessing: cleaning, gate, ring buffer  ◀── shared              │
 │      │                                                                │
 │      ├─────────────────┬───────────────────┐                          │
 │      ▼                 ▼                   ▼                          │
 │   SoC #1: BMS    SoC #2: Coulomb     SoC #3: CNN1D                    │
 │   (passthrough,  Counter (∫I dt,     (TFLite, mỗi 1s                  │
 │    10Hz)         10Hz)                trên window 1Hz)                │
 │      │                 │                   │                          │
 │      └─────────────────┼───────────────────┘                          │
 │                        ▼                                              │
 │              display: 3 battery icon              ──▶ màn hình OBD    │
 │              + web dashboard                      ──▶ điện thoại      │
 │                        │                                              │
 │                        ▼                                              │
 │              range_estimator (dùng SoC #3 + behavior)                 │
 │                        │                                              │
 │                        ▼                                              │
 │              logger ──▶ data/processed/*.csv                          │
 └────────────────────────┼──────────────────────────────────────────────┘
                          │ feedback
                          ▼
                    notebooks/ (retrain)
```

**Liên kết giữa training và runtime — không rời rạc:**

- `src/can_reader/` và `src/preprocessing/` là **shared library**. Notebook
  training import từ đây để đảm bảo decode và cleaning **giống hệt** runtime.
  Tránh tình trạng training quen với một format mà runtime cho format khác.
- `models/*.tflite` — output của training, input của runtime.
- `configs/*.yaml` — CAN IDs, battery specs, EWMA params, dùng chung.
- `data/processed/*.csv` — log từ runtime đẩy ngược về để retrain.

## 3. Cấu trúc thư mục

```
project/
├── data/                      # raw CAN logs + processed CSVs
├── models/                    # .tflite files (giao điểm training ↔ runtime)
├── configs/                   # yaml: CAN IDs, battery specs, EWMA params
├── src/                       # core code — runtime VÀ training cùng dùng
│   ├── can_reader/            # python-can wrapper, frame decoder [shared]
│   ├── preprocessing/         # cleaning, resample 10→1Hz, windowing [shared]
│   ├── coulomb_counter/       # SoC#2: tích phân dòng [runtime]
│   ├── soc_inference/         # SoC#3: CNN1D TFLite runner [runtime]
│   ├── range_estimator/       # energy-based + behavior layer [runtime]
│   ├── display/               # màn hình OBD + web dashboard Flask/FastAPI
│   ├── logger/                # CSV writer
│   └── main.py                # entry point — vòng lặp 10Hz
├── notebooks/                 # training + EDA, import từ src/
│   ├── 01_eda.ipynb
│   ├── 02_train_cnn1d.ipynb
│   └── 03_export_tflite.ipynb
├── deployment/                # systemd service files cho Pi
└── tests/
```

**Quy ước [shared] / [runtime]:**

- `[shared]` — module dùng được ở cả runtime VÀ notebook training. Phải pure,
  không phụ thuộc CAN hardware, test được offline.
- `[runtime]` — chỉ chạy on-Pi. Có thể phụ thuộc TFLite runtime, hardware.

## 4. Single-thread main loop (10Hz)

Toàn bộ runtime logic chạy trong **một vòng lặp duy nhất** ở ~10Hz. Tần số này
chọn để match với rate decode CAN thực tế từ dataset training (~7Hz median —
xem `database_schema.md`). Không threading cho phần xử lý dữ liệu — đơn giản,
dễ debug, đủ nhanh cho Pi 4.

**Cấu trúc loop (pseudo-code):**

```python
# src/main.py
TICK_HZ = 10
TICK_DT = 1.0 / TICK_HZ        # 100 ms
INFERENCE_EVERY_N_TICK = 10    # infer mỗi 1 giây

state = SharedState()
state_lock = threading.Lock()

# Web dashboard chạy thread riêng, chỉ ĐỌC state
start_web_server(state, state_lock)

tick = 0
while True:
    t0 = time.monotonic()

    # --- 10Hz: mỗi tick ---
    frame = can_bus.recv(timeout=TICK_DT)
    if frame is not None:
        decoded = can_reader.decode(frame)         # {V, I, T, speed, soc_bms}
        cleaned = preprocessing.clean(decoded)     # gate, unit check

        with state_lock:
            state.append_buffer(cleaned)                       # ring buffer
            state.soc_bms = cleaned.soc_bms                    # SoC #1
            state.soc_cc  = coulomb_counter.update(            # SoC #2
                cleaned.current, dt=TICK_DT
            )

    # --- 1Hz: mỗi 10 tick ---
    if tick % INFERENCE_EVERY_N_TICK == 0:
        window = preprocessing.resample_window(state.buffer)   # 10→1Hz
        soc_model = soc_inference.predict(window)              # SoC #3
        range_km  = range_estimator.compute(soc_model, window)

        with state_lock:
            state.soc_model = soc_model
            state.range_km  = range_km

        logger.write(state)

    tick += 1
    # giữ nhịp 10Hz
    time.sleep(max(0, TICK_DT - (time.monotonic() - t0)))
```

**Tại sao chọn single-thread ở 10Hz:**

- 10Hz = 100ms/tick. Mỗi tick chỉ làm: read frame, decode, clean, update CC.
  Đo thực tế trên Pi 4 ~1-2ms → còn rất nhiều margin (>95%).
- Inference (~20-50ms) chạy mỗi 1 giây, nằm gọn trong slot tick mà nó rơi vào,
  hoặc trễ tối đa 1 tick — không ảnh hưởng CAN reading vì recv có timeout.
- Web dashboard chạy thread riêng nhưng **chỉ đọc** `state` (không sửa) → lock
  rất nhẹ, không gây contention.
- 10Hz match với rate thực tế của BMS xe (theo dataset). Đọc nhanh hơn cũng
  không có thêm thông tin — chỉ lãng phí CPU vì sau đó vẫn resample về 1Hz.

**Khi nào nâng tick rate lên 100Hz:** chỉ khi đo thực tế trên Pi thấy BMS xe
phát signal nhanh hơn 10Hz và resolution cao có lợi (vd: cần bắt spike regen
braking, Coulomb counter cần dt nhỏ hơn). Đo trước, đổi sau. Đổi chỉ cần sửa
`TICK_HZ` và `INFERENCE_EVERY_N_TICK` (giữ inference 1Hz) + resize ring buffer.

**Khi nào không đủ single-thread:** nếu thực đo inference > 50ms và làm trễ
CAN reading, lúc đó mới tách inference sang thread riêng + queue. KHÔNG làm
trước khi đo.

## 5. Ba nguồn SoC song song

Đây là feature lõi của display: hiển thị **3 nguồn SoC độc lập** dưới dạng 3
battery icon riêng biệt với note rõ nguồn.

| Nguồn | Module | Tần số cập nhật | Tính chất |
|---|---|---|---|
| **#1 — BMS** | `can_reader` (passthrough) | 10Hz | BMS xe tự tính, decode từ frame. Tin cậy nhưng đôi khi conservative. |
| **#2 — Coulomb Counting** | `coulomb_counter` | 10Hz | `∫I dt` trên Pi. Đơn giản nhưng **drift theo thời gian** (bias sensor tích lũy). |
| **#3 — CNN1D model** | `soc_inference` | 1Hz | Model học từ V/I/T window. Best estimate, dùng cho range estimator. |

**Vì sao hiển thị cả 3:** mỗi nguồn có điểm mạnh/yếu khác nhau. Hiển thị song
song để so sánh trực quan — chính là điểm để defend "model CNN1D tốt hơn
baseline (CC)" và đối chiếu với ground truth (BMS).

**Nguồn dùng cho range estimator:** mặc định lấy **SoC #3 (CNN1D)** vì đây là
best estimate. Có thể đổi qua config nếu cần.

## 6. Trách nhiệm từng module

### `src/can_reader/` [shared]
- Đọc frame CAN qua `python-can` (chỉ ở runtime).
- Decode theo CAN IDs trong `configs/can_ids.yaml` (pure function — notebook
  cũng dùng được).
- **Input:** raw CAN frame (bytes).
- **Output:** dict `{timestamp, pack_voltage, pack_current, temp, speed, soc_bms, ...}`.
- **Quy tắc:** frame lỗi/checksum sai → log warning + skip, KHÔNG crash.

### `src/preprocessing/` [shared]
- Cleaning: gate giá trị ngoài vật lý, NaN handling.
- **Đảo dấu dòng điện** khi load training CSV (xem `database_schema.md` §1).
- Ring buffer giữ N sample gần nhất (default 10 sample = 1 giây ở 10Hz).
- Resample 10Hz → 1Hz cho model: trung bình (hoặc downsample) trên cửa sổ
  10 sample.
- Windowing: cắt window kích thước model cần.
- **Quy tắc:** pure function, không I/O, không hardware. Test được offline.

### `src/coulomb_counter/` [runtime]
- Maintain `soc_cc` qua tích phân dòng:
  `soc_cc -= I * dt / capacity_ah / 3600 * 100`.
- Initialize từ `soc_bms` lúc khởi động.
- Reset mỗi cycle (cắm sạc → 100%) để giới hạn drift.
- **Output:** SoC #2, dạng %, 0-100.
- **Quy tắc:** quy ước `I > 0` khi xả.

### `src/soc_inference/` [runtime]
- Load `models/soc_cnn1d.tflite` lúc khởi động.
- Maintain window các sample đã preprocessing.
- Run inference mỗi 1 giây.
- **Input:** window từ `preprocessing`.
- **Output:** SoC #3, SoH (dạng %, 0-100).
- **Quy tắc:** không truy cập CAN raw, không train.

### `src/range_estimator/` [runtime]
- Energy-based: Wh/km baseline qua EWMA (xem giáo trình chương 1).
- Behavior layer: avg speed, accel std, stop ratio trong cửa sổ 60s
  (chương 3).
- Linear regression × baseline → adjusted Wh/km (chương 4).
- `range_km = (SoC × pack_capacity_Wh × SoH) / Wh_per_km_adj`.
- **Input:** SoC #3, SoH, speed/current từ buffer.
- **Output:** `range_km`.

### `src/display/` [runtime]
- **Màn hình OBD** (vật lý gắn xe): 3 battery icon riêng biệt cho 3 SoC + range
  + Wh/km.
- **Web dashboard** (Flask hoặc FastAPI, xem trên điện thoại): mirror nội dung
  OBD + thêm chart timeline.
- Chạy thread riêng, chỉ ĐỌC `state` qua lock.
- **Quy tắc:** không chứa business logic, chỉ hiển thị.

### `src/logger/` [runtime]
- Ghi CSV mỗi 1 giây: timestamp, signal đã decode, 3 SoC, SoH, range.
- File CSV là **đầu vào để retrain** sau này.
- **Output:** `data/processed/*.csv` (schema chi tiết trong `database_schema.md` §2).

### `src/main.py` [runtime]
- Entry point. Vòng lặp 10Hz mô tả ở mục 4.
- Wire các module lại, KHÔNG chứa business logic của module nào.

## 7. Dependency rules

**Trong runtime, dependency đi một chiều theo data flow:**

```
can_reader → preprocessing → [coulomb_counter, soc_inference] → range_estimator
                          → display
                          → logger
```

**KHÔNG cho phép:**
- `preprocessing → can_reader` (preprocessing không quyết định cách đọc CAN).
- `soc_inference → preprocessing` của giai đoạn khác (chỉ nhận window, không
  gọi ngược).
- Bất kỳ module runtime nào → notebook code.

**Training (notebooks) → runtime (src/):**

- Notebook ĐƯỢC PHÉP import từ `src/can_reader/`, `src/preprocessing/` (các
  module [shared]).
- Notebook KHÔNG import các module [runtime].
- `src/` KHÔNG BAO GIỜ import notebook.

## 8. Logging

Mọi module phải log. Dùng `logging` chuẩn Python (xem `code_conventions.md` §8).

- `info`: connect CAN, load model, write CSV chunk.
- `warning`: bad CAN frame skipped, NaN trong sample.
- `error`: model file not found, CAN bus disconnect.

## 9. Nguyên tắc kiến trúc

- **Separation of concerns** — mỗi module một trách nhiệm.
- **Shared library, not duplicated code** — decode + preprocessing nằm trong
  `src/`, dùng chung training và runtime.
- **Single-thread main loop** — đơn giản, đủ nhanh, dễ debug.
- **Clear data flow** — một chiều, không vòng.
- **Testable** — module [shared] test được offline với CSV mẫu.
- **Do not over-engineer** — không thêm thread, queue, abstraction nếu chưa đo
  thấy cần.

---

## 10. Hướng dẫn cho Claude Code khi gom hai phần code

> **Phần dành riêng cho Claude Code.** Tài liệu này là "hợp đồng tích hợp" khi
> gom code training (của user) và code on-device (của teammate) vào một repo.

### Workflow gom code

**Bước 1 — Đọc hiểu trước, không sửa ngay.**

- Đọc toàn bộ file này.
- Đọc file `utils.py` (hoặc tương đương) của user.
- Đọc code của teammate.
- Tóm tắt 1-2 đoạn: hai phần code đang làm gì, khớp/không khớp với kiến trúc
  này ở chỗ nào.

**Bước 2 — Lập bảng mapping.**

| Code hiện có | Thuộc module nào | Đặt ở đâu |
|---|---|---|
| `utils.py::load_can_csv()` | training data loading | `notebooks/lib/` hoặc cell notebook |
| `utils.py::clean_data()` | preprocessing | `src/preprocessing/` (shared!) |
| `utils.py::decode_frame()` | can_reader | `src/can_reader/` (shared!) |
| `utils.py::CNN1D` class | model definition | `notebooks/lib/model.py` |
| `friend_code/can_loop.py` | runtime main loop | `src/main.py` |
| ... | ... | ... |

**Bước 3 — Hỏi user nếu chưa rõ.**

Bắt buộc hỏi nếu:
- Hàm không khớp rõ ràng với module nào.
- Hai code base định nghĩa cùng một thứ theo cách khác nhau (vd: hai cách
  decode cùng một CAN ID, hai version Coulomb counter).
- Refactor đụng tới >3 file hoặc thay đổi API public.

KHÔNG cần hỏi nếu:
- Chỉ move/rename file theo bảng mapping.
- Sửa import path sau khi move.

**Bước 4 — Trình bày plan trước khi thực thi.**

Liệt kê hành động cụ thể (vd: "move `utils.py::decode_frame` sang
`src/can_reader/decoder.py`"). User confirm rồi mới làm.

**Bước 5 — Tái cấu trúc bảo thủ.**

- **Giữ behavior không đổi** khi refactor cấu trúc.
- **Không over-engineering** — không thêm class, abstraction "phòng khi cần".
- **Không thêm dependency mới** — nếu thiếu, dừng và hỏi.
- **Một commit = một loại thay đổi** (move riêng, fix import riêng, sửa logic
  riêng).

### Các điểm dễ va chạm khi gom

- **CAN signal decoding** — nguồn sự thật duy nhất phải là `src/can_reader/`
  với config `configs/can_ids.yaml`. Notebook và runtime đều import từ đây.
  Nếu user và teammate đang decode khác nhau, hỏi user version nào đúng.
- **Quy ước dấu dòng điện** — `I > 0` khi xả trong code; **training CSV thì
  ngược** (`I < 0` khi xả). Đảo dấu khi load CSV, không sửa ở chỗ khác. Xem
  `database_schema.md` §1.
- **Đơn vị** — V, A, Ah, SoC/SoH dạng % (0-100, KHÔNG phải 0-1). Verify
  trước khi gom.
- **3 nguồn SoC phải tên rõ ràng** — không gọi chung là `soc`. Dùng `soc_bms`,
  `soc_cc`, `soc_model` để tránh nhầm.
- **Model interface** — runtime chỉ load `.tflite`. Nếu teammate đang load
  `.pt` trực tiếp, dấu hiệu chưa qua bước export — dừng và hỏi.
- **Tick timing** — main loop phải giữ được nhịp 100ms (10Hz). Nếu thấy code
  có `time.sleep(1)` hoặc I/O blocking trong loop chính, là red flag.
