# Service Communication Patterns

> Đọc file này khi: thêm thread mới (vd web server, background task), sửa cách
> module trong runtime trao đổi dữ liệu, hoặc debug race condition.
>
> Triết lý: **đơn giản tối đa**. Single-thread main loop + shared state object
> + một lock duy nhất cho web thread. Không message bus, không queue, không
> RPC, không IPC.

## 1. Tổng quan các kênh giao tiếp

Trong dự án có **3 loại "communication"**, mỗi loại có pattern riêng:

| Kênh | Giữa | Cơ chế |
|---|---|---|
| **A. Runtime nội bộ** | Các module trong `src/` (can_reader, preprocessing, soc_inference, …) | Function call trực tiếp trong main loop |
| **B. Runtime ↔ Web dashboard** | Main loop thread ↔ Web server thread | Shared `state` object + `threading.Lock` |
| **C. Training ↔ Runtime** | Notebook trên máy dev ↔ Pi | File system (`.tflite`, `.yaml`, `.csv`) |

Mỗi kênh giải thích chi tiết bên dưới.

## 2. Kênh A — Runtime nội bộ (function call)

Trong `src/main.py`, các module được gọi tuần tự **trong cùng một thread**.
Không có queue, không có pub/sub. Function call thuần.

```python
# src/main.py — pseudo
frame   = can_bus.recv(timeout=TICK_DT)
decoded = can_reader.decode(frame)
cleaned = preprocessing.clean(decoded)
state.append_buffer(cleaned)
state.soc_cc = coulomb_counter.update(cleaned.current, dt=TICK_DT)

if tick % INFERENCE_EVERY_N_TICK == 0:
    window     = preprocessing.resample_window(state.buffer)
    soc_model  = soc_inference.predict(window)
    range_km   = range_estimator.compute(soc_model, window)
    logger.write(state)
```

**Quy tắc:**

- Mỗi module export hàm thuần (input → output), không có hidden state ngoài
  internal buffer hợp lý (vd EWMA giữ giá trị trước).
- Module KHÔNG gọi ngược module ở trên (xem dependency rules trong
  `service_architecture.md` §7).
- Không có exception "lén lút" — module trả về `None` hoặc raise rõ ràng;
  main loop xử lý.

## 3. Kênh B — Runtime ↔ Web dashboard (shared state + lock)

Web dashboard cần dữ liệu realtime nhưng KHÔNG được làm chậm main loop. Giải
pháp: web server chạy thread riêng, **chỉ đọc** một `state` object dùng chung.
Lock đảm bảo đọc/ghi nguyên tử.

### Cấu trúc `SharedState`

Một dataclass đơn giản trong `src/state.py` (hoặc cùng `main.py` nếu nhỏ):

```python
from dataclasses import dataclass, field
from collections import deque

@dataclass
class SharedState:
    # Latest signals
    timestamp:      float = 0.0
    pack_voltage_v: float = 0.0
    pack_current_a: float = 0.0
    temp_c:         float = 0.0
    speed_kmh:      float = 0.0
    odo_km:         float = 0.0

    # 3 nguồn SoC
    soc_bms:   float = 0.0
    soc_cc:    float = 0.0
    soc_model: float = 0.0

    # Derived
    soh:        float = 100.0
    range_km:   float = 0.0
    wh_per_km:  float = 0.0

    # Ring buffer cho preprocessing/inference
    buffer: deque = field(default_factory=lambda: deque(maxlen=10))
```

### Lock pattern

Một `threading.Lock` duy nhất cho toàn bộ state. Không cần fine-grained lock
vì lock chỉ giữ trong vài microsecond.

```python
import threading

state      = SharedState()
state_lock = threading.Lock()
```

### Quy tắc đọc/ghi

| Ai | Có thể làm gì | Cách |
|---|---|---|
| **Main loop thread** | Ghi và đọc state | `with state_lock:` quanh mỗi block ghi |
| **Web server thread** | CHỈ đọc, KHÔNG ghi | `with state_lock:` để snapshot rồi thoát |

**Pattern ghi (main loop):**

```python
with state_lock:
    state.soc_cc = new_soc_cc
    state.soc_model = new_soc_model
    state.range_km = new_range_km
```

**Pattern đọc (web server):**

```python
@app.get("/api/state")
def get_state():
    with state_lock:
        snapshot = {
            "soc_bms":   state.soc_bms,
            "soc_cc":    state.soc_cc,
            "soc_model": state.soc_model,
            "range_km":  state.range_km,
            # ... copy đủ field cần dùng ...
        }
    # release lock TRƯỚC khi serialize JSON / format
    return snapshot
```

### Quy tắc giữ lock

- **Giữ lock càng ngắn càng tốt.** Chỉ làm read/write field, KHÔNG làm I/O,
  KHÔNG serialize JSON, KHÔNG log lệnh chậm trong vùng lock.
- **Không nested lock.** Chỉ có một lock duy nhất. Nếu thấy nhu cầu lock thứ
  hai, dừng và review thiết kế.
- **Không gọi function khác trong lock** trừ access field của state.

### Vì sao đơn giản như vậy là đủ

- Main loop ghi state ~10 lần/giây. Web server đọc khi có request (vài
  lần/giây). Va chạm thấp.
- Mỗi lần giữ lock ~vài µs. Web request latency tổng < 10ms.
- Không cần `multiprocessing`, `asyncio`, `Queue`, hay event bus. Single
  `Lock` + một object đủ dùng.

## 4. Kênh C — Training ↔ Runtime (file system)

Hai phần (notebook training trên máy dev, runtime trên Pi) **không bao giờ
chạy cùng lúc cùng máy**. Giao tiếp bằng file qua thư mục dùng chung:

| File | Hướng | Producer | Consumer | Format |
|---|---|---|---|---|
| `models/soc_cnn1d.tflite` | training → runtime | Notebook export | `soc_inference/` | TFLite binary |
| `configs/*.yaml` | hai chiều, edit thủ công | Người viết config | Cả hai phía | YAML |
| `data/raw/Evo200_*.csv` | bên ngoài → training | USB-CAN log | Notebook training | CSV (xem `database_schema.md` §1) |
| `data/processed/runtime_*.csv` | runtime → training | `logger/` | Notebook training (retrain) | CSV (xem `database_schema.md` §2) |

### Quy tắc

- **Không có file nào edit đồng thời bởi 2 process.** Runtime chỉ ghi
  `data/processed/`. Notebook chỉ đọc.
- **Schema phải ổn định.** Đổi schema = phá tương thích cả training và
  retrain. Khi cần đổi, update `database_schema.md` trước, rồi mới sửa code
  cả hai phía.
- **Tên file có timestamp** để tránh ghi đè: `runtime_2025-05-31_143022.csv`.
- **Atomic write cho `.tflite`:** export ra file `.tmp` rồi `os.rename` sang
  `.tflite` để Pi không load file đang ghi dở.

## 5. Khi nào cần upgrade pattern

Trong dự án này, single-thread + lock đủ dùng. Chỉ upgrade nếu **đo thực tế**
gặp một trong các tình huống:

| Triệu chứng | Upgrade |
|---|---|
| Inference > 50ms làm trễ CAN reading | Tách inference sang thread riêng + `queue.Queue` để pass window |
| Web request latency > 100ms | Cache snapshot mỗi giây, web đọc cache không cần lock |
| Cần stream realtime push sang dashboard | WebSocket (Flask-SocketIO hoặc FastAPI WebSocket) thay polling |
| Logger ghi đĩa làm trễ main loop | Tách logger sang thread riêng + `queue.Queue` |

**KHÔNG upgrade trước khi đo.** Đo bằng `time.monotonic()` quanh block nghi
ngờ, log kết quả, rồi mới quyết định.

## 6. Anti-patterns — không làm

- ❌ Tạo `threading.Lock` thứ hai cho một state khác.
- ❌ Đặt logic trong `state.py` (state phải là dataclass thuần, không có
  method).
- ❌ Web server **ghi** state. Tất cả command từ web (vd reset Coulomb counter)
  phải qua API riêng, không sửa state trực tiếp.
- ❌ `time.sleep()` hoặc I/O blocking trong vùng giữ lock.
- ❌ Pass mutable object (list, deque) ra khỏi lock mà không copy — caller có
  thể sửa và phá invariant của state.
