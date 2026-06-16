# Code Conventions

> Đọc file này khi: viết hoặc sửa bất kỳ đoạn code nào trong dự án.
>
> Mục tiêu: code **đơn giản, rõ ràng, dễ đọc, dễ hiểu** — đồng bộ giữa các thành viên,
> dễ bảo trì, và thân thiện với AI-assisted development.
>
> Các quy ước về Pull Request và Security được tách riêng (xem `agent_docs/`
> tương ứng khi viết sau).

## 1. Nguyên tắc chung

### Nguyên tắc 1 — Dễ đọc quan trọng hơn "ngắn gọn thông minh"

❌ Không nên:
```python
x = [i*i for i in range(100) if i%3==0 and i%5==0]
```

✅ Nên:
```python
valid_numbers = [
    number * number
    for number in range(100)
    if number % 3 == 0 and number % 5 == 0
]
```

### Nguyên tắc 2 — Một hàm = một trách nhiệm

❌ Không nên:
```python
def process_battery():
    load_data()
    clean_data()
    train_model()
    save_model()
```

✅ Nên:
```python
def load_battery_data(): ...
def clean_battery_data(): ...
def train_soh_model(): ...
def save_trained_model(): ...
```

### Nguyên tắc 3 — Tránh "magic number"

Số có ý nghĩa vật lý (ngưỡng điện áp, dung lượng, sample rate...) phải đặt thành hằng số có tên.

❌ Không nên:
```python
if voltage > 4.2:
    ...
```

✅ Nên:
```python
MAX_CELL_VOLTAGE_V = 4.2

if cell_voltage > MAX_CELL_VOLTAGE_V:
    ...
```

### Nguyên tắc 4 — Không over-engineering

Ưu tiên hàm thuần (pure function) và module phẳng. Không tạo class/abstraction khi một hàm là đủ.

❌ Không nên (over-engineering cho task đơn giản):
```python
class SocDecoderFactory:
    def create_decoder(self, can_id): ...
```

✅ Nên:
```python
def decode_soc(frame: bytes) -> float:
    ...
```

## 2. Quy ước đặt tên

### Biến — `snake_case`

✅ Tốt: `battery_voltage`, `cell_temperature`, `soh_prediction`, `pack_current_a`

❌ Tệ: `BatteryVoltage`, `batteryVoltage`, `x`, `temp1`

### Hằng số — `UPPER_CASE`

Kèm đơn vị trong tên nếu có rủi ro nhầm lẫn.

```python
MAX_CELL_VOLTAGE_V = 4.2
MIN_CELL_VOLTAGE_V = 2.5
SAMPLE_RATE_HZ = 10
NOMINAL_CAPACITY_AH = 32.0
```

### Hàm — bắt đầu bằng động từ

✅ Tốt: `load_data()`, `decode_can_frame()`, `calculate_soh()`, `save_results()`

❌ Tệ: `data()`, `frame()`, `soh()`

### Class — `PascalCase`

```python
class BatteryDataset: ...
class SocPredictor: ...
class CanReader: ...
class KalmanFilter: ...
```

### Đặc thù dự án: tên CAN signal

Khi đặt tên biến cho signal đọc từ CAN, dùng đúng tên trong file `configs/`:

```python
# configs/can_ids.yaml định nghĩa:
#   pack_voltage:  CAN_ID 0x123, byte 0-1
#   pack_current:  CAN_ID 0x123, byte 2-3

pack_voltage = decode_uint16(frame, offset=0) * 0.1   # V
pack_current = decode_int16(frame, offset=2) * 0.01   # A (âm = xả)
```

## 3. Docstring cho hàm public

Mọi hàm public (không bắt đầu bằng `_`) cần docstring. Định dạng Google-style.

```python
def calculate_soh(measured_capacity_ah: float, rated_capacity_ah: float) -> float:
    """
    Tính State of Health của pin.

    Args:
        measured_capacity_ah: Dung lượng đo được (Ah).
        rated_capacity_ah: Dung lượng định mức của pin (Ah).

    Returns:
        SoH dưới dạng phần trăm (0-100).
    """
    return (measured_capacity_ah / rated_capacity_ah) * 100
```

Hàm private (`_helper`) chỉ cần 1 dòng mô tả là đủ.

## 4. Comment

Comment **WHY**, không comment WHAT. Code đã nói WHAT.

❌ Không cần:
```python
i += 1  # increment i
```

✅ Cần — giải thích lý do:
```python
# Bỏ qua sample đầu vì sensor startup không ổn định trong ~50ms đầu.
i += 1
```

✅ Cần — cảnh báo cạm bẫy của dự án:
```python
# CAN raw cho dòng dương khi sạc; project quy ước I>0 khi xả.
# Phải đảo dấu ngay ở decoder, KHÔNG đảo ở chỗ khác.
pack_current_a = -decode_int16(frame, offset=2) * 0.01
```

## 5. Type hints

Bắt buộc cho hàm public. Khuyến khích cho hàm private khi không hiển nhiên.

```python
def predict_soc(
    voltage_window: np.ndarray,
    current_window: np.ndarray,
) -> float:
    ...
```

Đặc biệt với PyTorch / numpy, ghi rõ shape trong docstring nếu có:

```python
def run_inference(window: np.ndarray) -> float:
    """
    Args:
        window: shape (200, 4) — 200 sample, 4 channel (V, I, T, speed).

    Returns:
        SoC dự đoán, 0-100.
    """
```

## 6. Kích thước hàm

- Lý tưởng: 10-50 dòng.
- Quá 50 dòng: cân nhắc tách.
- Quá 100 dòng: gần như chắc chắn phải tách.

Hàm dài thường là dấu hiệu vi phạm Nguyên tắc 2 (một trách nhiệm).

## 7. Tách biệt business logic

Không trộn I/O, training, và visualization trong một hàm.

❌ Không nên:
```python
def do_everything():
    df = pd.read_csv("data.csv")
    model = train_lstm(df)
    torch.save(model, "model.pt")
    plt.plot(...)
    plt.show()
```

✅ Nên — mỗi trách nhiệm một hàm, ghép lại trong main:
```python
def load_dataset(path: str) -> pd.DataFrame: ...
def train_lstm(df: pd.DataFrame) -> nn.Module: ...
def save_model(model: nn.Module, path: str) -> None: ...
def plot_training_curve(history: dict) -> None: ...

if __name__ == "__main__":
    df = load_dataset("data/train.csv")
    model = train_lstm(df)
    save_model(model, "models/lstm.pt")
```

## 8. Logging — không dùng print

Dùng `logging` của Python, không `print` (trừ trong script một lần dùng).

❌ Không nên:
```python
print("Training started")
print(f"Loss = {loss}")
```

✅ Nên:
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Training started")
logger.debug("Loss = %.4f", loss)
```

Mức log:
- `debug` — chi tiết phục vụ gỡ lỗi (giá trị từng batch, từng frame).
- `info` — sự kiện bình thường (bắt đầu training, kết nối CAN thành công).
- `warning` — bất thường nhưng vẫn chạy được (frame CAN lỗi, fallback giá trị).
- `error` — sai nghiêm trọng (không đọc được model, mất kết nối CAN).

## 9. Xử lý lỗi

Không nuốt lỗi im lặng. Bắt đúng loại exception, log đầy đủ.

❌ Không nên:
```python
try:
    train_model()
except:
    pass
```

✅ Nên:
```python
try:
    train_model()
except FileNotFoundError as e:
    logger.error("Dataset not found: %s", e)
    raise
```

Với code đọc CAN — frame lỗi/checksum sai là **chuyện thường xuyên**, không nên crash. Log warning và skip frame đó:

```python
try:
    frame = can_bus.recv(timeout=1.0)
    decoded = decode_frame(frame)
except CanChecksumError as e:
    logger.warning("Bad CAN frame skipped: %s", e)
    return None
```

## 10. Metric đánh giá

Triết lý dự án: **chỉ dùng metric cần thiết**.

- Cho SoC và SoH: **MAE** và **RMSE** là đủ.
- KHÔNG thêm R², MAPE, NLL, v.v. trừ khi có lý do rõ ràng và được yêu cầu.
- Mỗi metric thêm vào phải có một câu giải thích "vì sao cần nó".
- Thêm plot cho cho MAE

```python
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Trả về MAE và RMSE — chỉ 2 metric, đúng nhu cầu dự án."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"mae": mae, "rmse": rmse}
```

## 11. Testing

Mỗi module core cần ít nhất một test cơ bản. Tên test bắt đầu bằng `test_`.

```python
def test_calculate_soh_normal_case():
    soh = calculate_soh(measured_capacity_ah=1.8, rated_capacity_ah=2.0)
    assert soh == 90.0

def test_decode_can_frame_invalid_checksum():
    bad_frame = b"\x00" * 8
    assert decode_can_frame(bad_frame) is None
```

Chi tiết cách viết và chạy test xem `agent_docs/running_tests.md` (sẽ viết sau).

## 12. Quy ước Git commit

Format:

```
type: short description
```

Các loại `type`:

- `feat` — thêm tính năng mới
- `fix` — sửa bug
- `docs` — sửa tài liệu
- `refactor` — đổi cấu trúc code, không đổi behavior
- `test` — thêm/sửa test
- `chore` — việc lặt vặt (cập nhật dependency, format...)

Ví dụ:

```
feat: add LSTM SoH predictor
fix: resolve voltage scaling bug in CAN decoder
docs: update training instructions
refactor: simplify preprocessing pipeline
test: add unit tests for SoH calculation
chore: bump pytorch to 2.3.0
```

Quy tắc bổ sung:
- Mô tả ngắn ≤ 72 ký tự, viết thường, không kết thúc bằng dấu chấm.
- Commit theo từng đơn vị logic — không gộp 5 thay đổi không liên quan vào 1 commit.

## 13. Checklist trước khi push

- Đặt tên theo convention (mục 2).
- Hàm nhỏ, mỗi hàm một trách nhiệm.
- Không có code lặp.
- Test pass (`pytest`).
- Docstring đầy đủ cho hàm public.
- Không có `print` debug còn sót lại.
- Không có code comment-out để đó.
- Commit message theo format mục 12.

**Khi phân vân — ưu tiên dễ đọc.**
