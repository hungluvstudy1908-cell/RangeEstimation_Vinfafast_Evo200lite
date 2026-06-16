# Running Tests

> Đọc file này khi: chạy test, viết test mới, hoặc verify dự án trước khi commit/PR.
>
> Triết lý: test **đơn giản, đủ dùng**. Không ép chỉ tiêu coverage, không thêm
> tooling rườm rà. Mục tiêu là bắt được regression cơ bản và verify pipeline
> không vỡ.

Chạy test trước khi:

- Tạo Pull Request hoặc merge code.
- Train/evaluate một model mới (đảm bảo pipeline còn hoạt động).
- Share kết quả với team.

## 1. Môi trường

Dự án hiện đang phát triển trên **Windows**. Pi 4 chưa deploy; phần dành cho
Linux ghi sẵn để khi deploy chỉ việc theo.

### Tạo virtual environment (lần đầu)

```powershell
python -m venv .venv
```

### Kích hoạt venv

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS (cho Pi sau này):
```bash
source .venv/bin/activate
```

### Verify Python version

```
python --version
```

Mong đợi: `Python 3.11.x` (đồng bộ giữa máy dev và Pi để tránh sai khác behavior).

## 2. Cài dependency

```
pip install -r requirements.txt
```

Verify nhanh:

```
pip list
```

Các package quan trọng cần có:

- `numpy`, `pandas` — xử lý dữ liệu
- `torch` — train model
- `python-can` — đọc CAN frame
- `pytest` — chạy test
- `tensorflow` hoặc `tflite-runtime` — convert/chạy TFLite (xem
  `agent_docs/building_the_project.md` cho luồng PyTorch → TFLite)

## 3. Cấu trúc thư mục test

```
project/
├── src/
├── tests/
│   ├── test_can_reader.py        # decode frame CAN
│   ├── test_soc_inference.py     # TFLite runner
│   ├── test_range_estimator.py   # energy + behavior
│   ├── test_logger.py            # CSV writer
│   └── fixtures/
│       └── sample_can_frames.bin # frame mẫu để test offline
└── data/
```

Test KHÔNG được phụ thuộc phần cứng CAN thật. Dùng frame mẫu trong
`tests/fixtures/` hoặc mock.

## 4. Chạy tất cả test

```
pytest
```

Mong đợi: tất cả test pass, không có failure.

## 5. Chạy một file hoặc một test

Một file:
```
pytest tests/test_can_reader.py
```

Một test theo tên (match substring):
```
pytest -k decode_soc
```

Verbose (xem từng test pass/fail):
```
pytest -v
```

## 6. Coverage (tùy chọn)

Theo triết lý dự án — **không ép chỉ tiêu coverage**. Tuy nhiên có thể chạy
để xem chỗ nào chưa được test:

```
pip install pytest-cov
pytest --cov=src
```

Dùng output như một gợi ý, không phải gate cho merge.

## 7. Test đặc thù dự án

### 7.1. Test CAN decoder

Mỗi decoder cần ít nhất:

- Một test với frame **hợp lệ** đã biết kết quả mong đợi.
- Một test với frame **lỗi** (checksum sai, độ dài sai) → phải trả về `None`
  hoặc raise đúng exception, KHÔNG crash.

```python
def test_decode_pack_voltage_valid():
    # Frame mẫu: pack_voltage = 350.4 V
    frame = bytes.fromhex("DAOD000000000000")
    result = decode_pack_voltage(frame)
    assert result == 350.4

def test_decode_pack_voltage_invalid_length():
    bad_frame = b"\x00\x01"  # quá ngắn
    assert decode_pack_voltage(bad_frame) is None
```

### 7.2. Test TFLite runner (SoC inference)

Smoke test — chỉ verify model load được và infer ra số trong khoảng hợp lệ:

```python
def test_tflite_runner_smoke():
    runner = TFLiteSocRunner("models/soc_lstm.tflite")
    dummy_window = np.zeros((200, 4), dtype=np.float32)  # 200 sample, 4 channel
    soc = runner.predict(dummy_window)
    assert 0.0 <= soc <= 100.0
```

### 7.3. Sanity check output model

SoC và SoH phải nằm trong `[0, 100]`. Giá trị ngoài khoảng = bug.

```python
def test_soc_output_in_valid_range():
    # ... feed test set ...
    for prediction in predictions:
        assert 0.0 <= prediction <= 100.0
```

### 7.4. Validate dataset (script, không phải pytest)

Kiểm tra log CSV trước khi train:

```
python scripts/check_dataset.py
```

Script này check:

- Missing values.
- Duplicate sample.
- Điện áp / dòng / nhiệt độ ngoài khoảng vật lý hợp lý.

Mong đợi: `Dataset validation passed.`

### 7.5. Smoke test training

KHÔNG phải performance test — chỉ verify pipeline train chạy được:

```
python train.py --epochs 1
```

Mong đợi:
- Không exception.
- Loss giảm hoặc ít nhất là số hữu hạn (không `NaN`, không `inf`).
- File model được lưu ra.

### 7.6. Smoke test inference

```
python -m src.soc_inference
```

Mong đợi:
- Load được `.tflite` model.
- Đọc được frame mẫu (hoặc CAN log).
- In ra SoC/SoH hợp lệ.
- Không exception.

## 8. Lỗi thường gặp

### `ModuleNotFoundError`

```
pip install -r requirements.txt
```

Kiểm tra venv đã activate chưa.

### `FileNotFoundError` cho dataset hoặc model

- Verify đường dẫn trong `configs/`.
- Verify file tồn tại trong `data/` hoặc `models/`.

### CUDA error khi train

```
RuntimeError: CUDA not available
```

→ Fallback CPU hoặc cài lại PyTorch đúng version CUDA. Khi train trên Windows
chưa có GPU, set `device = "cpu"` trong code training.

### TFLite version mismatch

Model convert bằng TF version nào thì runtime phải tương thích. Khi sang Pi
dùng `tflite-runtime`, verify version khớp với version đã convert.

## 9. Checklist trước khi PR / merge

- [ ] `pytest` pass toàn bộ.
- [ ] Validate dataset script pass (nếu có sửa data pipeline).
- [ ] Smoke training pass (nếu có sửa training code).
- [ ] Smoke inference pass (nếu có sửa inference / model).
- [ ] Không có `print` debug còn sót.
- [ ] Tài liệu được cập nhật nếu có thay đổi behavior.

## Tiêu chí "dự án còn khỏe"

✓ `pytest` pass.
✓ Dataset validation pass.
✓ Smoke training chạy được 1 epoch.
✓ Smoke inference ra SoC/SoH trong `[0, 100]`.
✓ Không có warning/error nghiêm trọng trong log.
