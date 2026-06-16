# Building the Project

> Tài liệu này mô tả cách dựng và chạy dự án. Là **điểm vào** — chỉ giới thiệu
> bao quát. Chi tiết kiến trúc, luồng dữ liệu, và trách nhiệm từng module xem
> `agent_docs/service_architecture.md`.

## Mục tiêu dự án

Giải mã dữ liệu CAN bus từ xe **VinFast Evo 200 Lite**, dùng làm input cho mô
hình deep learning **CNN1D** để dự đoán **SoC** (State of Charge) và **SoH**
(State of Health) của pin, sau đó **hiển thị trên màn hình OBD + web dashboard**
(xem trên điện thoại). Toàn bộ runtime chạy trên **Raspberry Pi 4 (8GB)**.

Một đặc điểm quan trọng: dashboard hiển thị **3 nguồn SoC song song** (BMS từ
CAN / Coulomb Counting / CNN1D model) dưới dạng 3 battery icon riêng biệt để
so sánh. Xem `service_architecture.md` §5 để hiểu vì sao và cách tính.

Triết lý code: **đơn giản, rõ ràng, dễ đọc, dễ hiểu.** Metric đánh giá chỉ ở
mức cần thiết — không thêm metric/abstraction rườm rà nếu chưa thực sự cần.

## Tech stack

- **Python** — viết utils, đọc/decode CAN, chạy inference, logger, và backend
  web dashboard.
- **Jupyter Notebook** — EDA, training, calibration, export TFLite (trong
  `notebooks/`).
- **PyTorch** — train mô hình **CNN1D** cho SoC.
- **TFLite** — sau khi train xong bằng PyTorch, **convert sang `.tflite`**
  (qua PyTorch → ONNX → TF → TFLite) để deploy và chạy inference trên Pi.
- **NumPy / pandas** — quản lý, cấu trúc, tiền xử lý dữ liệu CAN.
- **python-can** — đọc frame CAN từ phần cứng.
- **Flask hoặc FastAPI** — backend web dashboard.

### Phần cứng deploy

Toàn bộ runtime deploy trên **Raspberry Pi 4 (8GB)**. Model nặng được train
trên máy dev khác, chỉ file `.tflite` đã convert mới chạy trên Pi. Mọi quyết
định về model size, dependency cần cân nhắc giới hạn tài nguyên của Pi.

## Cấu trúc thư mục

```
project/
├── data/                      # raw CAN logs + processed CSVs
├── models/                    # .tflite files (giao điểm training ↔ runtime)
├── configs/                   # yaml: CAN IDs, battery specs, EWMA params
├── src/                       # core code — runtime VÀ training cùng dùng
│   ├── can_reader/            # python-can wrapper, frame decoder    [shared]
│   ├── preprocessing/         # cleaning, resample 100→1Hz, windowing [shared]
│   ├── coulomb_counter/       # SoC#2: tích phân dòng                [runtime]
│   ├── soc_inference/         # SoC#3: CNN1D TFLite runner           [runtime]
│   ├── range_estimator/       # energy-based + behavior layer        [runtime]
│   ├── display/               # màn hình OBD + web dashboard (Flask/FastAPI)
│   ├── logger/                # CSV writer cho retrain sau này
│   └── main.py                # entry point — vòng lặp 100Hz
├── notebooks/                 # training + EDA, import từ src/
│   ├── lib/                   # helper .py (refactor từ utils.py)
│   ├── 01_eda.ipynb
│   ├── 02_train_cnn1d.ipynb
│   └── 03_export_tflite.ipynb
├── deployment/                # systemd service files cho Pi
└── tests/
```

**Quy ước [shared] / [runtime]:**

- `[shared]` — module dùng được ở cả runtime VÀ notebook training. Notebook
  import từ đây để decode và cleaning đồng nhất với runtime.
- `[runtime]` — chỉ chạy on-Pi.

## Đặc điểm runtime (tóm tắt)

Chi tiết đầy đủ trong `service_architecture.md`. Bản tóm tắt:

- **Single-thread main loop ở 100Hz.** Một vòng lặp duy nhất tiếp nhận CAN
  mỗi 10ms. Inference CNN1D chạy mỗi 1 giây (mỗi 100 tick).
- **Resample 100Hz → 1Hz** giữa preprocessing và model.
- **Web dashboard chạy thread riêng**, chỉ ĐỌC state qua lock — không vi
  phạm nguyên tắc single-thread cho phần xử lý dữ liệu.
- **3 nguồn SoC song song** dùng chung data sau cleaning.

## Lệnh thường dùng

Dự án có hai phần: **Python (ML / CAN / inference / backend)** và **dashboard
frontend (Node/npm nếu dùng build process)**.

### Python (ML, CAN, runtime)

- `pip install -r requirements.txt` — cài dependency
- `python -m src.main` — chạy runtime loop 100Hz trên Pi
- `python -m src.can_reader` — test reader/decoder CAN độc lập
- `pytest` — chạy test (chi tiết trong `agent_docs/running_tests.md`)
- `jupyter notebook` — mở notebook để training/EDA

### Dashboard frontend (nếu cần build)

- `npm run dev` — chạy dev server
- `npm test` — chạy test
- `npm run build` — build production
- `npm run lint` — kiểm tra lint

Nếu dashboard chỉ là HTML/JS tĩnh do Flask/FastAPI serve, có thể bỏ qua npm.

## Rules cho Claude Code

1. **Ask clarifying questions before complex tasks.**
   Với bất kỳ task phức tạp nào (thiết kế model, sửa decode logic CAN, thay
   đổi pipeline dữ liệu, đụng tới nhiều module, gom code training/runtime),
   hãy **hỏi làm rõ trước khi bắt tay làm**. Không tự suy diễn yêu cầu mơ hồ.

2. **Show your plan and tasks before executing.**
   Trước khi thực thi, **trình bày kế hoạch và danh sách task** để review, rồi
   mới làm.

3. **Giữ code đơn giản.** Ưu tiên rõ ràng hơn là "thông minh". Không thêm
   abstraction, metric, hay dependency nếu chưa có lý do rõ ràng. KHÔNG thêm
   threading/async cho phần xử lý dữ liệu trừ khi đo thực tế thấy cần.

4. **Tôn trọng giới hạn của Pi 4 (8GB).** Cân nhắc tài nguyên cho mọi thứ
   chạy on-device. Inference phải dùng `.tflite`, không phải `.pt`.

5. **Đọc tài liệu liên quan trước khi sửa.**
   - Sửa code → đọc `code_conventions.md`.
   - Đụng kiến trúc / luồng dữ liệu / gom code → đọc `service_architecture.md`.
   - Chạy/viết test → đọc `running_tests.md`.
