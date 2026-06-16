# VinFast Evo 200 Lite — SoC/SoH Prediction

Giải mã CAN bus của xe điện VinFast Evo 200 Lite → dự đoán **SoC/SoH** bằng
deep learning (**CNN1D**) → ước lượng quãng đường còn lại (range) → hiển thị
trên **màn hình OBD + web dashboard**. Toàn bộ runtime deploy trên
**Raspberry Pi 4 (8GB)**.

Code phải **đơn giản, rõ ràng, dễ đọc, dễ hiểu**. Metric đánh giá chỉ ở mức
cần thiết (MAE + RMSE, không R²).

Hai đặc điểm cốt lõi:
- Dashboard hiển thị **3 nguồn SoC song song** (BMS từ CAN / Coulomb Counting /
  CNN1D model) dưới dạng 3 battery icon riêng để so sánh.
- Range estimator viết theo **giáo trình chương 1–4** (EWMA baseline +
  Coulomb counting cho SoH + 3 behavior features + linear regression).

## Rules (luôn áp dụng)

- Hỏi làm rõ trước các task phức tạp (ask clarifying questions before complex tasks).
- Trình bày plan và danh sách task trước khi thực thi (show plan before executing).
- Giữ code đơn giản — không thêm abstraction, dependency, threading/async,
  hay metric nếu chưa có lý do rõ ràng.
- Tôn trọng Pi 4 (8GB) — inference dùng `.tflite`, không phải `.pt`.
- Khi gom hoặc refactor code giữa training và runtime → đọc
  `service_architecture.md` trước và theo workflow 5 bước trong đó.
- Khi gặp lỗi đã biết (Coulomb bias, sign convention, sensor startup) → đọc
  `debugging_notes.md` trước khi đoán.
- **Ngôn ngữ:** tên hàm/biến/file **tiếng Anh** (`snake_case` theo
  `code_conventions.md`). Docstring và comment giải thích **tiếng Việt** để
  dễ hiểu cho đồ án và bảo vệ.
- Khi viết data pipeline hoặc CNN1D model → đọc 2 file `references/*` để
  giữ đúng kiến trúc 4 stage và normalize đúng cách (fit train only, lưu
  scaler kèm checkpoint).

## Tài liệu chi tiết — đọc file tương ứng khi task liên quan

- `agent_docs/building_the_project.md` — điểm vào: mục tiêu, tech stack, cấu trúc thư mục, lệnh build/run.
- `agent_docs/service_architecture.md` — kiến trúc, luồng dữ liệu, single-thread loop 10Hz, 3 nguồn SoC, workflow gom code. **Đọc khi đụng nhiều module hoặc thay đổi kiến trúc.**
- `agent_docs/code_conventions.md` — quy ước code, naming, docstring, Git commit. **Đọc khi viết hoặc sửa code.**
- `agent_docs/running_tests.md` — cách chạy và viết test, smoke test CAN/TFLite. **Đọc khi chạy hoặc viết test.**
- `agent_docs/database_schema.md` — schema CSV training (Evo200_*.csv, 7 cột header tiếng Việt) và schema runtime log. **Đọc khi đụng I/O dữ liệu hoặc thêm signal mới.**
- `agent_docs/service_communication_patterns.md` — 3 kênh giao tiếp (function call / shared state + lock / file system). **Đọc khi thêm thread hoặc sửa cách module trao đổi dữ liệu.**
- `agent_docs/debugging_notes.md` — các bug đã biết và cách phân tích (Coulomb bias, sign convention, sensor startup, ...). **Đọc khi debug hoặc gặp số liệu nghi ngờ.**
- `agent_docs/references/data_pipeline_reference.md` — kiến trúc 4 stage cho data processing CNN (Discovery → Preprocess → Scale → Window). **Đọc khi viết `src/preprocessing/` hoặc notebook EDA/training.**
- `agent_docs/references/cnn_model_reference.md` — kiến trúc CNN1D model + training loop + normalize workflow + plot bắt buộc. **Đọc khi thiết kế model, training, hoặc inference runtime.**
