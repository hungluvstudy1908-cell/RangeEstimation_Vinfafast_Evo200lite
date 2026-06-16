# Prompt cho Claude Code — Incremental Planning Mode

> Dùng prompt này khi: **repo đã setup** (có code, có `agent_docs/`, đã chạy
> được phần dashboard), cần Claude Code SCAN hiện trạng và đề xuất **CHỈ
> những thay đổi cần thiết** — không rewrite từ đầu.
>
> Mục tiêu: integrate code partner (`app.py`, `waveshare_decoder.py`) vào
> đúng cấu trúc, fix các gap so với `agent_docs/`, giữ tối đa những gì đã
> hoạt động.

---

## Prompt

Tôi cần bạn vào **planning mode** — chưa code, chỉ scan + đề xuất kế hoạch
delta tối thiểu.

### Bước 1 — Đọc kỹ trước khi đề xuất

Đọc theo thứ tự dưới đây. KHÔNG bỏ bước:

1. **`CLAUDE.md`** ở root — overview, rules, ngôn ngữ.
2. **Toàn bộ `agent_docs/`** (8 file gồm `references/`).
3. **Tất cả file `.md` ở root** (PROJECT_STATUS, RUN_PROJECT, CODE_REVIEW_*,
   TASKS_*, TRAINING_FIX_*, START_HERE, SETUP_LOCAL, QUICK_RUN) — trạng thái
   hiện tại và lịch sử thay đổi.
4. **Code đã có:**
   - `app.py` ở root (Flask + SocketIO + CAN reader của partner).
   - `waveshare_decoder.py` ở root (CAN decoder gốc của partner).
   - `verify_coulomb_bias.py` ở root (script debug bug Coulomb).
   - `check_outlier_files.py` ở root (utility).
   - `src/` — code đã có (scan kỹ, có thể partner đã move một phần).
   - `notebooks/` — notebook training.
   - `tests/` — test hiện có.
   - `models/` — checkpoint nếu có.
   - `index.html`, `bms.html` — template Flask.
5. **`requirements.txt`** — dependency hiện có.

### Bước 2 — Yêu cầu output (chỉ plan, KHÔNG code)

Trả lời theo 6 phần dưới đây.

**Phần 1 — Tóm tắt hiện trạng repo.**

Mỗi mục 2-4 dòng:

- `app.py` làm gì: decode CAN IDs nào (0x102, 0x201, 0x30A, 0x320, 0x311-0x31B,
  0x309), output gì lên SocketIO (`update_bms`, `update_dash`), threading
  model (Flask thread + CAN reader thread).
- `waveshare_decoder.py` vs `app.py`: cùng đọc Waveshare CAN nhưng parse CAN
  ID khác nhau (byte 4-7 vs byte 5-8). Verify cái nào đúng.
- `src/`, `notebooks/`, `tests/` hiện chứa gì (liệt kê file chính).
- Status từ PROJECT_STATUS.md, TASKS_*.md: dự án đang ở giai đoạn nào, task
  nào đã xong, task nào còn dở.
- Bug Coulomb bias: `verify_coulomb_bias.py` đã verify đến đâu, conclusion gì
  (đọc cả file để biết).
- Dataset: 15 file Evo200_Mixed*.csv đặt ở đâu, đã load thành công chưa.

**Phần 2 — Gap analysis (bảng).**

Đối chiếu hiện trạng vs kiến trúc đích trong `service_architecture.md`:

| Module / concern | Hiện trạng | Mong muốn (theo agent_docs) | Khoảng cách |
|---|---|---|---|
| `can_reader` | `app.py` + `waveshare_decoder.py` ở root | `src/can_reader/` | Move + refactor |
| `preprocessing` | ... | `src/preprocessing/` (shared) | ... |
| `coulomb_counter` | ... | `src/coulomb_counter/` | ... |
| `soc_inference` | ... | `src/soc_inference/` | ... |
| `range_estimator` | ... | `src/range_estimator/` | ... |
| `display` | `app.py` Flask + templates ở root | `src/display/` | Move |
| `logger` | ... | `src/logger/` | ... |
| `configs/can_ids.yaml` | Hard-code trong `app.py` | YAML file | Extract |
| Threading model | 2-thread (Flask + reader) | Single-thread loop 10Hz (§4) | **CONFLICT — xem Phần 4** |
| Sign convention I | App.py decode signed nhưng convention chưa rõ | `I > 0` khi xả | Verify + đảo nếu cần |
| Naming | `real_speed`, `is_park`, `total_vol`... | `snake_case` theo conventions | OK (đã snake_case) |

Bổ sung dòng cho mọi module/concern khác bạn phát hiện.

**Phần 3 — Plan integrate 2 file partner.**

Plan cụ thể, file-level:

| File hiện tại | Logic | Đích | Lý do |
|---|---|---|---|
| `app.py` (CAN reader loop) | `read_and_decode_waveshare()` | `src/can_reader/waveshare.py` | Tách CAN logic khỏi web |
| `app.py` (signal decoders) | parse 0x102, 0x201, 0x30A, 0x320 | `src/can_reader/decoder.py` | Tách decode logic |
| `app.py` (cell voltage decode) | parse 0x311-0x31B | `src/can_reader/decoder.py` | Cùng nhóm decode |
| `app.py` (Flask routes + SocketIO) | `@app.route`, `socketio.emit` | `src/display/app.py` | Web ở display layer |
| `app.py` (CAN ID constants) | hard-code 0x102, ... | `configs/can_ids.yaml` | Extract config |
| `waveshare_decoder.py` | Debug script | DELETE hoặc `scripts/` | Đã có app.py đầy đủ hơn |
| `index.html`, `bms.html` | Template Flask | `src/display/templates/` | Theo Flask convention |

**Lưu ý đặc biệt — 22 cell voltage:**

App.py decode 22 cell voltage (CAN ID 0x311-0x31B, scale 0.0001). Đây là
thông tin pack-level chi tiết chưa có trong `database_schema.md`. Đề xuất:
- Thêm vào `configs/can_ids.yaml` đầy đủ 22 cell.
- Thêm cột vào runtime log schema (`database_schema.md` §2) — gợi ý:
  `cell_voltage_v[0..21]` hoặc array column.
- Không cần dùng làm feature cho CNN1D (vì training dataset không có).
  Nhưng dùng được cho display BMS detail page.

**Phần 4 — Conflict resolution.**

Liệt kê các điểm BẤT NHẤT và đề xuất hướng:

1. **CAN ID parsing offset** — `waveshare_decoder.py` dùng byte 4-7,
   `app.py` dùng byte 5-8. Đề xuất: chạy thử cả hai với 1 frame mẫu thật,
   chọn cái match với CAN ID mong đợi (vd 0x102 thay vì 0x10200). Verify
   bằng `frame[2]` (length byte trong protocol Waveshare 20-byte) trước.

2. **Threading model** — `app.py` đã 2-thread (Flask + reader) chạy ổn,
   `service_architecture.md` §4 chốt single-thread loop. ĐỀ XUẤT 2
   lựa chọn:
   - **Option A — Giữ 2-thread (pragmatic):** đã work, không phá. Update
     `service_architecture.md` §4 để phản ánh pattern thực tế (1 reader
     thread + Flask thread + shared state với lock theo
     `service_communication_patterns.md` §3).
   - **Option B — Refactor về single-thread:** đúng spec, dễ debug. Nhưng
     đụng chạm nhiều, mất ~1 ngày work, không có lợi ích đo lường được.
   - **Khuyến nghị: Option A.** Đợi user confirm.

3. **Sign convention I** — `app.py::raw_current` decode signed. Khi `speed > 10`
   (xe đang chạy = xả), giá trị thực dương hay âm? Cần kiểm tra trên data
   thật. Nếu khớp `I > 0 khi xả` → OK. Nếu ngược → đảo dấu ngay tại decoder.

4. **CAN ID dictionary** — partner đã phát hiện 8 CAN ID có ích (0x102, 0x201,
   0x309, 0x30A, 0x311-0x31B, 0x320). `agent_docs/database_schema.md` chưa
   nhắc đến cụ thể. Đề xuất: tạo `configs/can_ids.yaml` từ thông tin partner
   (đây là nguồn sự thật thực nghiệm).

5. **Global vars trong app.py** — `cell_data`, `real_speed`, `real_soc`... là
   global mutable. Service comm pattern khuyên dùng `SharedState` dataclass
   với lock. Đề xuất refactor thành `SharedState` khi move sang
   `src/display/` (sửa logic tối thiểu — chỉ wrap, không đổi behavior).

**Phần 5 — Câu hỏi cho user.**

Chỉ những điều KHÔNG TỰ QUYẾT ĐƯỢC. Tối đa 5 câu. Ví dụ:

- Threading model: Option A (giữ 2-thread) hay B (refactor single-thread)?
- `waveshare_decoder.py` — delete hay giữ làm script debug trong `scripts/`?
- Templates HTML đặt `src/display/templates/` hay `templates/` ở root?
- Coulomb counter bug: đã có verify script, đã fix chưa? Nếu chưa, ưu tiên
  fix trước khi integrate?

**Phần 6 — Delta task list.**

Mỗi task atomic (1 commit), theo thứ tự an toàn (move trước, sửa sau).
Format: `[TAG] description`. Tag được phép:
- `[MOVE]` — di chuyển file/code, không sửa logic.
- `[EXTRACT]` — kéo hằng số/config ra file YAML.
- `[REFACTOR]` — sửa cấu trúc, không đổi behavior.
- `[FIX]` — sửa bug.
- `[ADD]` — thêm code mới (chỉ khi thật cần).
- `[TEST]` — thêm/sửa test.
- `[DOC]` — update docs trong `agent_docs/`.
- `[DELETE]` — xóa file.

Ví dụ:

```
1. [EXTRACT] CAN ID constants từ app.py → configs/can_ids.yaml
2. [MOVE] app.py::read_and_decode_waveshare → src/can_reader/waveshare.py
3. [MOVE] app.py decode functions → src/can_reader/decoder.py
4. [MOVE] app.py Flask routes → src/display/app.py
5. [MOVE] index.html, bms.html → src/display/templates/
6. [REFACTOR] global vars trong src/display/app.py → SharedState + lock
7. [TEST] tests/test_can_reader.py — verify decode 0x201 với frame mẫu
8. [FIX] Coulomb bias (nếu verify_coulomb_bias.py chưa fix)
9. [DOC] update database_schema.md — thêm 22 cell voltage signals
10. [DELETE] waveshare_decoder.py (đã có app.py đầy đủ hơn)
...
```

**KHÔNG đưa vào task list:**
- Viết lại từ đầu thứ đã có và work.
- Refactor toàn bộ vì "đẹp hơn".
- Thêm abstraction "phòng khi cần".
- Sửa naming nếu đã đúng `snake_case`.

### Constraint

- **KHÔNG rewrite.** Repo đã hoạt động được phần dashboard. Mục tiêu là
  integrate + minimal refactor cho khớp `agent_docs/`.
- **KHÔNG đụng những gì đã work** trừ khi conflict trực tiếp với kiến trúc
  hoặc gây bug.
- **2 file partner phải MOVE vào `src/`.** Không để rời rạc ở root.
- **Một commit = một thay đổi.** Đừng gộp `[MOVE]` + `[REFACTOR]` vào 1 task.
- **Hỏi user trước khi resolve conflict lớn** (Phần 4) — đừng tự quyết
  threading model.
- **Không thêm dependency mới** trừ khi đã có trong `requirements.txt`.

### Cấm tuyệt đối trong session này

- KHÔNG code, KHÔNG sửa file.
- KHÔNG bắt đầu task #1 trước khi tôi confirm plan.
- KHÔNG tự sửa file `agent_docs/*.md` (đó là spec, có nhánh task riêng để
  update sau khi conflict được resolve).
- KHÔNG đề xuất task chỉ vì "best practice" — phải có lý do cụ thể với dự
  án này.

---

## Sau khi nhận plan từ Claude Code

Review checklist:

- [ ] Hiện trạng (Phần 1) chính xác — verify được bằng cách open file thật.
- [ ] Gap analysis (Phần 2) đầy đủ — không bỏ sót concern lớn nào.
- [ ] Plan move (Phần 3) clear: từ đâu → đâu, có lý do.
- [ ] Conflict (Phần 4) liệt kê tất cả conflict thấy được; recommendation
  hợp lý.
- [ ] Câu hỏi (Phần 5) hợp lý, không hỏi điều đã có trong `agent_docs/`.
- [ ] Task list (Phần 6) atomic, an toàn (move trước sửa sau), không
  task nào "refactor toàn bộ".

Nếu OK: reply

```
Plan đã approve. Resolution cho conflict:
1. CAN ID parsing: [chọn cái nào]
2. Threading: [Option A / B]
3. waveshare_decoder.py: [delete / keep]
...

Bắt đầu từ task #1. Dừng sau mỗi task để tôi review trước khi sang task
tiếp theo.
```

Nếu cần sửa: chỉ rõ phần nào, sửa thế nào. Đừng để Claude Code đoán.
