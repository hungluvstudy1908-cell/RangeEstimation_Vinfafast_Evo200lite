# CNN Model Reference

> **ĐÂY LÀ TÀI LIỆU THAM KHẢO**, KHÔNG phải spec để copy nguyên xi.
>
> Mô tả pipeline model CNN1D cho bài toán dự đoán SOC, gốc viết cho dataset
> **LG 18650-HG2**. Dùng làm **kiến trúc tham khảo** khi viết
> `src/soc_inference/` và notebook training cho dự án Evo 200 Lite.
>
> Đọc file này khi: thiết kế hoặc sửa kiến trúc model CNN1D, training loop,
> hoặc inference runtime.

## Khác biệt khi adapt cho dataset Evo 200 Lite

Pipeline gốc thiết kế cho LG cell. Khi áp dụng cho Evo 200 Lite (pack-level),
giữ **kiến trúc model** nhưng **adapt các điểm sau**:

| Điểm | LG 18650-HG2 (gốc) | Evo 200 Lite (dự án) |
|---|---|---|
| **Metrics** | MAE + RMSE + R² | **MAE + RMSE thôi** (R² đã bỏ — xem `code_conventions.md`) |
| **`window_size`** | 200 | Cân nhắc lại sau khi resample (vd 60–200) |
| **`in_channels`** | 7 (Voltage, Current, Temp, Power, V_avg, I_avg, P_avg) | Tùy chọn feature — số channel phải khớp |
| **Plots cần thiết** | 4 nhóm phân tích | **Bắt buộc 2 plot tối thiểu** (xem dưới) |
| **Train/test split** | Theo loại chu kỳ | **10 train / 2 val / 3 test** trên 15 file |

### Plots bắt buộc cho Evo 200 Lite

1. **Training curve** — MAE và RMSE qua các epoch, **overlay 2 đường trên cùng 1 chart** (trục y có thể chia tỷ lệ riêng). Có cả train và val curve để dò overfit.
2. **True vs Predicted scatter** — scatter plot SoC dự đoán vs SoC thật trên test set, kèm đường y=x. Xem prediction có bám đường chéo hay không.

Có thể thêm các plot khác nếu muốn, nhưng **2 cái trên là tối thiểu**.

### Lưu ý critical về normalization

Đây là điểm dễ sai nhất khi adapt:

- **Fit scaler CHỈ trên train_set** (10 file), KHÔNG fit trên val+test.
- **Apply scaler cho cả train + val + test** (cùng một scaler).
- **Lưu scaler kèm checkpoint** trong file `.pt` (xem mục 7).
- **Khi inference runtime trên Pi**, load scaler từ checkpoint và áp cho frame
  CAN realtime trước khi đưa vào model.
- **KHÔNG fit lại scaler trên dữ liệu mới** ở runtime — sẽ phá thang đo model
  đã học.

Hiện code training cũ chưa chuẩn normalize. Khi adapt cho dự án, fix triệt
để theo workflow này.

---

# Pipeline gốc (LG 18650-HG2) — bản gốc của tài liệu

## 1. Bức tranh tổng thể

```
[Data Pipeline]            [Tensor & Loader]        [Model]          [Train Loop]            [Eval & Plot]
X,y (numpy)        →       TensorDataset      →     CNN1D_MODEL  →   Huber + Adam +      →   predict_soc()
(N, 200, 7)                DataLoader               (Conv→Conv→         ReduceLROnPlateau     + 4 nhóm biểu đồ
                           batch=64                 GAP→FC)            + EarlyStopping(MAE)
```

**Ví dụ trực giác về CNN cho time-series:** CNN ở đây hoạt động như một **bộ dò mẫu trượt** — nó rê các "kính lúp" (kernel) dọc theo 200 timestep, mỗi kính lúp học một dạng tín hiệu (vd: điện áp tụt nhanh, dòng tăng đột biến). Sau đó nó tổng hợp lại tất cả để đoán SOC. Khác với LSTM (đọc tuần tự, ghi nhớ), CNN nhìn **cục bộ và song song** — nhanh hơn, ít bị vanishing gradient.

## 2. Cấu hình tổng (Hyperparameters)

| Nhóm | Tham số | Giá trị | Ghi chú |
|------|---------|---------|---------|
| **Input** | `window_size` | `200` | Cho LG; Evo cân nhắc lại theo sample rate |
| | `in_channels` | `7` | = số FEATURE_COLUMNS |
| **Conv** | block1 out / block2 out | `64 / 128` | kernel=3, padding=1 (giữ nguyên độ dài W) |
| | dropout1 / dropout2 | `0.3 / 0.2` | Ở phần FC head |
| **Train** | `batch_size` | `64` | |
| | `NUM_EPOCHS` | `100` | |
| | optimizer | `Adam` | `lr=1e-4`, `weight_decay=1e-5` |
| | loss | `HuberLoss(delta=1.0)` | |
| | grad clip | `max_norm=1.0` | Chống bùng nổ gradient |
| **Scheduler** | `ReduceLROnPlateau` | `factor=0.5, patience=8, min_lr=1e-6` | Monitor **MAE** |
| **Early Stop** | `EarlyStopping` | `patience=15, min_delta=1e-5` | Monitor **MAE** |

## 3. Tạo Tensor & DataLoader

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

X_train_tensor = torch.FloatTensor(X_train)   # (N, 200, 7)
y_train_tensor = torch.FloatTensor(y_train)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)
```

| Loader | `shuffle` | Lý do |
|--------|-----------|-------|
| `train_loader` | `True` | Trộn để model không học theo thứ tự |
| `test_loader` | `False` | Giữ nguyên thứ tự để khớp metadata khi plot |

## 4. Kiến trúc model — `CNN1D_MODEL`

### 4.1 Dòng chảy shape

```
Input từ DataLoader : (batch, W=200, 7)
        │ permute(0,2,1)              ← đổi sang (batch, channels, length) cho Conv1d
        ▼
        (batch, 7, 200)
        │ Conv Block 1 : Conv1d(7→64, k=3, pad=1) → BatchNorm → ReLU
        ▼
        (batch, 64, 200)
        │ Conv Block 2 : Conv1d(64→128, k=3, pad=1) → BatchNorm → ReLU
        ▼
        (batch, 128, 200)
        │ AdaptiveAvgPool1d(1)        ← Global Average Pooling
        ▼
        (batch, 128, 1) → squeeze → (batch, 128)
        │ FC head : Dropout(0.3) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→1)
        ▼
Output : (batch, 1)                   ← SOC dự đoán (regression, không activation cuối)
```

### 4.2 Bảng layer

| Tầng | Phép toán | Output shape | Vai trò |
|------|-----------|--------------|---------|
| Permute | `(b, 200, 7) → (b, 7, 200)` | `(b, 7, 200)` | Đưa feature thành "kênh" cho Conv1d |
| Conv Block 1 | Conv1d(7→64) + BN + ReLU | `(b, 64, 200)` | Dò mẫu cục bộ cấp thấp |
| Conv Block 2 | Conv1d(64→128) + BN + ReLU | `(b, 128, 200)` | Dò mẫu phức tạp hơn |
| Global Avg Pool | AdaptiveAvgPool1d(1) | `(b, 128, 1)` | Nén chiều thời gian → bất biến độ dài |
| Flatten | squeeze(-1) | `(b, 128)` | |
| FC head | Dropout→Linear→ReLU→Dropout→Linear | `(b, 1)` | Hồi quy ra SOC |

> **Vì sao dùng Global Average Pooling thay vì Flatten + Linear lớn?** GAP gom 200 timestep thành 1 giá trị trung bình cho mỗi kênh → giảm mạnh số tham số, chống overfit, và giúp model **bất biến với độ dài chuỗi**.
>
> **Vì sao không có activation ở Linear cuối?** Đây là bài **regression** (đoán số thực SOC), không phải phân loại — nên để output tự do.

## 5. Thiết lập huấn luyện

### 5.1 Loss — Huber Loss (`delta=1.0`)

| Vùng sai số | Hành xử | Tính chất |
|-------------|---------|-----------|
| `|error| < 1` | giống **MSE** | Nhạy, học nhanh khi gần đúng |
| `|error| > 1` | giống **MAE** | **Robust** với outlier |

### 5.2 Optimizer & Scheduler

- **Adam** `lr=1e-4`, `weight_decay=1e-5` (L2 regularization nhẹ).
- **ReduceLROnPlateau**: khi **MAE** không giảm sau `patience=8` epoch → `lr ×= 0.5`, sàn `min_lr=1e-6`.
- **Gradient clipping** `max_norm=1.0`: chặn gradient bùng nổ.

### 5.3 Early Stopping (monitor **MAE**)

> **Điểm tinh tế:** EarlyStopping theo dõi **MAE** chứ không phải Huber loss. Lý do: MAE là metric thực tế cần tối ưu, và Huber loss với MAE **không luôn giảm đồng thời**. Cả `scheduler` và `early_stop` đều thống nhất monitor MAE để tránh tín hiệu mâu thuẫn.

- `patience=15`, `min_delta=1e-5`.
- Lưu lại **best state** và `restore()` sau khi dừng.

## 6. Vòng lặp huấn luyện

```
for epoch in 1..100:
    train_loss        = train_epoch(...)    # train + clip grad + Adam step
    val_loss, mae, rmse = eval_epoch(...)   # đánh giá trên val_loader
    scheduler.step(mae)                      # giảm LR theo MAE
    log(epoch, train_loss, val_loss, mae, rmse, lr)
    if early_stop.step(mae, model): break
early_stop.restore(model)
```

### Metrics đánh giá (cho dự án Evo)

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **MAE** | `mean(|pred − true|)` | Sai số tuyệt đối trung bình — metric chính để tối ưu |
| **RMSE** | `sqrt(mean((pred − true)²))` | Phạt sai số lớn nặng hơn |

> Pipeline LG gốc có thêm R², **dự án Evo bỏ R²** (xem `code_conventions.md`).

## 7. Lưu model

```python
torch.save({
    "model_state"  : model.state_dict(),
    "history"      : history,          # cần cho plot training curve
    "best_mae"     : early_stop.best_score,
    "scaler"       : scaler,           # ← BẮT BUỘC: lưu scaler để inference dùng đúng thang đo
    "feature_cols" : FEATURE_COLUMNS,
    "window_size"  : window_size,
    "config"       : {...},
}, "soc_cnn1d_best.pt")
```

> **Quan trọng:** `scaler` được lưu kèm. Khi inference (cả test offline và runtime trên Pi), **phải** dùng đúng scaler này. Nếu không thang đo lệch → dự đoán sai. Đây là gốc của vấn đề "normalize chưa đúng" trong code training cũ.

## 8. Đánh giá & Trực quan hóa

Hàm `predict_soc(model, loader, device)` chạy lại model trên `test_loader` → `y_true`, `y_predicted`. Sau đó gom mọi thứ vào `soc_plot_df` kèm metadata.

**Plots cho dự án Evo (tối thiểu 2):**

| # | Plot | Trả lời câu hỏi |
|---|------|-----------------|
| 1 | **Training curve** — MAE + RMSE overlay (train + val) qua epochs | Model có học/overfit không? |
| 2 | **True vs Predicted scatter** — y=x reference + scatter trên test | Prediction có bám đường thật không? |

Có thể thêm (optional): residual plot, error distribution histogram, chuỗi
SOC theo thời gian cho từng test file.

## 9. Checklist & lưu ý

| # | Điểm | Ghi chú |
|---|------|---------|
| 1 | `in_channels` phải = số feature | Đổi feature thì sửa cả hai chỗ |
| 2 | Scheduler & EarlyStop **cùng monitor MAE** | Tránh tín hiệu mâu thuẫn |
| 3 | `test_loader` để `shuffle=False` | Bắt buộc để khớp metadata khi plot |
| 4 | Lưu `scaler` cùng checkpoint | **Bắt buộc** — cần cho inference đúng thang đo |
| 5 | GAP làm model bất biến độ dài | Có thể đổi `window_size` mà không sửa kiến trúc FC |
| 6 | Train: shuffle. Val/Test: không shuffle | Val cũng nên giữ thứ tự để debug |

## 10. Tóm tắt một dòng

> Dữ liệu Evo 200 Lite được cắt thành cửa sổ `(N, W, F)` → CNN1D 2 lớp Conv + Global Average Pooling + FC head dự đoán SOC tại timestep cuối, huấn luyện bằng Huber loss + Adam, điều phối bởi ReduceLROnPlateau và EarlyStopping cùng theo dõi MAE, rồi đánh giá qua **MAE + RMSE** (không R²) và **2 plot bắt buộc** (training curve + true vs predicted scatter).
