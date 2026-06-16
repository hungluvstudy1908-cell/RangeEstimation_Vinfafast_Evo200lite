"""
Chẩn đoán bias CNN1D SoC: so sánh pipeline runtime (main.py) với PyTorch gốc.

Mục tiêu: xác định tại sao model ra ~50% trong khi BMS/CC ~66%.

Chạy từ thư mục gốc dự án:
    python scripts/diag_cnn_parity.py
"""

import sys
from pathlib import Path

# Đảm bảo import được src/
sys.path.insert(0, str(Path(__file__).parent.parent))

# Hỗ trợ tiếng Việt trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.preprocessing.loader import load_evo200_csv
from src.preprocessing.normalize import apply_minmax

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

TFLITE_PATH = Path("models/soc_cnn1d.tflite")
PT_PATH     = Path("models/soc_cnn1d.pt")
YAML_PATH   = Path("configs/model.yaml")
CSV_PATH    = Path("data/raw/Evo200_Mixed1.csv")

WINDOW_START = 10000   # Vùng driving thật (xem debugging_notes.md §8)
WINDOW_SIZE  = 60
FEATURE_COLS = ["pack_voltage_v", "pack_current_a", "temp_c", "speed_kmh"]


# ---------------------------------------------------------------------------
# CNN1D class — copy y hệt từ notebooks/02_train_cnn1d.ipynb
# (không có file riêng trong src/)
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """Kiến trúc CNN1D dùng để train soc_cnn1d.pt."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input từ caller: (batch, 60, 4) — channels-last
        # Permute sang channels-first để Conv1d hoạt động đúng
        x = x.permute(0, 2, 1)   # (batch, 4, 60)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "") -> None:
    """In dòng ngăn cách."""
    if title:
        pad = max(0, 60 - len(title) - 2)
        print(f"\n{'─' * 3} {title} {'─' * pad}")
    else:
        print("─" * 64)


def _norm_diff(a: dict, b: dict, label_a: str, label_b: str) -> bool:
    """So sánh hai bộ feature_norm, in ra mọi chênh lệch. Trả True nếu giống."""
    identical = True
    all_keys = sorted(set(a) | set(b))
    for col in all_keys:
        if col not in a:
            print(f"  THIẾU trong {label_a}: {col}")
            identical = False
            continue
        if col not in b:
            print(f"  THIẾU trong {label_b}: {col}")
            identical = False
            continue
        mn_diff = abs(a[col]["min"] - b[col]["min"])
        mx_diff = abs(a[col]["max"] - b[col]["max"])
        if mn_diff > 1e-6 or mx_diff > 1e-6:
            print(
                f"  LỆCH '{col}': "
                f"min {a[col]['min']:.4f} vs {b[col]['min']:.4f}  "
                f"max {a[col]['max']:.4f} vs {b[col]['max']:.4f}"
            )
            identical = False
    return identical


# ---------------------------------------------------------------------------
# Phần 1 — Kiểm tra TFLite
# ---------------------------------------------------------------------------

def check_tflite() -> bool:
    """Kiểm tra file TFLite và in thông tin shape nếu tồn tại."""
    _sep("PHẦN 1 — TFLite")
    print(f"Path : {TFLITE_PATH.resolve()}")
    print(f"Tồn tại: {TFLITE_PATH.exists()}")

    if not TFLITE_PATH.exists():
        print()
        print(">>> ROOT CAUSE #1 XÁC NHẬN <<<")
        print("TFLite KHÔNG tồn tại → SocInference rơi vào demo mode:")
        print("    dummy_soc = 50.0 + avg_normalized_current * 5.0  ≈ 50%")
        return False

    # File tồn tại — lấy shape
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter

        interp = Interpreter(model_path=str(TFLITE_PATH))
        interp.allocate_tensors()
        inp  = interp.get_input_details()[0]
        outs = interp.get_output_details()
        print(f"  Input  shape : {inp['shape']}  dtype={inp['dtype'].__name__}")
        for i, o in enumerate(outs):
            print(f"  Output[{i}] shape: {o['shape']}  dtype={o['dtype'].__name__}")
    except Exception as exc:
        print(f"  [Lỗi khi load TFLite interpreter]: {exc}")

    return True


# ---------------------------------------------------------------------------
# Phần 2 — Load .pt và so sánh normalization
# ---------------------------------------------------------------------------

def load_checkpoint_and_compare_norm():
    """Load checkpoint .pt, in keys, so sánh feature_norm với yaml."""
    _sep("PHẦN 2 — Checkpoint .pt vs model.yaml")

    if not PT_PATH.exists():
        print(f"KHÔNG tìm thấy checkpoint: {PT_PATH}")
        return None, None, None

    ckpt = torch.load(str(PT_PATH), map_location="cpu", weights_only=False)
    print(f"Checkpoint keys: {sorted(ckpt.keys())}")
    print(f"best_mae        : {ckpt.get('best_mae', 'N/A')}")
    print(f"best_epoch      : {ckpt.get('best_epoch', 'N/A')}")
    print(f"label_scale (.pt): {ckpt.get('label_scale', 'N/A')}")

    # Đọc yaml
    with open(YAML_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    label_scale_yaml = cfg.get("label_scale", 1.0)
    norm_yaml = cfg.get("normalization", {})
    norm_pt   = ckpt.get("feature_norm", {})

    print(f"label_scale (yaml): {label_scale_yaml}")
    print()
    print("So sánh feature_norm (.pt vs yaml):")
    identical = _norm_diff(norm_pt, norm_yaml, ".pt", "yaml")
    if identical:
        print("  ✓ Giống nhau hoàn toàn")

    return ckpt, norm_yaml, label_scale_yaml


# ---------------------------------------------------------------------------
# Phần 3 — Lấy cửa sổ driving thật
# ---------------------------------------------------------------------------

def load_window():
    """Đọc CSV, lấy rows 10000..10059."""
    _sep("PHẦN 3 — Cửa sổ driving (rows 10000–10059)")

    df = load_evo200_csv(str(CSV_PATH))
    window_end = WINDOW_START + WINDOW_SIZE

    print(f"Tổng số rows CSV (sau startup skip): {len(df)}")
    print(f"Cửa sổ: iloc[{WINDOW_START}:{window_end}]")

    seg = df.iloc[WINDOW_START:window_end]
    soc_truth = seg["soc_bms"].mean()
    print(f"soc_bms trung bình trong cửa sổ (ground truth): {soc_truth:.1f}%")
    print(f"pack_current_a  mean={seg['pack_current_a'].mean():.2f}A  "
          f"min={seg['pack_current_a'].min():.2f}  max={seg['pack_current_a'].max():.2f}")

    window_raw = seg[FEATURE_COLS].reset_index(drop=True)
    return window_raw, soc_truth


# ---------------------------------------------------------------------------
# Phần 4 — Tái tạo pipeline runtime (main.py)
# ---------------------------------------------------------------------------

def replicate_runtime_pipeline(window_raw: "pd.DataFrame", norm_yaml: dict,
                                tflite_exists: bool):
    """Chạy y hệt đường main.py: apply_minmax → .T[np.newaxis,:] → TFLite."""
    _sep("PHẦN 4 — Pipeline runtime (main.py path)")

    # Bước 1: normalize
    window_norm_df = apply_minmax(window_raw, norm_yaml)
    window_norm    = window_norm_df.to_numpy(dtype=np.float32)  # (60, 4)

    print("Sau apply_minmax — per-channel stats (kỳ vọng [0,1]):")
    for i, col in enumerate(FEATURE_COLS):
        ch = window_norm[:, i]
        flag = "  ← NGOÀI [0,1]!" if (ch.min() < -0.01 or ch.max() > 1.01) else ""
        print(f"  {col:20s}  min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}{flag}")

    # Bước 2: transpose giống main.py
    window_chfirst = window_norm.T[np.newaxis, :]   # (1, 4, 60) — như main.py hiện tại
    window_chlast  = window_norm[np.newaxis, :]     # (1, 60, 4) — đúng spec TFLite

    print(f"\nLayout main.py   : window_norm.T[np.newaxis,:] = {window_chfirst.shape}  (channels-first)")
    print(f"Layout đúng spec : window_norm[np.newaxis,:]   = {window_chlast.shape}  (channels-last)")

    # Bước 3: chạy TFLite nếu có
    if not tflite_exists:
        print("\nTFLite không tồn tại → bỏ qua inference TFLite.")
        return None, None

    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tflite_runtime.interpreter import Interpreter

        def _run(interp, data):
            interp.set_tensor(interp.get_input_details()[0]["index"], data)
            interp.invoke()
            return float(np.squeeze(interp.get_tensor(interp.get_output_details()[0]["index"])))

        interp = Interpreter(model_path=str(TFLITE_PATH))
        interp.allocate_tensors()

        # Thử layout main.py: (1, 4, 60)
        try:
            raw_chfirst = _run(interp, window_chfirst)
            print(f"\nsoc_tflite     (1,4,60) như main.py : raw={raw_chfirst:.4f}")
        except Exception as exc:
            raw_chfirst = None
            print(f"\nsoc_tflite     (1,4,60) như main.py : ERROR → {exc}")

        # Thử layout đúng spec: (1, 60, 4)
        try:
            raw_chlast = _run(interp, window_chlast)
            print(f"soc_tflite_alt (1,60,4) đúng spec   : raw={raw_chlast:.4f}")
        except Exception as exc:
            raw_chlast = None
            print(f"soc_tflite_alt (1,60,4) đúng spec   : ERROR → {exc}")

        return raw_chfirst, raw_chlast

    except Exception as exc:
        print(f"[Lỗi khởi tạo TFLite interpreter]: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Phần 5 — Chạy PyTorch model
# ---------------------------------------------------------------------------

def run_pytorch(window_raw: "pd.DataFrame", ckpt: dict) -> float:
    """Chạy CNN1D PyTorch trực tiếp, dùng feature_norm từ .pt."""
    _sep("PHẦN 5 — PyTorch model")

    if ckpt is None:
        print("Không có checkpoint → bỏ qua.")
        return float("nan")

    norm_pt    = ckpt.get("feature_norm", {})
    label_scale = float(ckpt.get("label_scale", 1.0))

    # Normalize bằng feature_norm từ .pt (có thể khác yaml)
    window_norm_pt = apply_minmax(window_raw, norm_pt).to_numpy(dtype=np.float32)

    # Input shape cho model: (1, 60, 4) — forward() tự permute
    x = torch.tensor(window_norm_pt, dtype=torch.float32).unsqueeze(0)

    model = CNN1D(in_channels=4)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        out = model(x)

    raw = float(out.squeeze().item())
    soc = raw * label_scale
    soc = float(np.clip(soc, 0.0, 100.0))

    print(f"Input shape  : {x.shape}  (→ permuted (1,4,60) trong forward())")
    print(f"Raw output   : {raw:.6f}")
    print(f"label_scale  : {label_scale}")
    print(f"soc_pytorch  : {soc:.2f}%")

    return soc


# ---------------------------------------------------------------------------
# Phần 6 — Bảng tổng kết
# ---------------------------------------------------------------------------

def print_summary(soc_truth, soc_pytorch, raw_chfirst, raw_chlast,
                  label_scale_yaml, tflite_exists):
    """In bảng tổng kết và kết luận."""
    _sep("PHẦN 6 — TỔNG KẾT")

    def _fmt(val, scale=1.0):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{val * scale:.2f}%"

    print(f"  Ground truth soc_bms (window 10000–10059) : {soc_truth:.1f}%")
    print(f"  soc_pytorch  (.pt, norm từ .pt)           : {_fmt(soc_pytorch)}")
    if tflite_exists:
        print(f"  soc_tflite   (1,4,60) như main.py         : {_fmt(raw_chfirst, label_scale_yaml)}")
        print(f"  soc_tflite_alt (1,60,4) đúng spec         : {_fmt(raw_chlast,  label_scale_yaml)}")
    else:
        print(f"  soc_tflite / soc_tflite_alt               : N/A (file missing)")

    print()
    print("NGUYÊN NHÂN XÁC ĐỊNH:")

    if not tflite_exists:
        print("  ✗ #1 TFLite KHÔNG tồn tại → SocInference demo mode → ~50%")
        print("     Fix: chạy notebooks/03_export_tflite.ipynb")

    print("  ✗ #2 main.py gửi (1,4,60) channels-first,")
    print("       nhưng TFLite expect (1,60,4) channels-last (permute baked in)")
    print("     Fix: đổi line ~386 main.py từ window_norm.T[np.newaxis,:]")
    print("          thành                      window_norm[np.newaxis,:]")

    if not np.isnan(soc_pytorch) and abs(soc_pytorch - soc_truth) < 10:
        print()
        print("  ✓ Model weights (.pt) cho output gần truth → weights đúng,")
        print("    vấn đề nằm ở runtime pipeline (TFLite missing + transpose sai)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tflite_exists             = check_tflite()
    ckpt, norm_yaml, lscale   = load_checkpoint_and_compare_norm()
    window_raw, soc_truth     = load_window()
    raw_cf, raw_cl            = replicate_runtime_pipeline(window_raw, norm_yaml, tflite_exists)
    soc_pytorch               = run_pytorch(window_raw, ckpt)

    print_summary(soc_truth, soc_pytorch, raw_cf, raw_cl, lscale, tflite_exists)
