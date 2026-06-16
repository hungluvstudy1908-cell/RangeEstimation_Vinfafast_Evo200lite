"""
Export CNN1D PyTorch model sang TFLite cho Raspberry Pi 4 runtime.

Cách chạy từ thư mục gốc dự án:
    python scripts/export_tflite.py

Yêu cầu:
    pip install tensorflow ai-edge-litert
"""

import sys
from pathlib import Path

# Thêm thư mục gốc vào path để import src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

# Kiểm tra dependencies nặng trước khi import — không tự cài
try:
    import tensorflow as tf
except ImportError:
    print("DỪNG: tensorflow chưa cài — pip install tensorflow")
    sys.exit(1)

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    print("DỪNG: ai-edge-litert chưa cài — pip install ai-edge-litert")
    sys.exit(1)

MODEL_DIR    = PROJECT_ROOT / "models"
PT_MODEL     = MODEL_DIR / "soc_cnn1d.pt"
TFLITE_MODEL = MODEL_DIR / "soc_cnn1d.tflite"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "model.yaml"

print(f"tensorflow {tf.__version__}")
print(f"PyTorch checkpoint : {PT_MODEL}")
print(f"Output TFLite      : {TFLITE_MODEL}\n")


# ---------------------------------------------------------------------------
# Bước 1 — Load PyTorch model
# Copy y hệt class CNN1D từ notebook training để đảm bảo khớp architecture
# ---------------------------------------------------------------------------

class CNN1D(nn.Module):
    """CNN1D dùng để train soc_cnn1d.pt."""

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
        # Input (batch, 60, 4) channels-last -> permute sang channels-first cho Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


# weights_only=False vì checkpoint lưu numpy scalars (feature_norm, best_mae)
# File nội bộ từ notebook training cùng project — nguồn tin cậy.
checkpoint = torch.load(str(PT_MODEL), map_location="cpu", weights_only=False)
model = CNN1D(in_channels=4)
model.load_state_dict(checkpoint["model_state"])
model.eval()
print(f"Loaded PyTorch checkpoint  best_mae={checkpoint['best_mae']:.6f}")


# ---------------------------------------------------------------------------
# Bước 2 — Đồng bộ feature_norm + label_scale -> configs/model.yaml
# ---------------------------------------------------------------------------

feature_norm = checkpoint["feature_norm"]
label_scale  = float(checkpoint["label_scale"])

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["normalization"] = feature_norm
cfg["label_scale"]   = label_scale

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

print(f"Synced feature_norm + label_scale -> {CONFIG_PATH.name}")


# ---------------------------------------------------------------------------
# Bước 3 — Build Keras model tương đương + transfer weights
# Input (60, 4) channels-last — Keras Conv1D dùng NLC, không cần permute
# epsilon=1e-5 khớp PyTorch BatchNorm1d mặc định
# ---------------------------------------------------------------------------

inp = tf.keras.Input(shape=(60, 4), dtype=tf.float32, name="input")

x = tf.keras.layers.Conv1D(64, 3, padding="same", name="conv1_conv")(inp)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name="conv1_bn")(x)
x = tf.keras.layers.ReLU(name="conv1_relu")(x)

x = tf.keras.layers.Conv1D(128, 3, padding="same", name="conv2_conv")(x)
x = tf.keras.layers.BatchNormalization(epsilon=1e-5, name="conv2_bn")(x)
x = tf.keras.layers.ReLU(name="conv2_relu")(x)

x = tf.keras.layers.GlobalAveragePooling1D(name="pool")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu", name="fc1")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, name="fc2")(x)

keras_model = tf.keras.Model(inputs=inp, outputs=x)

# Chuyển weights: Conv1d (out,in,k)->(k,in,out), Linear (out,in)->(in,out), BatchNorm giữ nguyên
sd = model.state_dict()

keras_model.get_layer("conv1_conv").set_weights([
    sd["conv1.0.weight"].numpy().transpose(2, 1, 0),
    sd["conv1.0.bias"].numpy(),
])
keras_model.get_layer("conv1_bn").set_weights([
    sd["conv1.1.weight"].numpy(),
    sd["conv1.1.bias"].numpy(),
    sd["conv1.1.running_mean"].numpy(),
    sd["conv1.1.running_var"].numpy(),
])
keras_model.get_layer("conv2_conv").set_weights([
    sd["conv2.0.weight"].numpy().transpose(2, 1, 0),
    sd["conv2.0.bias"].numpy(),
])
keras_model.get_layer("conv2_bn").set_weights([
    sd["conv2.1.weight"].numpy(),
    sd["conv2.1.bias"].numpy(),
    sd["conv2.1.running_mean"].numpy(),
    sd["conv2.1.running_var"].numpy(),
])
keras_model.get_layer("fc1").set_weights([
    sd["fc.1.weight"].numpy().T,
    sd["fc.1.bias"].numpy(),
])
keras_model.get_layer("fc2").set_weights([
    sd["fc.4.weight"].numpy().T,
    sd["fc.4.bias"].numpy(),
])

# Sanity check: PyTorch ≈ Keras trên random input
_x = np.random.randn(1, 60, 4).astype(np.float32)
with torch.no_grad():
    pt_out = float(model(torch.FloatTensor(_x)).squeeze())
keras_out = float(keras_model(_x, training=False).numpy().squeeze())
diff_pt_keras = abs(pt_out - keras_out)
print(f"PyTorch: {pt_out:.6f}  Keras: {keras_out:.6f}  diff: {diff_pt_keras:.2e}")
assert diff_pt_keras < 1e-3, f"Weight transfer thất bại (diff={diff_pt_keras:.4f})"
print("Weight transfer OK OK")


# ---------------------------------------------------------------------------
# Bước 4 — Keras -> TFLite
# ---------------------------------------------------------------------------

converter   = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_bytes = converter.convert()

with open(str(TFLITE_MODEL), "wb") as f:
    f.write(tflite_bytes)

print(f"\nExported -> {TFLITE_MODEL}  ({TFLITE_MODEL.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Bước 5 — Test TFLite với ai_edge_litert
# ---------------------------------------------------------------------------

interpreter = Interpreter(model_path=str(TFLITE_MODEL))
interpreter.allocate_tensors()

inp_det = interpreter.get_input_details()
out_det = interpreter.get_output_details()

print(f"TFLite input  shape={inp_det[0]['shape']}  dtype={inp_det[0]['dtype'].__name__}")
print(f"TFLite output shape={out_det[0]['shape']}")

test_np = np.random.randn(1, 60, 4).astype(np.float32)
interpreter.set_tensor(inp_det[0]["index"], test_np)
interpreter.invoke()
tflite_out = interpreter.get_tensor(out_det[0]["index"])

with torch.no_grad():
    pt_ref = model(torch.FloatTensor(test_np)).numpy()

diff_tflite = float(np.abs(tflite_out - pt_ref).max())
print(f"PyTorch: {pt_ref[0,0]:.6f}  TFLite: {tflite_out[0,0]:.6f}  diff: {diff_tflite:.2e}")
assert diff_tflite < 1e-4, f"TFLite diff quá lớn: {diff_tflite:.6f}"
print("TFLite parity OK OK")
