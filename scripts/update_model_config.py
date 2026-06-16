"""
Đọc feature_norm + label_scale từ models/soc_cnn1d.pt
và ghi vào configs/model.yaml.

Chạy sau khi train xong, thay thế cho bước D trong 03_export_tflite.ipynb
khi tensorflow/onnx chưa được cài.

Usage:
    python scripts/update_model_config.py
"""

from pathlib import Path
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
PT_MODEL     = PROJECT_ROOT / "models" / "soc_cnn1d.pt"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "model.yaml"


def main():
    # Load checkpoint
    ckpt = torch.load(str(PT_MODEL), map_location="cpu", weights_only=False)
    feature_norm = ckpt["feature_norm"]
    label_scale  = float(ckpt["label_scale"])

    print(f"Loaded from {PT_MODEL}")
    print(f"feature_norm:")
    for col, v in feature_norm.items():
        print(f"  {col}: min={v['min']:.4f}  max={v['max']:.4f}")
    print(f"label_scale: {label_scale}")

    # Merge vào configs/model.yaml
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["normalization"] = feature_norm
    cfg["label_scale"]   = label_scale

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"\nWritten to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
