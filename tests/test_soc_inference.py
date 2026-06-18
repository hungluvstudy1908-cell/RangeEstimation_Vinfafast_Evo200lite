
"""
Test SoC inference module.

Smoke tests để kiểm tra:
- Module có thể import
- SocInference khởi tạo được (chế độ demo nếu không có model)
- predict() nhận đúng shape và trả về giá trị hợp lý
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from src.soc_inference.inference import SocInference


def test_import():
    """Có thể import module."""
    assert SocInference is not None
    print("[OK] Import SocInference")


def test_init_demo_mode():
    """Khởi tạo ở chế độ demo (không có model thực)."""
    soc_inf = SocInference(model_path="/nonexistent/model.tflite")
    assert soc_inf is not None
    print("[OK] Demo mode initialization")


def test_predict_demo_mode():
    """predict() ở chế độ demo trả về dummy SoC hợp lý."""
    soc_inf = SocInference(model_path="/nonexistent/model.tflite")

    # Tạo dummy window (60, 4)
    window = np.random.randn(60, 4).astype(np.float32)

    # Gọi predict — chỉ trả về SoC (SoH tách ra SohEstimator)
    soc = soc_inf.predict(window)

    # Kiểm tra output
    assert isinstance(soc, float)
    assert 0.0 <= soc <= 100.0, f"SoC out of range: {soc}"
    print("[OK] Demo mode prediction works")


def test_predict_shape_validation():
    """predict() validate input shape."""
    soc_inf = SocInference(model_path="/nonexistent/model.tflite")

    # Window với sai số features
    bad_window = np.random.randn(60, 3).astype(np.float32)

    try:
        soc_inf.predict(bad_window)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[OK] Shape validation works")


def test_predict_with_batch_dim():
    """predict() xử lý batch dimension."""
    soc_inf = SocInference(model_path="/nonexistent/model.tflite")

    # Window (60, 4) — không có batch dimension
    window = np.random.randn(60, 4).astype(np.float32)
    soc = soc_inf.predict(window)

    assert 0.0 <= soc <= 100.0
    print("[OK] Batch dimension handling")


def test_model_info():
    """get_model_info() trả về dict hợp lý."""
    soc_inf = SocInference(model_path="/nonexistent/model.tflite")
    info = soc_inf.get_model_info()

    assert isinstance(info, dict)
    assert "status" in info
    print("[OK] Model info retrieval")


def test_predict_bounds():
    """predict() luôn clamp output vào [0, 100]."""
    soc_inf = SocInference(model_path="/nonexistent/model.tflite")

    # Thử các input khác nhau
    for i in range(10):
        window = np.random.randn(60, 4).astype(np.float32)
        soc = soc_inf.predict(window)

        assert 0.0 <= soc <= 100.0

    print("[OK] Bounds clamping (10 iterations)")


if __name__ == "__main__":
    print("Testing SocInference...")
    print()

    test_import()
    test_init_demo_mode()
    test_predict_demo_mode()
    test_predict_shape_validation()
    test_predict_with_batch_dim()
    test_model_info()
    test_predict_bounds()

    print()
    print("All smoke tests passed!")
