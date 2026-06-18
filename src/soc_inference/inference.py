"""
Ước lượng SoC #3 bằng CNN1D deep learning model (TensorFlow Lite).

Module này tải một mô hình TFLite được huấn luyện trước và thực hiện
suy diễn trên các cửa sổ dữ liệu được chuẩn bị sẵn từ preprocessing module.

Mô hình nhận input:
  - shape: (batch, window_size, 4)
  - 4 features: pack_voltage_v, pack_current_a, temp_c, speed_kmh (đã normalize)
  - window_size: mặc định 60 sample

Output:
  - soc_model: State of Charge ước lượng (%, 0–100)

Lưu ý:
  - Mô hình chỉ có 1 output (SoC). SoH do SohEstimator đảm nhận riêng.
  - Mô hình yêu cầu input đã được normalize (xem src/preprocessing/normalize.py).
  - Không tự thực hiện preprocessing — caller đảm bảo window đã sẵn sàng.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Cấu hình đường dẫn mô hình (để thay đổi dễ dàng)
_MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "soc_cnn1d.tflite"

# Window size mặc định (phải khớp với mô hình)
DEFAULT_WINDOW_SIZE = 60


class SocInference:
    """
    Ước lượng SoC bằng CNN1D model (TensorFlow Lite).

    Khởi tạo một lần tại startup, rồi gọi predict() mỗi khi có cửa sổ dữ liệu mới.

    Ví dụ sử dụng::

        soc_model = SocInference(model_path="models/soc_cnn1d.tflite")
        ...
        # window: shape (60, 4) — đã normalize
        soc_pred = soc_model.predict(window)   # SoH tách riêng → SohEstimator
    """

    def __init__(self, model_path: str = None, label_scale: float = 1.0):
        """
        Khởi tạo inference engine.

        Args:
            model_path: Đường dẫn đến file .tflite. Nếu None, dùng đường dẫn mặc định.
                       Nếu file không tồn tại, log warning và chạy ở chế độ demo.
            label_scale: Hệ số nhân để chuyển output [0,1] về %. Mặc định 1.0 (an toàn nếu
                        model cũ train không chia label). Truyền 100.0 nếu label đã /100 khi train.

        Raises:
            ImportError: Nếu tflite-runtime không được cài đặt.
        """
        self._label_scale = label_scale

        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                Interpreter = None

        if Interpreter is None:
            logger.warning(
                "ai-edge-litert và tflite-runtime đều chưa cài. "
                "Cài bằng: pip install ai-edge-litert"
            )
            self._interpreter = None
            self._input_details = None
            self._output_details = None
            return

        # Xác định đường dẫn mô hình
        if model_path is None:
            model_path = _MODEL_PATH

        model_path = Path(model_path)

        if not model_path.exists():
            abs_path = model_path.resolve()
            logger.warning(
                "⚠️ CNN model KHÔNG tìm thấy tại %s — SoC #3 đang là GIẢ "
                "(demo mode), KHÔNG phản ánh pin thật.", abs_path
            )
            self._interpreter = None
            self._input_details = None
            self._output_details = None
            self.demo_mode = True
            return

        # Tải mô hình
        try:
            self._interpreter = Interpreter(model_path=str(model_path))
            self._interpreter.allocate_tensors()

            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

            logger.info(
                f"Tải mô hình CNN1D thành công từ {model_path}. "
                f"Input shape: {self._input_details[0]['shape']}"
            )
        except Exception as e:
            logger.error(f"Lỗi tải mô hình: {e}")
            self._interpreter = None
            self._input_details = None
            self._output_details = None

    def predict(self, window: np.ndarray) -> float:
        """
        Ước lượng SoC từ một cửa sổ dữ liệu.

        SoH không trả về ở đây — do SohEstimator đảm nhận theo odo.

        Args:
            window: numpy array shape (window_size, 4) hoặc (1, window_size, 4):
                   - [:, 0] = pack_voltage_v (đã normalize)
                   - [:, 1] = pack_current_a (đã normalize)
                   - [:, 2] = temp_c (đã normalize)
                   - [:, 3] = speed_kmh (đã normalize)

        Returns:
            soc_model: SoC ước lượng (%, 0–100).

        Raises:
            ValueError: Nếu window shape không đúng.
        """
        if window.shape[-1] != 4:
            raise ValueError(
                f"Window phải có 4 features (channels-last), nhưng nhận được {window.shape[-1]}"
            )

        # Chế độ demo nếu không có mô hình thực
        if self._interpreter is None:
            logger.debug("Chế độ demo: trả về dummy SoC")
            avg_current = window[:, 1].mean()
            dummy_soc = 50.0 + avg_current * 5.0  # Heuristic đơn giản
            return float(np.clip(dummy_soc, 0.0, 100.0))

        # Chuẩn bị input: thêm batch dimension nếu cần
        input_data = window.astype(np.float32)
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)  # (1, window_size, 4)

        # Chạy inference
        self._interpreter.set_tensor(self._input_details[0]["index"], input_data)
        self._interpreter.invoke()

        soc_output = self._interpreter.get_tensor(self._output_details[0]["index"])
        soc_model = float(np.squeeze(soc_output))

        # Chuyển [0,1] → % nếu label đã /100 khi train (label_scale=100.0)
        soc_model = soc_model * self._label_scale

        return float(np.clip(soc_model, 0.0, 100.0))

    def get_model_info(self) -> dict:
        """
        Lấy thông tin về mô hình được tải.

        Returns:
            Dict chứa: input_shape, output_shapes, num_tensors, ...
        """
        if self._interpreter is None:
            return {"status": "demo_mode", "message": "Không có mô hình thực"}

        info = {
            "input_shape": tuple(self._input_details[0]["shape"]),
            "num_outputs": len(self._output_details),
            "output_shapes": [
                tuple(out["shape"]) for out in self._output_details
            ],
        }
        return info
