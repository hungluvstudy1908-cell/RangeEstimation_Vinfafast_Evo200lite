"""
Đọc và ghép khung CAN từ mạch chuyển đổi Waveshare USB-CAN.

Module này chịu trách nhiệm DUY NHẤT cho lớp vật lý:
  - Mở kết nối serial đến Waveshare (/dev/ttyUSB0).
  - Đệm (buffer) byte thô và phát hiện đầu khung bằng byte đồng bộ 0xAA 0x55.
  - Tách khung 20 byte hoàn chỉnh và trả về (can_id, data).
  - Tự động kết nối lại khi mất cáp hoặc lỗi.

Không xử lý ý nghĩa của tín hiệu — việc đó thuộc decoder.py.

Cấu trúc khung Waveshare (20 byte):
  Byte 0-1  : Đồng bộ (0xAA, 0x55)
  Byte 2-4  : Header (3 byte, bỏ qua)
  Byte 5-8  : CAN ID (4 byte, Little-Endian)
  Byte 9    : DLC / flags (1 byte, bỏ qua)
  Byte 10-17: DATA payload (8 byte)
  Byte 18-19: Trailing (2 byte, bỏ qua)
"""

import logging
import time

import serial

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số giao thức Waveshare
# ---------------------------------------------------------------------------
_SYNC_BYTE_0   = 0xAA   # Byte đồng bộ đầu tiên
_SYNC_BYTE_1   = 0x55   # Byte đồng bộ thứ hai
_FRAME_SIZE    = 20     # Kích thước khung cố định (byte)
_CAN_ID_OFFSET = 5      # Vị trí bắt đầu CAN ID trong khung
_DATA_OFFSET   = 10     # Vị trí bắt đầu DATA trong khung
_DATA_LENGTH   = 8      # Độ dài DATA (byte)

# ---------------------------------------------------------------------------
# Cấu hình serial mặc định — khớp với Waveshare USB-CAN
# ---------------------------------------------------------------------------
DEFAULT_PORT     = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 2_000_000   # 2 Mbps — tốc độ mặc định của Waveshare


class WaveshareReader:
    """
    Đọc khung CAN liên tục từ mạch Waveshare USB-CAN qua cổng serial.

    Cách dùng trong main loop (10Hz):
        reader = WaveshareReader()
        reader.connect()
        while True:
            for can_id, data in reader.read_frames():
                decoded = decoder.decode(can_id, data)
                ...
    """

    def __init__(self, port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUDRATE):
        """
        Khởi tạo reader với thông số cổng serial.

        Args:
            port: Đường dẫn cổng serial, mặc định '/dev/ttyUSB0' trên Pi.
            baudrate: Tốc độ baud, mặc định 2,000,000 bps (Waveshare).
        """
        self.port     = port
        self.baudrate = baudrate
        self._ser     = None
        self._buffer  = bytearray()

    def connect(self) -> None:
        """
        Mở kết nối serial đến Waveshare.

        Raises:
            serial.SerialException: Nếu không thể mở cổng.
        """
        self._ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
        self._buffer.clear()
        logger.info("Đã kết nối Waveshare tại %s (%d bps)", self.port, self.baudrate)

    def disconnect(self) -> None:
        """Đóng kết nối serial."""
        if self._ser and self._ser.is_open:
            self._ser.close()
            logger.info("Đã ngắt kết nối Waveshare")

    def read_frames(self) -> list[tuple[int, bytes]]:
        """
        Đọc tất cả khung hoàn chỉnh đang có trong buffer serial.

        Gọi hàm này mỗi tick (10Hz). Mỗi lần gọi xử lý hết byte
        đang chờ trong buffer, trả về danh sách các khung đã ghép được.
        Khung chưa đủ 20 byte được giữ lại cho lần gọi tiếp theo.

        Returns:
            Danh sách các tuple (can_id: int, data: bytes).
            Trả về list rỗng nếu không có dữ liệu mới hoặc chưa đủ khung.

        Raises:
            serial.SerialException: Nếu mất kết nối serial.
        """
        if self._ser is None or not self._ser.is_open:
            raise serial.SerialException("Chưa kết nối serial — gọi connect() trước.")

        frames = []

        # Đọc tất cả byte đang chờ trong OS buffer của serial
        waiting = self._ser.in_waiting
        if waiting > 0:
            self._buffer += self._ser.read(waiting)

        # Tách khung từ buffer cho đến khi hết
        while len(self._buffer) >= _FRAME_SIZE:
            # Kiểm tra byte đồng bộ đầu khung
            if self._buffer[0] == _SYNC_BYTE_0 and self._buffer[1] == _SYNC_BYTE_1:
                frame = bytes(self._buffer[:_FRAME_SIZE])
                del self._buffer[:_FRAME_SIZE]
                frames.append(_parse_frame(frame))
            else:
                # Byte đầu không đúng sync → bỏ đi 1 byte, tìm sync tiếp theo
                logger.debug("Bỏ byte lệch khung: 0x%02X", self._buffer[0])
                del self._buffer[0]

        return frames


def connect_with_retry(
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    retry_delay_s: float = 2.0,
) -> WaveshareReader:
    """
    Tạo WaveshareReader và thử kết nối liên tục cho đến khi thành công.

    Dùng khi khởi động Pi — đợi Waveshare được cắm vào trước khi tiếp tục.

    Args:
        port: Cổng serial.
        baudrate: Tốc độ baud.
        retry_delay_s: Thời gian chờ giữa các lần thử (giây).

    Returns:
        WaveshareReader đã kết nối thành công.
    """
    reader = WaveshareReader(port=port, baudrate=baudrate)
    while True:
        try:
            reader.connect()
            return reader
        except serial.SerialException as e:
            logger.warning("Chưa kết nối được Waveshare: %s — thử lại sau %.1fs", e, retry_delay_s)
            time.sleep(retry_delay_s)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _parse_frame(frame: bytes) -> tuple[int, bytes]:
    """
    Tách CAN ID và DATA từ khung 20 byte đã xác nhận đồng bộ.

    Cấu trúc: xem docstring module.
    CAN ID đọc theo Little-Endian (byte thấp trước).

    Args:
        frame: Bytes khung đủ 20 byte, đã xác nhận có sync 0xAA 0x55.

    Returns:
        Tuple (can_id: int, data: bytes) trong đó data dài 8 byte.
    """
    # CAN ID: 4 byte Little-Endian tại offset 5
    can_id = (
        frame[_CAN_ID_OFFSET]
        | (frame[_CAN_ID_OFFSET + 1] << 8)
        | (frame[_CAN_ID_OFFSET + 2] << 16)
        | (frame[_CAN_ID_OFFSET + 3] << 24)
    )

    data = frame[_DATA_OFFSET: _DATA_OFFSET + _DATA_LENGTH]

    return can_id, data
