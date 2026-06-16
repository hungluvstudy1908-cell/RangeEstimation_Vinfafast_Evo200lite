import serial

def decode_waveshare():
    try:
        # Giữ nguyên tốc độ 2 triệu bps
        ser = serial.Serial('/dev/ttyUSB0', 2000000, timeout=0.1)
        print("🚀 Đã bóc tách thành công! Đang hiển thị CAN ID và Data...")
        print("-" * 60)
        
        buffer = bytearray()
        
        while True:
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)
                
                # Chờ buffer gom đủ 20 byte
                while len(buffer) >= 20:
                    if buffer[0] == 0xAA and buffer[1] == 0x55:
                        frame = buffer[:20]
                        del buffer[:20] 
                        
                        # BÓC TÁCH CAN ID (Từ byte 4 đến 7, đọc ngược chiều Little-Endian)
                        can_id_hex = f"{frame[7]:02X}{frame[6]:02X}{frame[5]:02X}{frame[4]:02X}"
                        
                        # Cắt bỏ các số 0 vô nghĩa ở đầu cho dễ nhìn
                        can_id = can_id_hex.lstrip('0') 
                        if can_id == '': 
                            can_id = '00'
                            
                        # BÓC TÁCH DATA (Từ byte 8 đến 15)
                        data_hex = " ".join([f"{b:02X}" for b in frame[8:16]])
                        
                        # In ra màn hình thẳng hàng
                        print(f"🆔 ID: 0x{can_id:<4}  |  📊 Data: {data_hex}")
                        
                    else:
                        buffer.pop(0)
                        
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng theo dõi.")
    except Exception as e:
        print(f"\n❌ Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    decode_waveshare()