import serial
import csv
import time
from datetime import datetime

now = datetime.now()
# timestamp_format = "%Y-%m-%d-%H_%M_%S" 
timestamp_format = "%Y-%m-%d"
formatted_timestamp = now.strftime(timestamp_format)

path = f"data/{formatted_timestamp}_imu_data.csv"
try:
    ser = serial.Serial(
        port='COM4',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1  # Timeout in seconds for read operations
    )
    print(f"Serial port {ser.name} opened successfully.")
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])  # Header
        start_time = time.time()
        while (time.time() - start_time) < 10:
            ser.reset_input_buffer()
            data = ser.readline().decode('utf-8').strip()  # Read a line and decode to string
            if data:
                print(f"{data}")
                fields = [p.strip() for p in data.split(',')]
                writer.writerow(fields)
                file.flush()
    
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    if ser.is_open:
        ser.close()
        print(f"Serial port {ser.name} closed.")