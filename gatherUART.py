import serial
import csv

try:
    ser = serial.Serial(
        port='COM4',
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1  # Timeout in seconds for read operations
    )
    print(f"Serial port {ser.name} opened successfully.")
    with open('imu_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z'])  # Header
        while True:
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