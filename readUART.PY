import serial

try:
    ser = serial.Serial(
        port='COM4',  # Replace with your serial port (e.g., COM1 on Windows)
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1  # Timeout in seconds for read operations
    )
    print(f"Serial port {ser.name} opened successfully.")
    while True:
        data = ser.readline().decode('utf-8').strip()  # Read a line and decode to string
        if data:
            print(f"Received: {data}")
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    if ser.is_open:
        ser.close()
        print(f"Serial port {ser.name} closed.")