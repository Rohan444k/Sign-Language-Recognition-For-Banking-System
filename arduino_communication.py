import serial
import time

# Set the correct COM port and baud rate
arduino_port = 'COM4'  # Update this if needed
baud_rate = 9600

try:
    # Try opening the serial connection
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)

    while True:
        predicted_letter = 'A'  # Replace with your predicted value

        # Sending letter to Arduino
        ser.write(predicted_letter.encode('utf-8'))
        time.sleep(1)

except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("Program terminated by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
    print("Serial connection closed.")
