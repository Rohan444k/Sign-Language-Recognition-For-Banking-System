import serial
import time

# Replace 'COM3' with the correct port for your Arduino
try:
    arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit()

def send_data(data):
    # Send data to Arduino
    arduino.write(data.encode())
    time.sleep(0.05)  # Small delay to ensure the data is sent

print("Enter 'A', 'B', or 'L' to send data to Arduino. Type 'exit' to quit.")

while True:
    user_input = input("Input: ").strip().upper()
    if user_input in ['A', 'B', 'L']:
        send_data(user_input)
    elif user_input == 'EXIT':
        print("Exiting...")
        break
    else:
        print("Invalid input, only 'A', 'B', or 'L' are allowed.")
