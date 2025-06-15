import os
import cv2
import numpy as np

# Directory where data will be saved
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Class labels for gestures
gesture_labels = ['D', 'W', 'B']  # D = Deposit, W = Withdraw, B = Balance Check
dataset_size = 300

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera was opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Parameters for the hand placement box (from second code)
box_x, box_y, box_w, box_h = 300, 100, 300, 300  # Position and size of hand placement box
box_color = (0, 255, 0)  # Green color for the box
box_thickness = 2

def draw_hand_box(img):
    """Draw the hand placement box and instructions on the frame"""
    # Draw main hand placement box
    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, box_thickness)
    
    # Draw instruction text
    cv2.putText(img, 'Place hand inside green box', 
                (box_x - 50, box_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img

# Loop through each gesture class
for label in gesture_labels:
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for gesture: {label}')

    # Show instruction screen
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = draw_hand_box(frame)
        
        # Add gesture-specific instructions
        cv2.putText(frame, f'Get ready for gesture: {label}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, 'Press "Q" to start collecting data!', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Start data collection for this gesture
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Draw hand box on each frame
        frame = draw_hand_box(frame)
        
        # Crop only the hand region for saving (from the box area)
        hand_roi = frame[box_y:box_y+box_h, box_x:box_x+box_w]
        
        # Save only the hand region instead of full frame
        file_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(file_path, hand_roi)  # Save just the hand region
        print(f'Saved {file_path}')
        counter += 1

        # Display collection progress
        cv2.putText(frame, f'Collecting {label}: {counter}/{dataset_size}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Show the frame with hand box
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            print('Stopping data collection early.')
            break

cap.release()
cv2.destroyAllWindows()