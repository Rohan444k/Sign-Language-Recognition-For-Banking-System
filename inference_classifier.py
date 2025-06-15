import cv2
import numpy as np
import mediapipe as mp
import serial
import time
from collections import deque
import pyttsx3
import threading
from pathlib import Path
from keras.models import load_model

# Initialize text-to-speech
engine = pyttsx3.init()

def speak_text(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def preprocess_landmarks(landmark_list):
    """Convert 42-length flat list into 7x6x1 numpy array"""
    if len(landmark_list) != 42:
        return None
    reshaped = np.array(landmark_list).reshape((7, 6, 1))
    return reshaped

def main():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / 'gesture_cnn_final.keras'

    if not model_path.exists():
        print(f"Error: CNN model file not found at {model_path}")
        return

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.75, min_tracking_confidence=0.75)
    
    labels_dict = {0: 'D', 1: 'W', 2: 'B'}
    label_to_command = {'D': 'Deposit', 'W': 'Withdraw', 'B': 'Balance Check'}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    # Arduino Serial Setup
    arduino_port = 'COM5'  # Change if needed
    baud_rate = 9600
    ser = None

    try:
        ser = serial.Serial(arduino_port, baud_rate, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Warning: Could not connect to Arduino - {e}")
        ser = None

    predictions_queue = deque(maxlen=5)
    last_spoken_command = ""
    frame_counter = 0
    process_interval = 3
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1/fps if fps > 0 else 0.033

    # Gesture hold timer variables
    gesture_hold_time = 0
    current_gesture = None
    hold_threshold = 1.5
    command_sent = False
    last_prediction_time = time.time()
    last_command_display_time = 0
    command_display_duration = 3

    # Define detection box parameters
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    box_size = int(min(frame_width, frame_height) * 0.6)
    box_x1 = (frame_width - box_size) // 2
    box_y1 = (frame_height - box_size) // 2
    box_x2 = box_x1 + box_size
    box_y2 = box_y1 + box_size

    # Confidence threshold
    confidence_threshold = 0.85

    # UI Colors
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_PURPLE = (255, 0, 255)
    BG_COLOR = (50, 50, 50)

    # Initialize results variable
    results = None
    hand_in_box = False
    box_color = COLOR_GREEN
    current_command = ""
    system_active = True

    try:
        while system_active:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed.")
                break

            current_time = time.time()
            
            # Create UI frame
            ui_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            ui_frame[:] = BG_COLOR
            ui_frame = cv2.addWeighted(frame, 0.7, ui_frame, 0.3, 0)

            # Draw header
            cv2.putText(ui_frame, "Banking Gesture Control", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)
            
            # Process frame for hand detection
            frame_counter += 1
            if frame_counter % process_interval == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                hand_in_box = False

                if results and results.multi_hand_landmarks:  # Check if results exists
                    for hand_landmarks in results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                        wrist_x = int(wrist.x * frame_width)
                        wrist_y = int(wrist.y * frame_height)
                        
                        if (box_x1 < wrist_x < box_x2 and box_y1 < wrist_y < box_y2):
                            hand_in_box = True
                            box_color = COLOR_GREEN
                            
                            landmark_list = []
                            for lm in hand_landmarks.landmark:
                                landmark_list.extend([lm.x, lm.y])

                            features = preprocess_landmarks(landmark_list)
                            if features is not None:
                                try:
                                    features = np.expand_dims(features, axis=0)
                                    prediction = model.predict(features, verbose=0)
                                    predicted_class = np.argmax(prediction)
                                    confidence = np.max(prediction)
                                    
                                    if confidence >= confidence_threshold:
                                        predicted_label = labels_dict.get(predicted_class, 'Unknown')
                                        predictions_queue.append(predicted_label)
                                        
                                        if predicted_label == current_gesture:
                                            gesture_hold_time += frame_interval * process_interval
                                            if gesture_hold_time >= hold_threshold and not command_sent:
                                                if predictions_queue:
                                                    most_common = max(set(predictions_queue), key=predictions_queue.count)
                                                    current_command = label_to_command.get(most_common, '')
                                                    
                                                    if ser and current_command:
                                                        try:
                                                            ser.write((current_command + '\n').encode('utf-8'))
                                                        except Exception as e:
                                                            print(f"Serial error: {e}")
                                                            ser = None

                                                    if current_command and current_command != last_spoken_command:
                                                        threading.Thread(target=speak_text, args=(current_command,)).start()
                                                        last_spoken_command = current_command
                                                        command_sent = True
                                                        last_command_display_time = current_time
                                        else:
                                            current_gesture = predicted_label
                                            gesture_hold_time = 0
                                            command_sent = False
                                    else:
                                        cv2.putText(ui_frame, f"Low confidence: {confidence:.2f}", 
                                                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                                                   0.7, COLOR_RED, 2)
                                except Exception as e:
                                    print(f"Prediction error: {e}")
                        else:
                            box_color = COLOR_RED
                            current_gesture = None
                            gesture_hold_time = 0
                            command_sent = False
                else:
                    current_gesture = None
                    gesture_hold_time = 0
                    command_sent = False
                    box_color = COLOR_RED

            # Draw detection box
            cv2.rectangle(ui_frame, (box_x1, box_y1), (box_x2, box_y2), box_color, 2)
            cv2.putText(ui_frame, "Place hand here", (box_x1, box_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Draw hand landmarks if available
            if results and results.multi_hand_landmarks:  # Check if results exists
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_color = COLOR_GREEN if hand_in_box else COLOR_RED
                    connection_color = COLOR_BLUE if hand_in_box else COLOR_RED
                    mp.solutions.drawing_utils.draw_landmarks(
                        ui_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=landmark_color, thickness=3, circle_radius=3),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=connection_color, thickness=2))

            # Display gesture hold progress
            if current_gesture and gesture_hold_time > 0:
                progress = min(gesture_hold_time / hold_threshold, 1.0)
                cv2.rectangle(ui_frame, (box_x1, box_y2 + 10), 
                             (box_x2, box_y2 + 25), COLOR_WHITE, 1)
                cv2.rectangle(ui_frame, (box_x1, box_y2 + 10), 
                             (int(box_x1 + box_size * progress), box_y2 + 25), 
                             COLOR_GREEN, -1)
                cv2.putText(ui_frame, f"Holding: {current_gesture} ({gesture_hold_time:.1f}s)", 
                          (box_x1, box_y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, COLOR_YELLOW, 1)

            # Display command
            if current_time - last_command_display_time < command_display_duration and last_spoken_command:
                cv2.putText(ui_frame, f"Command: {last_spoken_command}", 
                          (frame_width - 350, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, COLOR_GREEN, 2)
                cv2.putText(ui_frame, "✓", (frame_width - 400, 80), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_GREEN, 3)
                cv2.rectangle(ui_frame, (frame_width - 420, 30), 
                            (frame_width - 20, 100), COLOR_GREEN, 2)

            # Display system status
            status_text = f"Status: {'Active' if hand_in_box else 'Waiting for hand'}"
            status_color = COLOR_GREEN if hand_in_box else COLOR_RED
            cv2.putText(ui_frame, status_text, (20, frame_height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Display help text
            help_text = "Gestures: D=Deposit, W=Withdraw, B=Balance | Q=Quit"
            cv2.putText(ui_frame, help_text, (20, frame_height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

            cv2.imshow('Banking Gesture Control', ui_frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                system_active = False

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        if ser:
            ser.close()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    main()