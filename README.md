# Hand Gesture Recognition & Arduino Communication System

This project enables real-time hand gesture recognition using computer vision and machine learning, then sends the recognized gestures to an Arduino board to perform various hardware actions.

## 📁 Project Structure

- `collect_imgs.py` — Collects image data for each hand gesture using webcam.
- `create_dataset.py` — Creates a dataset from the collected images for model training.
- `train_classifier.py` — Trains a CNN model using Keras for gesture classification.
- `inference_classifier.py` — Performs real-time gesture classification using webcam.
- `sent_to_arduino.py` — Sends recognized characters to an Arduino over serial communication.
- `arduino_communication.py` — Handles Arduino communication (for test/demo).

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt

2. Collect Data
Run:
python collect_imgs.py

3. Create Dataset
python create_dataset.py

4. Train Model
python train_classifier.py

5. Run Inference (Prediction)
python inference_classifier.py

6. Send to Arduino
python sent_to_arduino.py

Model:- 
CNN with 3 Conv layers + Dense layers

Trained on a 7x6 reshaped landmark grid from MediaPipe

Saved as gesture_cnn_final.keras

Requirements:- 
Use the provided requirements.txt to install all libraries.

pip install -r requirements.txt

Arduino Configuration:-
Baud Rate: 9600

Serial Port: Update 'COM4' in sent_to_arduino.py and arduino_communication.py based on your OS

Libraries Used:- 
opencv-python

mediapipe

numpy

keras

tensorflow

scikit-learn

pyserial

Notes:-
Ensure the webcam is accessible

Arduino should be connected via USB and available at the correct COM port
