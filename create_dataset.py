import os
import pickle
import random
import cv2
import numpy as np
import mediapipe as mp
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)  # Fixed typo: min_detection_confidence

# Define the directory for storing data
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []
labels = []

# Map directory names to gesture labels
label_map = {'D': 'D', 'W': 'W', 'B': 'B'}

def augment_image(img):
    """Generate augmented versions of the input image"""
    augmented_images = []
    
    # Original image
    augmented_images.append(img)
    
    # Random rotation
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    augmented_images.append(rotated)
    
    # Random brightness adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[...,2] = hsv[...,2] * random.uniform(0.7, 1.3)
    bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    augmented_images.append(bright)
    
    # Flip horizontally
    flipped = cv2.flip(img, 1)
    augmented_images.append(flipped)
    
    return augmented_images

# Process each class directory (D, W, B)
for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)

    # Skip if not a directory or not in label_map
    if not os.path.isdir(class_dir) or dir_ not in label_map:
        continue

    print(f"\nProcessing {len(os.listdir(class_dir))} images for class {dir_}")
    processed = 0

    for img_path in os.listdir(class_dir):
        img_path_full = os.path.join(class_dir, img_path)

        # Read the image
        img = cv2.imread(img_path_full)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Warning: Could not read image {img_path_full}. Skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image using MediaPipe
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Visualize landmarks (optional)
                mp.solutions.drawing_utils.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('Landmarks', img)
                cv2.waitKey(10)  # Reduced delay for smoother visualization
                
                # Generate augmented versions
                for augmented_img in augment_image(img_rgb):
                    # Process each augmented version
                    augmented_results = hands.process(augmented_img)
                    
                    if augmented_results.multi_hand_landmarks:
                        for aug_landmarks in augmented_results.multi_hand_landmarks:
                            landmark_list = []
                            for lm in aug_landmarks.landmark:
                                landmark_list.append(lm.x)
                                landmark_list.append(lm.y)

                            # Ensure we have the correct number of features
                            if len(landmark_list) == 42:  # 21 landmarks * 2 (x, y)
                                data.append(landmark_list)
                                labels.append(label_map[dir_])
                                
                processed += 1
                if processed % 50 == 0:
                    print(f"Processed {processed} images for {dir_}")

# Save data and labels
with open('gesture_dataset.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("\nDataset created successfully!")
print(f"Total samples collected: {len(data)}")
print(f"Class distribution: {np.unique(labels, return_counts=True)}")

cv2.destroyAllWindows()