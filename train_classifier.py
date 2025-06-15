import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import os

# Set up environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
K.clear_session()

# Load and prepare the dataset
def load_data():
    with open('gesture_dataset.pickle', 'rb') as f:
        data_dict = pickle.load(f)
    
    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])
    
    # Convert labels to numerical format
    label_map = {'D': 0, 'W': 1, 'B': 2}
    y = np.array([label_map[label] for label in labels])
    
    # Reshape data for CNN (assuming landmarks can be represented as 1D "image")
    # We'll use 7x6 grid (42 landmarks) with 1 channel
    X = data.reshape(-1, 7, 6, 1)  # 7 rows, 6 columns, 1 channel
    
    return X, y, len(label_map)

# CNN Model Architecture
def create_model(num_classes):
    model = Sequential()
    
    # Convolutional layers
    model.add(Conv2D(32, (2, 2), input_shape=(7, 6, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    
    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    
    model.add(Conv2D(128, (2, 2), activation='relu'))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', 
                 optimizer=sgd, 
                 metrics=['accuracy'])
    
    return model

def main():
    # Load and prepare data
    X, y, num_classes = load_data()
    
    # Convert labels to one-hot encoding
    y = to_categorical(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create model
    model = create_model(num_classes)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint('best_gesture_cnn.h5', 
                               monitor='val_accuracy',
                               save_best_only=True,
                               mode='max',
                               verbose=1)
    
    # Train model
    history = model.fit(X_train, y_train,
                       validation_data=(X_test, y_test),
                       epochs=50,
                       batch_size=32,
                       callbacks=[checkpoint])
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Save final model
    model.save('gesture_cnn_final.keras')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()