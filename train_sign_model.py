import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 1. Load Data ---
DATA_PATH = os.path.join('DATA') 
actions = np.array(['hello', 'thank', 'iloveyou', 'i_am'])
no_sequences = 30
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 2. Build and Train the LSTM Model ---

model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(30,63)),
    Dropout(0.3),
    LSTM(64, return_sequences=False, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
])


early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_sign_model.keras', save_best_only=True)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(
    X_train, y_train,
    epochs=500,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
) # Training might take some time

# --- 3. Save the Model ---
model.save('sign_model.keras')
print("Model trained and saved as sign_model.keras (last epoch) and best_sign_model.keras (best validation epoch)")