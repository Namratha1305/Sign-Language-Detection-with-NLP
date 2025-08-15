import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

# --- 1. WORD PREDICTOR MODEL SETUP ---

def train_or_load_word_predictor():
    """
    Trains the word predictor model from an external text file if it doesn't exist,
    otherwise loads the pre-trained model.
    """
    corpus_path = 'large_text_corpus.txt'
    if not os.path.exists(corpus_path):
        with open(corpus_path, 'w') as f:
            f.write("hello how are you\nthank you\niloveyou\ni am going\n")
        print(f"'{corpus_path}' not found. A default file has been created.")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        text_data = f.read().lower()

    model_exists = os.path.exists('word_predictor.keras')
    tokenizer_exists = os.path.exists('word_tokenizer.pkl')
    max_len_exists = os.path.exists('max_sequence_len.pkl')

    if model_exists and tokenizer_exists and max_len_exists:
        print("Loading existing word predictor model.")
        model = load_model('word_predictor.keras')
        with open('word_tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('max_sequence_len.pkl', 'rb') as f:
            max_sequence_len = pickle.load(f)
        return model, tokenizer, max_sequence_len

    print("Training new word predictor model from corpus. This may take some time...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text_data])
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in text_data.split('\n'):
        if not line: continue
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences]) if input_sequences else 10
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    X, y = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
        tf.keras.layers.LSTM(150),
        tf.keras.layers.Dense(total_words, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=1)

    model.save('word_predictor.keras')
    with open('word_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('max_sequence_len.pkl', 'wb') as f:
        pickle.dump(max_sequence_len, f)
    print("Training complete.")
    return model, tokenizer, max_sequence_len

def predict_next_words(seed_text, num_suggestions=3):
    """Predicts the next most likely words based on the seed text."""
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = word_model.predict(token_list, verbose=0)[0]
    predicted_indices = np.argsort(predicted_probs)[-num_suggestions:][::-1]
    
    output_words = []
    for index in predicted_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                output_words.append(word)
                break
    return output_words

# --- 2. SIGN LANGUAGE RECOGNIZER SETUP ---
actions = np.array(['hello', 'thank', 'iloveyou', 'i_am']) # Using the 4-sign model
try:
    sign_model = load_model('best_sign_model.keras')
except Exception as e:
    print(f"Error loading sign model: {e}")
    print("Please make sure 'best_sign_model.keras' exists and is trained with the correct actions.")
    exit()


# --- 3. GUI APPLICATION SETUP ---
class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg='#2C2F33')

        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_frame = tk.Frame(window, bg='#23272A')
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.chat_frame = tk.Frame(window, bg='#2C2F33', width=400)
        self.chat_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(self.video_frame, width=self.width, height=self.height, bg='#23272A', highlightthickness=0)
        self.canvas.pack()

        self.chat_display = tk.Text(self.chat_frame, wrap=tk.WORD, state=tk.DISABLED, bg='#36393F', fg='white', font=("Helvetica", 12), relief=tk.FLAT, padx=10, pady=10)
        self.chat_display.pack(padx=5, pady=5, expand=True, fill=tk.BOTH)
        
        self.button_frame = tk.Frame(self.chat_frame, bg='#2C2F33')
        self.button_frame.pack(fill=tk.X, pady=(0, 5))

        self.copy_button = tk.Button(self.button_frame, text="Copy Text", command=self.copy_to_clipboard, bg='#7289DA', fg='white', relief=tk.FLAT, font=("Helvetica", 10, "bold"))
        self.copy_button.pack(side=tk.LEFT, padx=5)
        
        self.suggestion_frame = tk.Frame(self.chat_frame, bg='#2C2F33')
        self.suggestion_frame.pack(fill=tk.X, pady=5)
        
        self.input_frame = tk.Frame(self.chat_frame, bg='#40444B')
        self.input_frame.pack(fill=tk.X)
        
        self.text_input = tk.Entry(self.input_frame, bg='#40444B', fg='white', font=("Helvetica", 14), relief=tk.FLAT, insertbackground='white')
        self.text_input.pack(side=tk.LEFT, expand=True, fill=tk.X, ipady=10, padx=10)
        
        self.text_input.bind("<Return>", self.send_message_event)
        
        self.send_button = tk.Button(self.input_frame, text="Send", command=self.send_message, bg='#7289DA', fg='white', relief=tk.FLAT, font=("Helvetica", 10, "bold"), padx=10)
        self.send_button.pack(side=tk.RIGHT, padx=10)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.sequence = []
        self.current_state = "WAITING"
        self.last_hand_pos = None
        self.motion_threshold = 0.02
        self.sign_threshold = 0.9

        self.delay = 15
        self.update()
        self.window.mainloop()

    def copy_to_clipboard(self):
        """Copies the content of the chat display to the clipboard."""
        self.window.clipboard_clear()
        chat_content = self.chat_display.get("1.0", tk.END)
        self.window.clipboard_append(chat_content)
        print("Chat content copied to clipboard.")

    def send_message_event(self, event):
        """Allows the Enter key to trigger sending a message."""
        self.send_message()

    def send_message(self):
        message = self.text_input.get()
        if message:
            self.chat_display.config(state=tk.NORMAL)
            self.chat_display.insert(tk.END, "You: " + message + "\n\n")
            self.chat_display.config(state=tk.DISABLED)
            self.chat_display.see(tk.END)
            self.text_input.delete(0, tk.END)
            self.update_suggestions()

    def add_suggestion(self, word):
        current_text = self.text_input.get()
        if current_text and not current_text.endswith(' '):
            self.text_input.insert(tk.END, " " + word)
        else:
            self.text_input.insert(tk.END, word)
        self.update_suggestions()

    def update_suggestions(self):
        for widget in self.suggestion_frame.winfo_children():
            widget.destroy()
            
        current_text = self.text_input.get().strip()
        if not current_text:
            return
            
        suggestions = predict_next_words(current_text, 3)
        
        for word in suggestions:
            btn = tk.Button(self.suggestion_frame, text=word, command=lambda w=word: self.add_suggestion(w), bg='#40444B', fg='white', relief=tk.FLAT)
            btn.pack(side=tk.LEFT, padx=2)

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            display_text = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
                    current_hand_pos = keypoints[:3]

                    if self.last_hand_pos is not None:
                        motion = np.linalg.norm(current_hand_pos - self.last_hand_pos)
                        
                        # --- NEW: Smarter Cooldown Logic ---
                        if self.current_state == "WAITING" and motion > self.motion_threshold:
                            self.current_state = "RECORDING"
                            self.sequence = []
                            
                        elif self.current_state == "RECORDING":
                            self.sequence.append(keypoints)
                            if len(self.sequence) == 30:
                                res = sign_model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                                predicted_action = actions[np.argmax(res)]
                                
                                if res[np.argmax(res)] > self.sign_threshold:
                                    self.add_suggestion(predicted_action.replace('_', ' '))
                                
                                # Go to cooldown immediately after prediction
                                self.current_state = "COOLDOWN"

                        elif self.current_state == "COOLDOWN":
                            # If hand is still, reset to waiting
                            if motion < self.motion_threshold:
                                self.current_state = "WAITING"

                    self.last_hand_pos = current_hand_pos
            else:
                # If hand leaves the frame, reset to waiting
                self.last_hand_pos = None
                if self.current_state != "RECORDING":
                     self.current_state = "WAITING"

            if self.current_state == "RECORDING":
                display_text = f"RECORDING... {len(self.sequence)}/30"
            elif self.current_state == "WAITING":
                display_text = "WAITING FOR SIGN"
            elif self.current_state == "COOLDOWN":
                display_text = "RESETTING..."

            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    word_model, tokenizer, max_sequence_len = train_or_load_word_predictor()
    Application(tk.Tk(), "Sign Language to Text Chat")
