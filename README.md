**Real-Time Sign Language Recognition and Next-Word Prediction**

This project is an interactive application that uses computer vision and natural language processing to recognize sign language gestures in real-time and suggest the next likely words, creating a seamless communication tool. The app features a chat-style interface where users can compose messages by performing signs and selecting from AI-powered suggestions.

**🌟 Features**

Real-Time Sign Recognition: Recognizes a vocabulary of custom-trained signs directly from a webcam feed.

Intelligent Word Suggestions: An LSTM-based NLP model predicts and suggests the next three most likely words to follow the recognized sign.

Interactive Chat Interface: A user-friendly GUI with a live video feed and a chat window to compose messages.

Automatic Motion Detection: The app intelligently waits for the user to start signing, automatically triggering the recording and prediction process without any key presses.

Dynamic UI: Features a smarter cooldown logic that resets instantly after a sign, making the conversation flow faster.

Chat Functionality: Includes a "Send" button, "Enter" key functionality to send messages, and a "Copy to Clipboard" button.

**💻 Technology Stack**

Python: The core programming language.

OpenCV: For real-time video capture and image processing.

MediaPipe: For high-fidelity hand and landmark tracking.

TensorFlow / Keras: For building and training the two deep learning models (LSTM networks).

Tkinter: For creating the graphical user interface (GUI).

NumPy: For numerical operations and data handling.

Pillow (PIL): For integrating OpenCV video frames into the Tkinter GUI.

**🤔 How It Works**

The application's intelligence comes from two separate and specialized neural networks working in tandem:

The Vision Model (The "Eyes" 👀): A recurrent neural network (LSTM) trained on sequences of 3D hand landmark data. Its only job is to watch the user's hand movements and classify them into a known sign (e.g., 'hello', 'i_am').

The Language Model (The "Brain" 🧠): A second LSTM network trained on a text corpus. Its only job is to take a word or phrase as input and predict the most statistically likely words to follow, based on the patterns it learned from the text.

The main application acts as a coordinator, taking the word output from the Vision Model and feeding it as input to the Language Model to generate the final suggestions.

**🛠️ Setup and Installation**

Clone the repository:

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Create and activate a virtual environment:

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS / Linux
python3 -m venv venv
source venv/bin/activate

Install the required libraries:

pip install -r requirements.txt

**▶️ How to Use**

Train the Models (First-time setup):

To train the sign language model, run data_collector.py to gather your sign data, then run train_sign_model.py.

The word predictor model trains automatically the first time you run the main app.

Launch the Application:

python chat_app.py

Using the App:

The app will display "WAITING FOR SIGN".

Make a clear, deliberate sign in front of the camera. The app will automatically detect the motion and change its status to "RECORDING...".

Once the sign is recognized, the word will be added to the text box, and suggestions will appear.

Hold your hand still or move it out of frame to reset the app for the next sign.

Use the chat interface to build your message and click "Send" or press "Enter".

**📁 Project Structure**
.
├── 📄 chat_app.py             # Main application file with GUI
├── 📄 data_collector.py       # Script to collect sign language data
├── 📄 train_sign_model.py     # Script to train the vision model
├── 📄 large_text_corpus.txt   # Text file for training the language model
├── 📄 requirements.txt        # List of Python dependencies
├── 📁 models/                 # Folder for the trained models
│   ├── 📄 best_sign_model.keras
│   ├── 📄 word_predictor.keras
│   ├── 📄 word_tokenizer.pkl
│   └── 📄 max_sequence_len.pkl
└── 📄 README.md
