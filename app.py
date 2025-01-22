from flask import Flask, render_template, request, redirect, session, flash, url_for
import tensorflow as tf
import tensorflow_text as text
import numpy as np
import speech_recognition as sr
import tensorflow_hub as hub
import pyaudio
import wave
import threading
import os
from langdetect import detect
from googletrans import Translator
import pytesseract
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
os.environ['PATH'] += r";C:\Program Files\Tesseract-OCR"
app = Flask(__name__)
app.secret_key = "your_secret_key"  

client = MongoClient('mongodb://localhost:27017/')
db = client['multilingual']
users_collection = db['users']

bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3")

class BERTPreprocessLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return bert_preprocess(inputs)

class BERTEncoderLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return bert_encoder(inputs)

loaded_model = tf.keras.models.load_model(
    'sentiment_model.h5',
    custom_objects={
        'BERTPreprocessLayer': BERTPreprocessLayer,
        'BERTEncoderLayer': BERTEncoderLayer
    }
)
print("Model loaded successfully.")

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('new_page'))
    return redirect(url_for('login'))

@app.route('/new')
def new_page():
    if 'username' in session:
        return render_template('new.html')
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        phone = request.form['phone']
        email = request.form['email']
        address = request.form['address']
        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for('signup'))

        if users_collection.find_one({'username': username}):
            flash("Username already exists.")
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        users_collection.insert_one({
            'username': username,
            'password': hashed_password,
            'phone': phone,
            'email': email,
            'address': address
        })

        flash("Signup successful! Please login.")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            session['username'] = username
            return redirect(url_for('new_page'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully.")
    return redirect(url_for('login'))

def translate_text(text, target_lang='en'):
    translator = Translator()
    detected_lang = detect(text)
    if detected_lang != target_lang:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    return text

def analyze_text_sentiment(input_text):
    input_text = translate_text(input_text)
    prediction = loaded_model.predict([input_text])
    return prediction

def analyze_audio_sentiment(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        input_text = recognizer.recognize_google(audio)
        input_text = translate_text(input_text)
        prediction = analyze_text_sentiment(input_text)
        return prediction

    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error accessing the speech recognition service: {e}"
    except Exception as e:
        return f"Error processing audio file: {e}"

# Analyze text input
@app.route('/analyze_input1', methods=['POST'])
def analyze_input1():
    input_text = request.form['input_text']
    prediction = analyze_text_sentiment(input_text)
    print(prediction)

    out = ""
    if isinstance(prediction, np.ndarray) and prediction[0][0] > 0.7:
        out = "The Content detected is Not Offence"
    else:
        out = "The Content detected is Offence"
    return render_template('output.html', sentiment_result=out)

# Analyze audio input
@app.route('/analyze_input2', methods=['POST'])
def analyze_input2():
    audio_file = request.files['audio_file']
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(audio_path)
    prediction = analyze_audio_sentiment(audio_path)

    print(prediction)

    out = ""
    if isinstance(prediction, np.ndarray) and prediction[0][0] > 0.7:
        out = "The Content detected is Not Offence"
    else:
        out = "The Content detected is Offence"
    return render_template('output.html', sentiment_result=out)

@app.route('/upload_meme', methods=['POST'])
def upload_meme():
    meme_image = request.files['meme_image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], meme_image.filename)

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    meme_image.save(image_path)

    extracted_text = extract_text_from_image(image_path)

    if extracted_text:
        translated_text = translate_text(extracted_text)
        prediction = analyze_text_sentiment(translated_text)

        if isinstance(prediction, np.ndarray) and prediction[0][0] > 0.7:
            sentiment_result = "The Content detected is Not Offence"
        else:
            sentiment_result = "The Content detected is Offence"
        
        return render_template('output.html', sentiment_result=sentiment_result)
    else:
        return render_template('output.html', sentiment_result="No text found in the meme image.")

def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return None

class Recorder:
    def __init__(self, file_name, save_dir, chunk=1024, channels=2, rate=44100, format=pyaudio.paInt16):
        self.file_name = file_name
        self.save_dir = save_dir
        self.file_path = os.path.join(save_dir, file_name)
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.format = format
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        self.recording = False
        self.lock = threading.Lock()

    def start_recording(self):
        self.recording = True
        while self.recording:
            data = self.stream.read(self.chunk)
            with self.lock:
                self.frames.append(data)

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        with wave.open(self.file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            with self.lock:
                wf.writeframes(b''.join(self.frames))

recorder = None

@app.route('/start')
def start():
    global recorder
    if recorder is None or not recorder.recording:
        recorder = Recorder("recorded_audio.wav", app.config['UPLOAD_FOLDER'])
        threading.Thread(target=recorder.start_recording).start()
    return redirect('/')

@app.route('/stop')
def stop():
    global recorder
    if recorder is not None and recorder.recording:
        threading.Thread(target=recorder.stop_recording).start()
    return redirect('/')

@app.route('/analyse_live')
def analyse_live():
    audio_file = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.wav")
    prediction = analyze_audio_sentiment(audio_file)

    out = ""
    if isinstance(prediction, np.ndarray) and prediction[0][0] > 0.7:
        out = "The Content detected is Not Offence"
    else:
        out = "The Content detected is Offence"
    return render_template('output.html', sentiment_result=out)

@app.route('/translate', methods=['POST'])
def translate():
    input_text = request.form['input_text']
    target_language = request.form['language']
    translation = translate_text(input_text, target_language)
    return render_template('translated.html',original_text=input_text, translated_text=translation, sentiment_result=f'Translated text: {translation}')

@app.route('/profile')
def profile():
    if 'username' in session:
        user = users_collection.find_one({'username': session['username']})
        if user:
            return render_template('profile.html', user=user)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True, port=5050)