import numpy as np
import os
from flask import Flask, request, render_template
import librosa
from keras.models import load_model

app = Flask(__name__)

model = load_model('./model/speech_model2.h5')  

def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None) 
    data = np.array(audio).reshape(-1, 16000, 1)
    return data

def classify_audio(audio):
    my_array = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy',
            'house', 'left', 'marvel', 'nine', 'no', 'off', 'on', 'one', 'right',
            'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
    predictions = model.predict(audio)
    predicted_class = np.argmax(predictions)
    return my_array[predicted_class]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'audio' not in request.files:
            return render_template('index.html', error='No audio file selected')
        
        file = request.files['audio']
        
        if file.filename == '':
            return render_template('index.html', error='No audio file selected')
        
        if file and file.filename.lower().endswith(('.wav')):
            audio_path = 'temp.wav'
            file.save(audio_path)
            audio = process_audio(audio_path)
            predictions = classify_audio(audio)
            os.remove(audio_path)
            
            return render_template('index.html', predictions=predictions)
        else:
            return render_template('index.html', error='Invalid file format. Only WAV files are supported.')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)