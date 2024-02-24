import streamlit as st
import numpy as np    
import tensorflow as tf
import os, urllib
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from io import BytesIO

def main():
    selected_box = st.sidebar.selectbox(
        'Choose an option..',
        ('Emotion Recognition', 'view source code')
    )

    if selected_box == 'Emotion Recognition':
        st.sidebar.success('To try by yourself by adding an audio file.')
        application()
    if selected_box == 'view source code':
        st.code(get_file_content_as_string("app.py"))
st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache_resource(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/chiluveri-sanjay/Emotion-recognition/main/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model('mymodel.h5')
    return model

def application():
    models_load_state = st.text('\n Loading models..')
    model = load_model()
    models_load_state.text('\n Models Loading..complete')

    file_to_be_uploaded = st.file_uploader("Choose an audio...", type=["wav", "mp3", "ogg"])

    if file_to_be_uploaded:
        st.audio(file_to_be_uploaded, format='audio/wav')

        # Save the uploaded audio file to a temporary file
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as temp_file:
            temp_file.write(file_to_be_uploaded.read())

        # Convert the temporary audio file to WAV format using pydub
        audio = AudioSegment.from_file(temp_audio_path)
        wav_path = "converted_audio.wav"
        audio.export(wav_path, format="wav")

        # Add the following lines to display the spectrogram
        st.text("Spectrogram of the Audio:")
        plot_spectrogram(wav_path)

        st.success('Emotion of the audio is  ' + predict(model, wav_path))

def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def plot_spectrogram(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), y_axis='log', x_axis='time')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot()

def predict(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]

if __name__ == "__main__":
    main()
