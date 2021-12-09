import streamlit as st
from utils import plot_piano_roll_librosa, piano_roll_to_pretty_midi
import pretty_midi
import requests
import numpy as np
import joblib
import io
from scipy.io import wavfile
import base64
import os
from preprocessing import preprocess, reshape_piano_roll
from postprocessing import postprocess

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

st.markdown('V. 0.3')

uploaded_file = st.file_uploader("Choose a file", type=['mid'])

st.markdown('''
Let us  find you a melody for your MIDI file !
''')

pm = pretty_midi.PrettyMIDI(uploaded_file)
# pretty_midi.PrettyMIDI(uploaded_file).write('new.mid')


# def pretty_midi_to_audio(pm):
with st.spinner(f"Transcribing to FluidSynth"):
    midi_data = pm
    audio_data = midi_data.fluidsynth()
    audio_data = np.int16(
        audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
    )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, 44100, audio_data)

st.audio(virtualfile)

if uploaded_file:
    plot_piano_roll_librosa(pm, 'Your file')
    # pretty_midi_to_audio(pm)


# # Plot user MIDI file
# plot_piano_roll_librosa(pm, "Your MIDI file")

# # Listen user MIDI file
# st.audio(bytes_data, format='wav', start_time=0)

# form = st.form(key='my-form')

# piano_roll = pretty_midi.PrettyMIDI(uploaded_file).get_piano_roll(fs=50)

# chords = adding_chords_info('../raw_data/chords_midi.csv', piano_roll)

# preproc_file = piano_roll.reshape((piano_roll.shape[0], -1))
# preproc_file = np.concatenate((chords, preproc_file), axis=1, dtype=np.int8)

# def predict():

#     # Call API
#     predicted_melody = requests(method='POST', url=url, file=preproc_file)

#     predicted_melody = assembled_target_to_melody(predicted_melody)
#     plot_piano_roll_librosa(piano_roll_to_pretty_midi(predicted_melody, fs=50), "Your brand new melody !")

#     # Listen audio
#     st.audio(piano_roll_to_pretty_midi(predicted_melody, fs=50))
X = pm.get_piano_roll(fs=50)

def predict(X):

    X = preprocess(X)
    tree = joblib.load('Model/API_tree.joblib')
    pred = tree.predict(X)
    print(f'SHAPE PRED : {pred.shape}')
    return pred

pred = predict(X)

if X.shape[-1] != 500:
        X = reshape_piano_roll(X)

mel, full_music = postprocess(X, pred[0])
pm_mel = piano_roll_to_pretty_midi(mel, fs=50)
pm_full_music = piano_roll_to_pretty_midi(full_music, fs=50)

plot_piano_roll_librosa(pm_mel, 'Melody')
# pretty_midi_to_audio(pm_mel)

plot_piano_roll_librosa(pm_full_music, 'Full music')
# pretty_midi_to_audio(pm_full_music)

st.title('Download your melody !')

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

pm_mel.write('pm_mel.mid')
st.markdown(get_binary_file_downloader_html('pm_mel.mid', 'Text Download'), unsafe_allow_html=True)