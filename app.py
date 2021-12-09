import streamlit as st
from utils import plot_piano_roll_librosa, piano_roll_to_pretty_midi
import pretty_midi
import requests
import numpy as np
import io
from scipy.io import wavfile
import base64
import os
from preprocessing import preprocess, reshape_piano_roll
from postprocessing import postprocess
from midi2audio import FluidSynth

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

st.markdown('V. 0.5')
uploaded_file = st.file_uploader("Choose a file", type=['mid'])
fs = FluidSynth()
st.markdown('''
Let us  find you a melody for your MIDI file !
''')

pm = pretty_midi.PrettyMIDI(uploaded_file)
# pretty_midi.PrettyMIDI(uploaded_file).write('new.mid')


def pretty_midi_to_audio(pm):
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
    pretty_midi_to_audio(pm)


X = pm.get_piano_roll(fs=50)

if X.shape[-1] != 500:
        X = reshape_piano_roll(X)

def predict(X):

    X = preprocess(X)
    url = "https://europe-west1-wagon-bootcamp-328620.cloudfunctions.net/midi_composing_api"

    acc = {"acc_to_predict":X.tolist()}
    response = requests.post(url, json=acc)
    return np.asarray(response.json()['result'])

pred = predict(X)

mel, full_music = postprocess(X, pred[0])
pm_mel = piano_roll_to_pretty_midi(mel, fs=50)
pm_full_music = piano_roll_to_pretty_midi(full_music, fs=50)

plot_piano_roll_librosa(pm_mel, 'Melody')
pretty_midi_to_audio(pm_mel)

plot_piano_roll_librosa(pm_full_music, 'Full music')
pretty_midi_to_audio(pm_full_music)

st.title('Download your melody !')

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

pm_mel.write('AI_mel.mid')
st.markdown(get_binary_file_downloader_html('pm_mel.mid', 'AI_melody.mid'), unsafe_allow_html=True)