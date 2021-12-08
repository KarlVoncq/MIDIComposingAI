from librosa.core.audio import resample
from matplotlib.pyplot import plot
from numpy.core.numeric import full
import streamlit as st
from requests.api import request
from utils import plot_piano_roll_librosa
import pretty_midi
import requests
import numpy as np
import joblib
from preprocessing import  adding_chords_info
from postprocessing import assembled_target_to_melody, assemblate_accompaniment_melody
from midi2audio import FluidSynth
import io
from scipy.io import wavfile
from midi2audio import FluidSynth
import base64
import os

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

st.title('Downloader')

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    print(href)
    return href

st.markdown(get_binary_file_downloader_html('test.mid', 'Text Download'), unsafe_allow_html=True)

st.markdown('''
Let us  find you a melody for your MIDI file !
''')

# Load user MIDI file
uploaded_file = st.file_uploader("Choose a file", type=['mid'])

pretty_midi.PrettyMIDI(uploaded_file).write('new.mid')

fs = FluidSynth()

fs.midi_to_audio('new.mid', 'new.wav')

st.audio('new.wav')

with st.spinner(f"Transcribing to FluidSynth"):
    midi_data = pretty_midi.PrettyMIDI(uploaded_file)
    audio_data = midi_data.fluidsynth()
    audio_data = np.int16(
        audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
    )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, 44100, audio_data)

st.audio(virtualfile)
st.markdown("Download the audio by right-clicking on the media player")

# st.audio('tes', format='audio/mp3', start_time=0)
# # pm = pretty_midi.PrettyMIDI('file.mid')
# # url = 'URL/TO/API'

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




# predicted_melody = form.form_submit_button(label='Give me some melody', on_click=predict())
predicted_melody = pretty_midi.PrettyMIDI('test.mid')
# # TODO : plot predicted melody, listen predicted melody, download predicted_melody, plot acc + melody, listen acc + melody, download acc + melody

# full_music = assemblate_accompaniment_melody(piano_roll, predicted_melody)

# # full_music = piano_roll_to_pretty_midi(full_music).write('full_music.mid')



plot_piano_roll_librosa(predicted_melody, fs=50, name_fig='YOur music with anew melody !')
# st.audio(full_music)

import streamlit.components.v1 as components

html_script = f"""<script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>

    <midi-player
    src={uploaded_file}>
    </midi-player>
"""

components.html(html_script)

# st.header("File Download - A Workaround for small data")