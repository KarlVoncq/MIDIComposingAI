import streamlit as st
from utils import plot_piano_roll_librosa, piano_roll_to_pretty_midi
import pretty_midi
import requests
import numpy as np
import base64
import os
from preprocessing import preprocess, reshape_piano_roll
from postprocessing import postprocess
import streamlit.components.v1 as components

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

st.markdown('V. 0.5')
uploaded_file = st.file_uploader("Choose a file", type=['mid'])
st.markdown('''
Let us  find you a melody for your MIDI file !
''')


pm = pretty_midi.PrettyMIDI(uploaded_file)
# pretty_midi.PrettyMIDI(uploaded_file).write('new.mid')


# def pretty_midi_to_audio(pm):
#     with st.spinner(f"Transcribing to FluidSynth"):
#         midi_data = pm
#         audio_data = midi_data.fluidsynth()
#         audio_data = np.int16(
#             audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
#         )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

#         virtualfile = io.BytesIO()
#         wavfile.write(virtualfile, 44100, audio_data)

#     st.audio(virtualfile)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

if uploaded_file:
    plot_piano_roll_librosa(pm, 'Your file')

pm.write('acc.mid')
link_tag_acc = get_binary_file_downloader_html('acc.mid', 'acc.mid')

file_url_acc = link_tag_acc.split('href="')[1].split('" download=')[0]

html_string_acc = f'''
    <midi-player src="{file_url_acc}"></midi-player>
    <midi-visualizer type="piano-roll" id="myVisualizer"></midi-visualizer>
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
'''
components.html(html_string_acc)


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

pm_mel.write('mel.mid')
link_tag_mel = get_binary_file_downloader_html('mel.mid', 'mel.mid')

file_url_mel = link_tag_mel.split('href="')[1].split('" download=')[0]

html_string_mel = f'''
    <midi-player src="{file_url_mel}"></midi-player>
    <midi-visualizer type="piano-roll" id="myVisualizer"></midi-visualizer>
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
'''
components.html(html_string_mel)
st.markdown(link_tag_mel, unsafe_allow_html=True)

plot_piano_roll_librosa(pm_full_music, 'Full music')


pm_full_music.write('full.mid')
link_tag_full = get_binary_file_downloader_html('full.mid', 'full.mid')

file_url_full = link_tag_full.split('href="')[1].split('" download=')[0]

html_string_full = f'''
    <midi-player src="{file_url_full}"></midi-player>
    <midi-visualizer type="piano-roll" id="myVisualizer"></midi-visualizer>
    <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
'''
components.html(html_string_full)

st.title('Download your melody !')

st.markdown(link_tag_full, unsafe_allow_html=True)