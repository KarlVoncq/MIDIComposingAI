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

st.set_page_config(layout="wide")

st.markdown('V. 0.5')
uploaded_file = st.file_uploader("Choose a file", type=['mid'])
st.markdown('''
Let us  find you a melody for your MIDI file !
''')

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def play_music(pm, file_name, type):

    pm.write(file_name)
    link_tag = get_binary_file_downloader_html(file_name, file_name)

    file_url = link_tag.split('href="')[1].split('" download=')[0]

    html_string = f'''
        <midi-player src="{file_url}"
        sound-font visualizer="#myVisualizer"
        ></midi-player>
        <midi-visualizer type={type} id="myVisualizer"></midi-visualizer>
        <script src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.23.1/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.4.0"></script>
    '''
    components.html(html_string)

col1, col2, col3 = st.columns(3)

def predict(X):

    X = preprocess(X)
    url = "https://europe-west1-wagon-bootcamp-328620.cloudfunctions.net/midi_composing_api"

    acc = {"acc_to_predict":X.tolist()}
    response = requests.post(url, json=acc)
    return np.asarray(response.json()['result'])

if uploaded_file:
    pm_user = pretty_midi.PrettyMIDI(uploaded_file)
    with col1:
        play_music(pm_user, 'your_music.mid', 'piano-roll')

if st.button("Give me some melody !"):

    if uploaded_file:
        X = pm_user.get_piano_roll(fs=50)

        if X.shape[-1] != 500:
                X = reshape_piano_roll(X)

        pred = predict(X)
        mel, full_music = postprocess(X, pred[0])
        pm_mel = piano_roll_to_pretty_midi(mel, fs=50)
        pm_full_music = piano_roll_to_pretty_midi(full_music, fs=50)
        with col2:
            play_music(pm_mel, 'new_melody.mid', 'piano-roll')
        with col3:
            play_music(pm_full_music, 'full_music.mid', 'piano-roll' )
    else:
        st.markdown("You need to upload a file first ! :)")



try:
    link_tag_mel = get_binary_file_downloader_html('new_melody.mid', 'new_melody.mid')

    st.markdown(link_tag_mel, unsafe_allow_html=True)

    link_tag_full = get_binary_file_downloader_html('full.mid', 'full.mid')


    st.title('Download your melody !')

    st.markdown(link_tag_full, unsafe_allow_html=True)
except:
    pass