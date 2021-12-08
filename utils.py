import librosa.display
import matplotlib.pyplot as plt
import pretty_midi
import numpy as np
import streamlit as st

def plot_piano_roll_librosa(pm, name_fig, fs=50):
    """
    Use librosa's specshow function for displaying the piano roll (in streamlit framework
    """
    # pm = pretty_midi.PrettyMIDI(midi_file)
    fig = plt.figure(figsize=(10, 8))
    librosa.display.specshow(pm.get_piano_roll(fs),
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note')
                            #  fmin=pretty_midi.note_number_to_hz(start_pitch))
    plt.title(f"{name_fig}", fontsize="x-large")
    plt.xlabel("Time (s)", fontsize="x-large")
    plt.ylabel("Pitch", fontsize="x-large")
    st.pyplot(fig)

def reshape_piano_roll(piano_roll, size=500):
    """
    Reshape the piano roll to match into the required shape for predictions
    """
    if piano_roll.shape[-1] < size:
        
        zeros_array = np.zeros((128, size - piano_roll.shape[-1]))
        return np.concatenate((piano_roll, zeros_array), axis=1)
    
    if piano_roll.shape[-1] > size:
        
        piano_rolls =  np.split(piano_roll, [500], axis=1)
        
        while piano_rolls[-1].shape[-1] > 500:
            
            last_split = np.split(piano_rolls[-1], [500], axis=1)
            piano_rolls[-1] = last_split[0]
            piano_rolls.append(last_split[-1])
        
        # while np.array(piano_rolls[-1]).shape > size:
        #     piano_rolls.append(np.split(piano_roll[-1], [500], axis=1))
        return piano_rolls