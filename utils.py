from os import fsdecode
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
    return fig

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

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm