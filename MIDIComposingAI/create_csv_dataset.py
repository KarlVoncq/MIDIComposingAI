import numpy as np
import pandas as pd
import joblib
from os import listdir

def extract_accompaniment_melody(pretty_midi_file, fs=1_000, ratio=0.01, sample_size=10_000, sample_set=0):
    """
    Extract melody from a pretty_midi file.
    
    Args :
        pretty_midi_file : a pretty_midi.Pretty_midi() file
        fs : number of frame per second, use to create the piano roll from the pretty_midi file
        ratio : the ratio above wich we won't accept a note to be extract, it prevents from having jerky melodies
        sample_size : integer, the size of the sample we wan't to extract the melody from.
                      If you want all the piece -> sample_size=piano_roll.shape[1]
        sample_set : integer, it allows you to choose where in the piece you want to extract the melody
                      
    Return : a tuple of pretty_midi.piano_roll variables : (accompaniment, melody)
    """
    piano_roll = pretty_midi_file.get_piano_roll(fs=fs)[:, sample_size*sample_set:sample_size*(sample_set+1)]
    empty_piano_roll = np.zeros(piano_roll.shape)
    nb_instant = 0
    for i in range(sample_size):
        nb_instant += 1
        for j in range(127, 0, -1):
            try:
                if piano_roll[j][i] != 0. and abs(last_played_note - piano_roll[j][i])/nb_instant <= ratio:
                    last_played_note = piano_roll[j][i]
                    empty_piano_roll[j][i] = last_played_note
                    piano_roll[j][i] = 0.
                    nb_instant = 0
                    break
            except:
                if piano_roll[j][i] != 0.:
                    last_played_note = piano_roll[j][i]
                    empty_piano_roll[j][i] = last_played_note
                    piano_roll[j][i] = 0.
                    nb_instant = 0
                    break
    return (piano_roll, empty_piano_roll)

def create_sample(pretty_midi_file):
    """
    Return a sample of the file
    """
    piano_roll = pretty_midi_file.piano_roll(fs=1_000)
    

def create_simple_dataset(file, mode=None):
    """
    Create a simple dataset for ML/DL

    Args :
        file : A pretty_midi file

    Return : A tuple with X = accompaniment, y = melody
    """
    X = []
    y = []
    i = 0
    while True:
        try:
            accompaniment, melody = extract_accompaniment_melody(file, sample_set=i)
            X.append(accompaniment)
            y.append(melody)
            i += 1
        except:
            break

    return np.array(X), np.array(y)
