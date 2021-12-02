import numpy as np
import pandas as pd
import joblib
from os import listdir

def extract_accompaniment_melody(pretty_midi_file, fs=50, ratio=0.01, sample_length=10, sample_set=0):
    """
    Extract melody from a pretty_midi file.
    
    Args :
        pretty_midi_file : a pretty_midi.Pretty_midi() file
        fs : number of frame per second, use to create the piano roll from the pretty_midi file
        ratio : the ratio above wich we won't accept a note to be extract, it prevents from having jerky melodies
        sample_lentgh : integer, the length in seconds of the sample we wan't to extract the melody from.
        sample_set : integer, it allows you to choose where in the piece you want to extract the melody
                      
    Return : a tuple of pretty_midi.piano_roll variables : (accompaniment, melody)
    """
    
    sample_size = sample_length * fs
    
    piano_roll = pretty_midi_file.get_piano_roll(fs=fs)[:, sample_size*sample_set:sample_size*(sample_set+1)]
    empty_piano_roll = np.zeros(piano_roll.shape)
    nb_instant = 0
    for i in range(sample_size):
        nb_instant += 1
        for j in range(127, 0, -1):
            try:
                if piano_roll[j][i] != 0. and abs(last_played_note - piano_roll[j][i])/nb_instant <= ratio:
                    last_played_note = piano_roll[j][i]
                    # We want our values to be between 0 and 127
                    if last_played_note <= 127:
                        empty_piano_roll[j][i] = last_played_note
                    else:
                        empty_piano_roll[j][i] = 127
                    piano_roll[j][i] = 0.
                    nb_instant = 0
                    break
            except:
                if piano_roll[j][i] != 0.:
                    last_played_note = piano_roll[j][i]
                    if last_played_note <= 127:
                        empty_piano_roll[j][i] = last_played_note
                    else:
                        empty_piano_roll[j][i] = 127
                    piano_roll[j][i] = 0.
                    nb_instant = 0
                    break
    return (piano_roll, empty_piano_roll)

def separate_pitch_velocity(target):
    """
    Separate pitch and velocity within the target
    """
    # Lists of each velocities and pitches for each sample
    sample_velocities = []
    sample_pitches = []
    
    for sample in target:
        # Lists of velocities and pitches within the sample
        velocities = []
        pitches = []
        
        for frame in sample.T:
            frame = list(frame)
            velocity = np.sum(frame)
            velocities.append(velocity)
            pitches.append(frame.index(velocity))
        sample_velocities.append(velocities)
        sample_pitches.append(pitches)
    
    return (sample_pitches, sample_velocities)

def create_sample(pretty_midi_file, fs):
    """
    Return a sample of the file
    """
    piano_roll = pretty_midi_file.piano_roll(fs=fs)
    

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

def create_nparray_dataset(file, directory ,name, store=True):
    """
    Create a nparray dataset
    """
    X, y = create_simple_dataset(file)
    
    pitches, velocities = separate_pitch_velocity(y)
    
    X_accompaniment = np.array([accompaniment.T for accompaniment in X])
        
    # Then we add the two target to the dataframe
    y_pitch = np.array([np.array(pitch) for pitch in pitches])
    y_velocity = np.array([np.array(velocity) for velocity in velocities])
    
    dataset = (X_accompaniment, y_pitch, y_velocity)
    
    if store:
        joblib.dump(dataset, f'../raw_data/pandas_dataframes/{directory}/{name}')
    else:
        return dataset
    
    # In the end we need to delete the variables in order to save some RAM
    del([X, y, pitches, velocities, X_accompaniment, y_pitch, y_velocity, dataset])

def create_tuple_target_dataset(file):
    """
    Create a dataset with a target being a list of tuples (pitch, velocity)
    Args:
        file : a pretty_midi object/file
    """
    X, y = create_simple_dataset(file)

    new_target = []

    for sample in y:

        frames = []

        for frame in sample.T:

            list_of_tuples = [(note, velocity) if velocity > 0 else (0, 0) for note, velocity in enumerate(frame)]
            frames.append(list_of_tuples)

        new_target.append(frames)
        
    return X, np.array(new_target)

def create_classified_melody(melody, to_list=True):
    """
    Return a classified melody (0 if no note is played, 1 if a note is played)
    """
    classified_melody = [1 if note > 0 else 0 for note in melody]

    if not to_list:
        return np.array(classified_melody)
    
    return classified_melody
