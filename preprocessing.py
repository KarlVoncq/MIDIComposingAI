from posixpath import abspath
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import os
import pathlib

def get_unique_pitches_one_oct(piano_roll):
    """
    Return a list of notes pitches presents in the accompaniment,
    reduced to one octave

    Args
    ----

    piano_roll: array of shape (128, nb_of_instants)
        A pretty_midi.piano roll of some MIDI music

    Returns
    -------

    list
        a list of all notes presents in the accompaniment, reduce to one octave
        (0 <= len(list) <= 12)
    """

    # Get all semi_tones for each instant in the accompaniment
    pitches = [
        index % 12 for instant in piano_roll
        for index, vel in enumerate(instant) if vel > 0
        ]

    # Get the unique values
    notes_list = list(set(pitches))

    return notes_list


def adding_chords_info(dataset, verbose=0):
    """
    Create chords feature to a dataset of n piano rolls

    Args
    ----
    dataset: ndarray of shape (n samples, 128, n instants)
        A dataset of piano rolls, samples of MIDI files

    verbose: int
        A basic verbose arguments, because the process can be very long
        If 1 : every 100 sample, print the number of sample done
    
    Returns
    -------
    chords_encoded: array
        An array with 0 or 1 either the chord is present in the accompaniment
        or not
    """

    # Reshaping dataset if it's only one sample of 2 dim
    if len(dataset.shape) < 3:
        dataset = dataset.reshape((1, dataset.shape[0], dataset.shape[1]))
    
    # Get relative path of chords_midi.csv to use the function wherever we want
    abs_path = os.path.join(os.path.dirname(__file__), 'Data/chords_midi.csv')
    path = os.path.relpath(abs_path)

    # Create a dataframe from csv
    chords_df = pd.read_csv(path, sep=";")

    # Create a dict of each chord from the dataframe
    chords_dict = chords_df.set_index('Chord').T.to_dict('list')

    # One Hot encoding chords feature
    mlb = MultiLabelBinarizer()
    mlb.fit([chords_dict.keys()])
    df_list = []
    for i, data in enumerate(dataset):
        pitches = get_unique_pitches_one_oct(data)

        chords = [key for key, value in chords_dict.items() if set(
            value).issubset(pitches)]
        list_for_df = [[data, chords]]
        df_list.append([list_for_df, chords])
        if verbose == 1 and i % 100 == 0:
            print(f'{i+1} done')

    df = pd.DataFrame(df_list, columns=['Acc', 'Chords'])

    chords_encoded = mlb.transform(df['Chords'])

    return chords_encoded


def reshape_piano_roll(piano_roll, size=500):
    """
    Reshape the piano roll to match into the required shape for predictions

    Args
    ----
    piano_roll: array of shape(128, n instants) or (n samples, 128, n instants)
        A pretty_midi.piano_roll
    size: int
        Number of instants on the the piano_roll we want for each sample
        size = seconds * fs
    
    Returns
    -------
    piano_roll: array of shape(128, size)
        The reshape piano_roll
    piano_rolls: array of shape(n samples, 128, size)
        An array of reshaped piano rolls
    """

    # Reshaping too short piano_roll
    if piano_roll.shape[-1] < size:

        # Create an array of zeros to complement piano_roll
        zeros_array = np.zeros((128, size - piano_roll.shape[-1]))

        # Concatenate both
        piano_roll = np.concatenate((piano_roll, zeros_array), axis=1)

        return piano_roll
    
    # Reshaping too long piano_roll into different samples of the correct size
    if piano_roll.shape[-1] > size:

        # We create a first split (array of shape (128, 500), array of shape ?)
        piano_rolls = np.split(piano_roll, [size], axis=1)

        # We do it while the last split is too long
        while piano_rolls[-1].shape[-1] > size:

            # We continue the split until every split is <= size
            last_split = np.split(piano_rolls[-1], [size], axis=1)
            piano_rolls[-1] = last_split[0]
            piano_rolls.append(last_split[-1])

        # Then we reshape the last split if it's inferior to the size argument
        if piano_rolls[-1].shape[-1] < size:

            zeros_array = np.zeros((128, size - piano_rolls[-1].shape[-1]))
            piano_rolls[-1] = np.concatenate((piano_rolls[-1],
                                             zeros_array), axis=1)

        # We want an array, not a list
        piano_rolls = np.asarray(piano_rolls)

        return piano_rolls


def preprocess(X, size=500):
    """
    Preprocessing X to put it as an input for our model
    Args
    ----
    X: np.array of dim(n samples, 128, n_instants)
        An array of piano rolls
    size: int
        The number of instants we want our piano rolls to have
    
    Returns
    -------
    X: np.array of dim(n samples, 128, size)
    """

    # First we make sure the shape fit (50 fs, 10s)
    if X.shape[-1] != size:
        X = reshape_piano_roll(X, size)

    # Then we add a dim for adding chords
    if len(X.shape) < 3:
        X = X.reshape((1, X.shape[0], X.shape[1]))

    # We add chords
    chord = adding_chords_info(X)

    # We flattened the accompaniment
    X = X.reshape((X.shape[0], -1))

    # We merge the two
    X = np.concatenate((chord, X), axis=1)

    return X
