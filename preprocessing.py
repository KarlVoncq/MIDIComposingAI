import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import joblib

def get_unique_pitches_one_oct(acc):
    
    pitches = [index % 12 for instant in acc for index, vel in enumerate(instant) if vel > 0]
     
    return list(set(pitches))

def adding_chords_info(dataset, verbose=0):

    chords_df = pd.read_csv('Data/chords_midi.csv', sep=";")

    chords_dict = chords_df.set_index('Chord').T.to_dict('list')

    mlb = MultiLabelBinarizer()
    mlb.fit([chords_dict.keys()])
    df_list = []
    for i, data in enumerate(dataset):
        pitches = get_unique_pitches_one_oct(data)
        
        chords = [key for key, value in chords_dict.items() if set(value).issubset(pitches)]
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
    """
    if piano_roll.shape[-1] < size:
        
        zeros_array = np.zeros((128, size - piano_roll.shape[-1]))
        return np.concatenate((piano_roll, zeros_array), axis=1)
    
    if piano_roll.shape[-1] > size:
        
        # We create a first split (array of shape (128, 500), array of shape ?)
        piano_rolls = np.split(piano_roll, [size], axis=1)
        
        while piano_rolls[-1].shape[-1] > size:
            # We continue the split until every split is <= size
            last_split = np.split(piano_rolls[-1], [size], axis=1)
            piano_rolls[-1] = last_split[0]
            piano_rolls.append(last_split[-1])
        
        if piano_rolls[-1].shape[-1] < size:
            # We reshape the last split if it's inferior to the size argument
            zeros_array = np.zeros((128, size - piano_rolls[-1].shape[-1]))
            piano_rolls[-1] = np.concatenate((piano_rolls[-1], zeros_array), axis=1)
        
        return np.asarray(piano_rolls)

def preprocess(X):
    """
    Preprocessing X to put it as an input for our model
    """

    if X.shape[-1] != 500:
        # First we make sure the shape fit (50 fs, 10s)
        X = reshape_piano_roll(X)

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
