import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import joblib

def get_unique_pitches_one_oct(acc):
    
    pitches = [index % 12 for instant in acc for index, vel in enumerate(instant) if vel > 0]
     
    return list(set(pitches))

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
        
        piano_rolls =  np.split(piano_roll, [500], axis=1)
        
        while piano_rolls[-1].shape[-1] > 500:
            
            last_split = np.split(piano_rolls[-1], [500], axis=1)
            piano_rolls[-1] = last_split[0]
            piano_rolls.append(last_split[-1])
        
        # while np.array(piano_rolls[-1]).shape > size:
        #     piano_rolls.append(np.split(piano_roll[-1], [500], axis=1))
        return piano_rolls

def preprocess(X):
    
    if X.shape[-1] != 500:
        X = reshape_piano_roll(X)
        
    if len(X.shape) < 3:
        X = X.reshape((1, X.shape[0], X.shape[1]))

    chord = adding_chords_info(X)

    X = X.reshape((X.shape[0], -1))
    X = np.concatenate((chord, X), axis=1)

    return X