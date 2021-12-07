import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from MIDIComposingAI.data import get_chords_data

def get_unique_pitches_one_oct(acc):

    pitches = [index % 12 for instant in acc for index, vel in enumerate(instant) if vel > 0]

    return list(set(pitches))

def adding_chords_info(dataset):
    # Get chords from csv & store as dict
    chords_df = get_chords_data()
    chords_dict = chords_df.set_index('Chord').T.to_dict('list')

    # Create multilabelbinarizer with chords
    mlb = MultiLabelBinarizer()
    mlb.fit([chords_dict.keys()])

    df_list = []
    for data in dataset:
        # Get unique pitches for each sample
        pitches = get_unique_pitches_one_oct(data)
        # Get list of all chords in each sample
        chords = [key for key, value in chords_dict.items() if set(value).issubset(pitches)]
        list_for_df = [[data, chords]]
        df_list.append([list_for_df, chords])

    df = pd.DataFrame(df_list, columns=['Acc', 'Chords'])
    # Get array with 0 / 1 based on the chords in each sample
    chords_encoded = mlb.transform(df['Chords'])

    return chords_encoded
