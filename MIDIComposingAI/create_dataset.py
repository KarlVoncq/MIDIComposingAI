import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer


def extract_accompaniment_melody(pretty_midi, fs=50, sample_length=10, ratio=0.1, sample_set=0):
    """
    Extract melody and accompaniment from a pretty_midi file.

    Args
    ----
    pretty_midi: a pretty_midi.Pretty_midi() object or file
    fs: int
        Number of frames per second, used to create the piano roll
        from the pretty_midi file
    ratio: float
        The ratio (difference between two last notes (pitch)
        divided by the number of frames between them)
        above wich we won't accept a note to be extract,
        it prevents from having jerky melodies
    sample_lentgh: int
        The length in seconds of the sample we wan't to extract the melody from.
    sample_set: int
        It allows you to choose where in the piece you want to extract the melody

    Returns
    -------
    accompaniment: array of shape (128, fs * sample_length)
        The extracted accompaniment
    melody: array of shape (128, fs * sample_length)
        The extracted melody, basically the highest note at each frame
    """
        
    sample_size = sample_length * fs

    piano_roll = pretty_midi.get_piano_roll(
        fs=fs)[:, sample_size*sample_set:sample_size*(sample_set+1)]

    # An empty piano roll to be filled with the melody
    empty_piano_roll = np.zeros(piano_roll.shape)

    nb_frames = 0

    test = True
    
    def test_statement():
        # This function allow us to execute the next loop without writing it twice
        # just to instanciate the first played note
        try:
            test = abs(last_played_note - piano_roll[j][i])/nb_frames <= ratio
        except:
            test = True
            
    for i in range(sample_size):
        nb_frames += 1
        for j in range(127, 0, -1):
            
            test_statement()
            if piano_roll[j][i] > 0 and test:

                last_played_note = piano_roll[j][i]
                
                if last_played_note <= 127:
                    empty_piano_roll[j][i] = last_played_note
                else:
                    empty_piano_roll[j][i] = 127
                piano_roll[j][i] = 0
                nb_frames = 0
                break
    
    melody = empty_piano_roll

    accompaniment = piano_roll

    return accompaniment, melody


def separate_pitch_velocity(melody):
    """
    Separate pitch and velocity within the melody

    Args
    ----
    melody: array of shape(128, number of frames) or (n samples, 128, number of frames)
        Melody extracted with extract_accompaniment_melody
    
    Returns
    -------
    target: array of shape (number of samples, number of frames * 2)
        The future target of our model
    """

    if len(melody.shape) < 3:
        melody = melody.reshape(1, 128, 500)
        velocities = []
        pitches = []
    else:
        sample_velocities = []
        sample_pitches = []

    for sample in melody.T:

        if melody.shape[0] > 1:
            velocities = []
            pitches = []

        for frame in sample.T:
            frame = list(frame)
            velocity = np.sum(frame)
            if velocity > 127:
                velocity = 127
            velocities.append(velocity)
            pitches.append(frame.index(velocity))

        if melody.shape[0] > 1:
            sample_velocities.append(velocities)
            sample_pitches.append(pitches)

    try:
        target = np.array((sample_pitches, sample_velocities),
                          dtype=np.int8).reshape(sample_pitches.shape[0], -1)
    except:
        target = np.array((pitches, velocities), dtype=np.int8).reshape(1, -1)

    return target


def create_simple_dataset(file, ratio=0.1):
    """
    Create a simple dataset for a model

    Args
    ----
    file: a pretty_midi file or object

    Returns
    -------
    X: array of shape(number of samples, 128, number of frames)
        The accompaniment
    y: array of shape(number of samples, number of frames * 2)
    """
    X = []
    y = []
    i = 0
    while True:
        try:
            accompaniment, melody = extract_accompaniment_melody(
                file, ratio=ratio, sample_set=i)
            X.append(accompaniment)
            y.append(separate_pitch_velocity(melody))
            i += 1
        except:
            break

    X = np.array(X, dtype=np.int8)
    y = np.array(y, dtype=np.int8)

    return X, y


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
    abs_path = os.path.join(os.path.dirname(__file__), '..raw_data/chords_midi.csv')
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

def create_dataset(pretty_midi, **kwargs):
    """
    Create a full dataset with chords feature

    Args
    ----
    pretty_midi: a pretty_midi.PrettyMidi object or file
    
    Returns
    -------
    X: np.array of shape(number of samples, 128 * number of frames + 24)
        Our features
    y: np.array of shape(number of samples, number of frames * 2)
        Our target
    """
    X, y = create_simple_dataset(pretty_midi, ratio=0.1)

    chord = adding_chords_info(X, **kwargs)

    X = X.reshape((X.shape[0], -1))
    X = np.concatenate((chord, X), axis=1, dtype=np.int8)

    return X, y
