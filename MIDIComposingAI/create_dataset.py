import numpy as np
import joblib

def extract_accompaniment_melody(pretty_midi_file, fs=50, sample_length=10, ratio=0.1, sample_set=0):
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

    liste = []
    nb_instant = 0

    for i in range(sample_size):
        nb_instant += 1
        for j in range(127, 0, -1):
            try:
                if piano_roll[j][i] > 0 and (abs(last_played_note - piano_roll[j][i])/nb_instant <= ratio):
                    last_played_note = piano_roll[j][i]
                    # We want our values to be between 0 and 127
                    if last_played_note <= 127:
                        empty_piano_roll[j][i] = last_played_note
                    else:
                        empty_piano_roll[j][i] = 127
                    piano_roll[j][i] = 0
                    liste.append([[j],[i]])
                    nb_instant = 0
                    break
            except:
                if piano_roll[j][i] > 0:
                    last_played_note = piano_roll[j][i]
                    if last_played_note <= 127:
                        empty_piano_roll[j][i] = last_played_note
                    else:
                        empty_piano_roll[j][i] = 127
                    piano_roll[j][i] = 0
                    break
    return (piano_roll, empty_piano_roll)

def separate_pitch_velocity(target):
    """
    Separate pitch and velocity within the target
    """
    # Lists of each velocities and pitches for each sample

    if len(target.shape) < 3:
        target = target.reshape(1, 128, 500)
        velocities = []
        pitches = []

    else:
        sample_velocities = []
        sample_pitches = []

    for sample in target.T:
        # Lists of velocities and pitches within the sample

        if target.shape[0] > 1:
            velocities = []
            pitches = []

        for frame in sample.T:
            frame = list(frame)
            velocity = np.sum(frame)
            if velocity > 127:
                velocity = 127
            velocities.append(velocity)
            pitches.append(frame.index(velocity))

        if target.shape[0] > 1:
            sample_velocities.append(velocities)
            sample_pitches.append(pitches)

    try :
        melody = np.array((sample_pitches, sample_velocities), dtype=np.int8).reshape(sample_pitches.shape[0], -1)
    except:
        melody = np.array((pitches, velocities), dtype=np.int8).reshape(1, -1)

    return (melody)

def create_simple_dataset(file, ratio=0.1):
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
            accompaniment, melody = extract_accompaniment_melody(file, ratio=ratio, sample_set=i)
            X.append(accompaniment)
            y.append(separate_pitch_velocity(melody))
            i += 1
        except:
            break

    return np.array(X, dtype=np.int8), np.array(y, dtype=np.int8)

def create_dataset(file, ratio=0.1, directory=None ,name=None, store=False):
    """
    Create a nparray dataset
    """
    X, y = create_simple_dataset(file, ratio=ratio)

    pitches, velocities = separate_pitch_velocity(y)

    y_melody = np.array(
                [(pitch, velocity) for pitch, velocity in zip(pitches, velocities)]
            )

    # X_accompaniment = np.array([accompaniment.T for accompaniment in X])

    # Then we add the two target to the dataframe
    # y_pitch = np.array([np.array(pitch) for pitch in pitches])
    # y_velocity = np.array([np.array(velocity) for velocity in velocities])

    dataset = (X, y_melody)

    if store:
        joblib.dump(dataset, f'../raw_data/pandas_dataframes/{directory}/{name}')

        del([X, y, y_melody, pitches, velocities, dataset])

    else:
        return dataset
