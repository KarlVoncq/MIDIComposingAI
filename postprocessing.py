import numpy as np

def melody_to_piano_roll(pitches, velocities):
    """
    Create a piano roll from a list of pitches and a list of velocities
    """

    piano_roll = np.zeros((128, pitches.shape[0]))
    
    for i, (pitch, velocity) in enumerate(zip(pitches, velocities)):
        if pitch > 0:
            piano_roll[int(pitch)][i] = velocity
            
    return piano_roll

def assembled_target_to_melody(target):
    """
    TODO : doc string
    """
    
    pitches, velocities =  target[:500], target[500:]

    piano_roll = melody_to_piano_roll(pitches, velocities)

    return piano_roll

def assemblate_accompaniment_melody(accompaniment, melody):
    """
    Assemblate accompaniment and melody, must be the same size
    """
    # for i, frame in enumerate(melody):
    #     for j, note in enumerate(frame):
    #         if note > 0:
    #             accompaniment[i][j] = note

    full_musics = accompaniment + melody

    return full_musics
