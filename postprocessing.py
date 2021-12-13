import numpy as np
from numpy.lib.utils import safe_eval

from preprocessing import reshape_piano_roll

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

    # Mid size : the separation between pitch and velocity
    mid_size = int(target[0].shape[-1]/2)
    pitches, velocities =  target[0][0][:mid_size], target[0][0][mid_size:]
    piano_roll = melody_to_piano_roll(pitches, velocities)

    for sample in target[1:]:

        pitches, velocities =  sample[0][:mid_size], sample[0][mid_size:]
        piano_roll = np.concatenate((piano_roll, melody_to_piano_roll(pitches, velocities)))
    
    # Reshape to merge with accompaniment
    return piano_roll.reshape((-1, 128, 500))

def assemblate_accompaniment_melody(accompaniment, melody):
    """
    Assemblate accompaniment and melody, must be the same size
    """
    # for i, frame in enumerate(melody):
    #     for j, note in enumerate(frame):
    #         if note > 0:
    #             accompaniment[i][j] = note


    

    if accompaniment.shape[-1] != 500:
        accompaniment = reshape_piano_roll(accompaniment)

    return accompaniment + melody

def postprocess(acc, pred):

    mel = assembled_target_to_melody(pred)

    full_music = assemblate_accompaniment_melody(acc, mel)

    sample_mel = mel[0]
    for sample in mel[1:]:
        sample_mel = np.concatenate((sample_mel, sample), axis=1)
    mel = sample_mel

    sample_full = full_music[0]
    for sample in full_music[1:]:
        sample_full = np.concatenate((sample_full, sample), axis=1)
    full_music = sample_full
    
    return mel, full_music
