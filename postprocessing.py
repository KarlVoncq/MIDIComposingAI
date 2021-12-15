import numpy as np
from preprocessing import reshape_piano_roll

def melody_to_piano_roll(pitches, velocities):
    """
    Create a piano roll from a list of pitches and a list of velocities
    
    Args
    ----
    pitches: array
        A list or a 1d array of len = n instants
    velocities: array
        A list or 1d array of the same lenght
    
    Returns
    -------
        piano_roll: an array of size (128, n instants)
    """

    # First we instanciate an array of zeros
    piano_roll = np.zeros((128, pitches.shape[0]))
    
    # Then we add the note
    for i, (pitch, velocity) in enumerate(zip(pitches, velocities)):
        # pitch => index
        # velocity => value at this index
        if pitch > 0:
            piano_roll[int(pitch)][i] = velocity

    return piano_roll

def assembled_target_to_melody(target):
    """
    As our model have a target (and so an output) of shape (1, n instants),
    we want to get back a piano roll
    
    Args
    ----
    target: array of shape(1, n instants)
        The output of the model
    
    Returns
    -------
    piano_roll: array of shape (128, n instants)
        The piano roll we'll use to get back music as MIDI
    """

    # Mid size : the separation between pitch and velocity
    mid_size = int(target[0].shape[-1]/2)

    # We separate pitch and velocity from the target
    pitches, velocities =  target[0][0][:mid_size], target[0][0][mid_size:]

    # Then we reconstruct a piano roll
    piano_roll = melody_to_piano_roll(pitches, velocities)

    for sample in target[1:]:
        pitches, velocities =  sample[0][:mid_size], sample[0][mid_size:]
        piano_roll = np.concatenate((piano_roll, melody_to_piano_roll(pitches, velocities)))
    
    # Reshape to merge with accompaniment
    piano_roll = piano_roll.reshape((-1, 128, mid_size))

    return piano_roll

def assemblate_accompaniment_melody(accompaniment, melody):
    """
    Assemblate accompaniment and melody, must be the same size

    Args
    ----
    accompaniment: array of shape (128, n instants)
        The accompaniment of the music
    melody: array of the same shape
        The melody of the music
    
    Returns
    -------
    accompaniment + melody
    """
    
    # We want to reshape the accompaniment to fit with the melody
    if accompaniment.shape[-1] != 500:
        accompaniment = reshape_piano_roll(accompaniment)

    return accompaniment + melody

def postprocess(accompaniment, prediction):
    """
    Postprocess the input and output of our prediction
    in order to be able to transform them in pretty_midi

    Args
    ----
    accompaniment: array of shape (128, n instants) or (n samples, 128, n instants)
        The input of our prediction
    melody: array of the same shape (except sometimes for the last sample)
        The output of our prediction

    Returns
    -------
    melody: array of shape (128, n instants)
        Our model's prediction postprocessed
    full_music: array of shape(128, n instants)
        Our model's prediction with the original accompaniment, postprocessed
    """

    # Getting back a piano roll from our prediction
    melody = assembled_target_to_melody(prediction)

    full_music = assemblate_accompaniment_melody(accompaniment, melody)

    # Here we assemblate all samples in one piano roll
    sample_mel = melody[0]
    for sample in melody[1:]:
        sample_mel = np.concatenate((sample_mel, sample), axis=1)
    melody = sample_mel

    sample_full = full_music[0]
    for sample in full_music[1:]:
        sample_full = np.concatenate((sample_full, sample), axis=1)
    full_music = sample_full
    
    return melody, full_music
