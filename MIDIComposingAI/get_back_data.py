import numpy as np

def target_to_melody(target):
    """
    Create a melody from a list of tuples (pitch, velocity)
    Args:
        target, the target of the models, ndim = 3
    Return:
        A piano_roll object
    """
    piano_roll = np.zeros(target.shape[:2])

    for i, frame in enumerate(target.T):

        for pitch_velocity in frame:
            
            piano_roll[int(pitch_velocity[0])][i] = pitch_velocity[1]
    
    return piano_roll

def melody_to_piano_roll(pitches, velocities):
    """
    Create a piano roll from a list of pitches and a list of velocities
    """
    piano_roll = np.zeros((128, pitches.shape[1]))
    
    for i, (pitch, velocity) in enumerate(zip(pitches[0], velocities[0])):
        if pitch > 0:
            piano_roll[int(pitch)][i] = velocity
            
    return piano_roll

def assemblate_accompaniment_melody(accompaniment, melody):
    """
    Assemblate accompaniment and melody, must be the same size
    """
    