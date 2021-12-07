import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

def pattern_recognition(acc, mel):
    """
    Create a pattern of the evolution of relative pitch and rythm from accompaniment and melody
    """
    # Create list of pitches for each frame in accompaniment's piano roll
    acc_pitches = []
    
    for frame in acc.T:
        pitches = []
        for vel in frame:
            if vel > 0:
                pitches.append(list(frame).index(vel))
        if not pitches:
            pitches.append(0)
        acc_pitches.append(pitches)
    
    # Take the mean pitch of each frame
    mean_pitches_acc = [(np.sum(pitches) / len(pitches)) for pitches in acc_pitches]
    
    # Get only the pitches (not the velocities)
    melody_pitches = mel[:500]
    
    
    # Create the pattern for accompaniment
    acc_passed_pitch = []
    pitch_pattern_acc = [0]
    
    # Do we start with a note played or not ?
    if mean_pitches_acc[0] > 0:
        rythm_pattern_acc = [1]
    else:
        rythm_pattern_acc = [0]
        
    for pitch in mean_pitches_acc:
        if acc_passed_pitch:
            if pitch != acc_passed_pitch[-1]:
                relative_pitch = pitch - acc_passed_pitch[-1]
                pitch_pattern_acc.append(relative_pitch)
                rythm_pattern_acc.append(1)
            else:
                pitch_pattern_acc.append(pitch_pattern_acc[-1])
                rythm_pattern_acc.append(0)
        acc_passed_pitch.append(pitch)
    
    # Create the pattern for melody
    mel_passed_pitch = []
    pitch_pattern_mel = [0]
    
    # Do we start with a note played or not ?
    if melody_pitches[0] > 0:
        rythm_pattern_mel = [1]
    else:
        rythm_pattern_mel = [0]
        
    for pitch in melody_pitches:
        if mel_passed_pitch:
            if pitch != mel_passed_pitch[-1]:
                relative_pitch = pitch - mel_passed_pitch[-1]
                pitch_pattern_mel.append(relative_pitch)
                rythm_pattern_mel.append(1)
            else:
                pitch_pattern_mel.append(pitch_pattern_mel[-1])
                rythm_pattern_mel.append(0)
        mel_passed_pitch.append(pitch)

    return {
        'rythm_pattern':{'acc':rythm_pattern_mel,'mel':rythm_pattern_mel},
        'pitch_pattern':{'acc':pitch_pattern_mel,'mel':pitch_pattern_mel}
           }    
    
def custom_metric(accompaniment, predicted_melody, weight=0.5):
    """
    TODO : Doc string
    """
    # Get the pitch and rythm pattern for both accompaniment and melody
    patterns = pattern_recognition(accompaniment, predicted_melody)
    pitch_pattern_acc = patterns['pitch_pattern']['acc']
    pitch_pattern_mel = patterns['pitch_pattern']['mel']
    rythm_pattern_acc = patterns['rythm_pattern']['acc']
    rythm_pattern_mel = patterns['rythm_pattern']['mel']

    # Compute the mean of velocities for both accompaniment and melody
    list_mean_vel_acc = [
        np.mean(
            [vel for vel in frame if vel > 0]
        )
        for frame in accompaniment.T
        if np.sum(frame) > 0]

    # Check if the list is empty
    if not list_mean_vel_acc:
        list_mean_vel_acc = [0]

    mean_vel_acc = np.mean(list_mean_vel_acc)

    list_mean_vel_pred = [pred for pred in predicted_melody[500:] if pred > 0]

    # Check if the list is empty
    if not list_mean_vel_pred:
        list_mean_vel_pred = [0]
 
    mean_vel_pred = np.mean(list_mean_vel_pred)

    # Compute the diff beetween the two velocities mean
    velocity_diff = abs(mean_vel_acc - mean_vel_pred)

    # Compute the "diff pattern" beetween accompaniment and melody
    diff_pitch_pattern = np.array([abs(acc - mel) for acc, mel in zip(pitch_pattern_acc, pitch_pattern_mel)]).reshape(-1, 1)

    # Compute the entropy of the diff pattern
    if np.sum(diff_pitch_pattern) == 0:
        pitch_entropy_score = 0
    else:
        pitch_entropy_score = entropy(diff_pitch_pattern)[0]

    # Now we do the same for rythm pattern
    diff_rythm_pattern = np.array([acc + mel for acc, mel in zip(rythm_pattern_acc, rythm_pattern_mel)]).reshape(-1, 1)
        
    # Compute the entropy of the diff pattern
    if np.sum(diff_rythm_pattern) == 0:
        rythm_entropy_score = 0
    else:
        rythm_entropy_score = entropy(diff_rythm_pattern)[0]
        
    # Compute the "note's length score" of the melody
    notes_length_score = rythm_pattern_mel.count(1)
    
    # Compute the final score
    entropy_score = pitch_entropy_score + rythm_entropy_score

    return np.mean([entropy_score, velocity_diff, notes_length_score])

def custom_grid_search(X, y, params):
    """
    A custom grid search to use with our custom metric
    """
    params_and_scores = {}

    for split in range(5):
        X_train, X_test, X_reshaped_train, X_reshaped_test, y_train, y_test = train_test_split(X, X_reshaped, y, test_size=0.1)
        params_and_scores[f'split_{split}'] = []
        for i, param in enumerate(params):

            tree = DecisionTreeRegressor(**param)
            tree.fit(X_reshaped_train, y_train)
            predictions = tree.predict(X_reshaped_test)
            scores = [custom_metric(test, pred) for test, pred in zip(X_test, predictions)]
            score = np.mean(scores)
            params_and_scores[f'split_{split}'].append({'params':param, 'score':score})
        print(f'split {split+1} done')