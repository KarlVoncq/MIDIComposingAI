import pretty_midi
import glob
import joblib

# Get your path to adl-piano-midi dataset
# See README

def store_data(path_to_adl_dataset, path_to_pretty_midi_dir):

    # Get all paths to files inside the dataset
    targetPattern = fr"{path_to_adl_dataset}/*/*/*/*/*.mid"
    midi_files_paths = glob.glob(targetPattern)

    # Store the pretty_midi dataset
    for file in midi_files_paths:
        try: # We chose to ignore the few files not working with pretty_midi
            joblib.dump(pretty_midi.PrettyMIDI(file), f'{path_to_pretty_midi_dir}/{file.split("/")[-1][:-4]}')
        except:
            pass
