import pretty_midi
import glob
import joblib

# Get your path to adl-piano-midi dataset
# See README
path_to_dataset = 'YOUR/PATH/TO/DATASET'

# Get all paths to files inside the dataset
targetPattern = fr"{{path_to_dataset}}/*/*/*/*/*/*.mid"
midi_files_paths = glob.glob(targetPattern)

# Store the pretty_midi dataset
path_to_pretty_midi_dir = "YOUR/PATH/TO/DIR"
for file in midi_files_paths:
    try: # We chose to ignore the few files not working with pretty_midi
        joblib.dump(pretty_midi.PrettyMIDI(file), f'{path_to_pretty_midi_dir}/{file.split("/")[-1][:-4]}')
    except:
        pass
