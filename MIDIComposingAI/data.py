from google.cloud import storage
from MIDIComposingAI.create_dataset import create_simple_dataset
import numpy as np
import pandas as pd
import joblib
import gcsfs

PROJECT_ID='wagon-bootcamp-328620'
BUCKET_NAME = "wagon-data-770-midi-project"
RESULT_DATA_PATH = "result"
BUCKET_CHORDS_DATA = "data/chords_midi.csv"

def get_data_from_gcp(**kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # ---- CREATE CLIENT & BUCKET FOR GCP ----
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(client.list_blobs(bucket, prefix='data/pretty_midi/'))
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID, token='cloud')
    # ---- GET PRETTY MIDI FILES & CONCATENATE THEM AS ONE TUPLE OF NP ARRAYS ----
    with fs.open(f'{BUCKET_NAME}/{blobs[0].name}') as f:
        file = joblib.load(f)
    X, y = create_simple_dataset(file)
    for blob in blobs[1:]:
        try:
            with fs.open(f'{BUCKET_NAME}/{blob.name}') as f:
                file = joblib.load(f)
            loaded = create_simple_dataset(file)
            X = np.concatenate((X, loaded[0]))
            y = np.concatenate((y, loaded[1]))
        except:
            pass
    return (X, y)

def load_result(reload_midi=False):
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID, token='cloud')
    # ---- RELOAD RESULT FROM PRETTY MIDI FILES ----
    if reload_midi:
        result = get_data_from_gcp()
        with fs.open(f'{BUCKET_NAME}/{RESULT_DATA_PATH}', 'wb') as f:
            joblib.dump(result, f)
    # ---- LOAD RESULT FROM GCP ----
    else:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(RESULT_DATA_PATH)
        with fs.open(f'{BUCKET_NAME}/{blob.name}') as f:
            result = joblib.load(f)
    return result

def get_chords_data(local=False):
    """method to get the chords data from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        path = "raw_data/chords_midi.csv"
    else:
        path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_CHORDS_DATA)
    df = pd.read_csv(path, sep=";")
    return df
