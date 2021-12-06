import pandas as pd
from google.cloud import storage
from MIDIComposingAI.create_dataset import create_nparray_dataset, extract_accompaniment_melody
import joblib
import numpy as np

PROJECT_ID='wagon-bootcamp-328620'
BUCKET_NAME = "wagon-data-770-midi-project"
BUCKET_TRAIN_DATA_PATH = "data/abcd"

import pandas as pd
import gcsfs

def get_data_from_gcp(optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    # blob = bucket.blob(BUCKET_TRAIN_DATA_PATH)
    blobs=list(client.list_blobs(bucket))
    # blobs=bucket.list_blobs() # DEPRECATED
    fs = gcsfs.GCSFileSystem(project=PROJECT_ID)
    with fs.open(f'{BUCKET_NAME}/{blobs[1].name}') as f:
        file = joblib.load(f)
    X, y = create_nparray_dataset(file, 'data' ,'abcd', store=False)
    for blob in blobs[2:]:
        # if(not blob.name.endswith("/")): #no longer needed cause we change it to list
        with fs.open(f'{BUCKET_NAME}/{blob.name}') as f:
            file = joblib.load(f)
        loaded = create_nparray_dataset(file, 'data' ,'abcd', store=False)
        X = np.concatenate((X, loaded[0]))
        y = np.concatenate((y, loaded[1]))
    print(X.shape, y.shape)
    return X, y

# p.map(process_file, listdir(input))
# List blobs iterate in folder
