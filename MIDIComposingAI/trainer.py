import joblib
import numpy as np
from google.cloud import storage
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from MIDIComposingAI.data import load_result
from MIDIComposingAI.encoders import adding_chords_info

MODEL_NAME = "MIDIComposingAI"
MODEL_VERSION = "3"
BUCKET_NAME = "wagon-data-770-midi-project"

STORAGE_LOCATION = 'models/MIDIComposingAI/2/model.joblib'

class Trainer(object):
    def __init__(self, X, y):
        """
            X: numpy array
            y: numpy array
        """
        self.model = None
        self.X = X
        self.y = y

    def preprocess(self):
        chord = adding_chords_info(self.X)
        self.X = self.X.reshape((self.X.shape[0], -1))
        self.X = np.concatenate((chord, self.X), axis=1)
        self.y = self.y.reshape((self.y.shape[0], -1))

    def train_model(self):
        """method that trains the model"""
        self.model = DecisionTreeRegressor()
        self.model.fit(self.X, self.y)
        print("trained model")
        return self.model

    def save_model(self):
        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.model, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.__upload_model_to_gcp()
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")

    def __upload_model_to_gcp(self):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

if __name__ == "__main__":
    X, y = load_result(reload_midi=True)
    print("Loading done")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2)
    trainer = Trainer(X=X_train, y=y_train)
    trainer.preprocess()
    trainer.train_model()
    trainer.save_model()
