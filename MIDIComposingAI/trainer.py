import joblib
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from MIDIComposingAI.data import load_result
from MIDIComposingAI.encoders import adding_chords_info

MODEL_NAME = "MIDIComposingAI"
MODEL_VERSION = "1"

class Trainer(object):
    def __init__(self, X, y):
        """
            X: numpy array
            y: numpy array
        """
        self.X = X
        self.y = y

    def preprocess(self):
        chord = adding_chords_info(self.X)
        self.X = self.X.reshape((self.X.shape[0], -1))
        self.X = np.concatenate((chord, self.X), axis=1, dtype=np.int8)
        self.y = self.y.reshape((self.y.shape[0], -1))

    def train_model(self):
        """method that trains the model"""
        tree = DecisionTreeRegressor()
        tree.fit(self.X, self.y)
        print("trained model")
        return tree

    def save_model_to_gcp(model):
        local_model_name = 'model.joblib'
        # saving the trained model to disk (which does not really make sense
        # if we are running this code on GCP, because then this file cannot be accessed once the code finished its execution)
        joblib.dump(model, local_model_name)
        print("saved model.joblib locally")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = bucket.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

if __name__ == "__main__":
    X, y = load_result()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=2)
    trainer = Trainer(X=X_train, y=y_train)
    trainer.preprocess()
    trainer.train_model()
