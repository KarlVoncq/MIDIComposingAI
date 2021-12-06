import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        preproc_pipe = # PIPELINE_TO_DEFINE

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('tree_model', DecisionTreeRegressor())
        ])

    def run(self):
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == "__main__":
    # TO DO
