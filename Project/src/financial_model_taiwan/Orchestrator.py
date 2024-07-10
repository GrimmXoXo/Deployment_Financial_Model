import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from .data_preprocessing import PreprocessingPipeline
from .Data_Resampling import Resample
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator

class FinModel:

    def __init__(self, data_path, target_column, model_path=None):
        self.data_path = data_path
        self.target_column = target_column
        self.model_path = model_path
        self.pipeline = None
        self.trainer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_transformed = None
        self.X_test_transformed = None
        self.X_resampled = None
        self.y_resampled = None

    def data_ingestion(self):
        df = pd.read_csv(self.data_path)
        self.X = df.drop(self.target_column, axis=1)
        self.y = df[self.target_column]

    def data_preprocessing(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)
        preprocess = PreprocessingPipeline()
        self.pipeline = preprocess.preprocessing()
        self.pipeline.fit(self.X_train, self.y_train)
        self.X_train_transformed = self.pipeline.transform(self.X_train)
        self.X_test_transformed = self.pipeline.transform(self.X_test)

    def data_resampling(self):
        resampler = Resample(self.X_train_transformed, self.y_train)
        self.X_resampled, self.y_resampled = resampler.resampling()

    def train_model(self, n_trials_rf=20, n_trials_xgb=20):
        self.trainer = ModelTrainer(self.X_resampled, self.y_resampled)
        self.trainer.optimize_rf(n_trials=n_trials_rf)
        self.trainer.optimize_xgb(n_trials=n_trials_xgb)
        self.trainer.train_stacked_model()

    def load_model(self):
        if self.model_path:
            self.trainer = joblib.load(self.model_path)
        else:
            raise ValueError("Model path is not provided or invalid")

    def save_model(self, output_path):
        if self.trainer:
            joblib.dump(self.trainer, output_path)
        else:
            raise ValueError("No trained model to save")

    def evaluate_model(self):
        if not self.trainer:
            raise ValueError("Model is not trained or loaded")
        evaluator = ModelEvaluator(self.trainer.stacked_model, self.X_test_transformed, self.y_test)
        return evaluator.evaluate()

# Example Usage

# pipeline = FinModel(data_path='../data/data.csv')
# pipeline.data_ingestion()
# pipeline.data_preprocessing()
# pipeline.data_resampling()
# pipeline.train_model()
# pipeline.save_model('trained_model.bin')
# evaluation_results = pipeline.evaluate_model()
# print(evaluation_results)




# If using a pre-trained model

# pipeline = FinModel(data_path='../data/data.csv', model_path='trained_model.bin')
# pipeline.data_ingestion()
# pipeline.data_preprocessing()
# pipeline.load_model()
# evaluation_results = pipeline.evaluate_model()
# print(evaluation_results)
