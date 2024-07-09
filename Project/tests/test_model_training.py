import pytest
import pandas as pd
from src.Orchestrator import FinModel  

@pytest.fixture
def pipeline():
    data_path = 'data/test.csv'  # Adjust based on your data location
    target_column = 'Bankrupt?'
    return FinModel(data_path, target_column,model_path='test_model.bin')

def test_data_ingestion(pipeline):
    pipeline.data_ingestion()
    assert not pipeline.X.empty
    assert not pipeline.y.empty
    assert 'Bankrupt?' not in pipeline.X.columns
    assert pipeline.y.name == 'Bankrupt?'

def test_data_preprocessing(pipeline):
    pipeline.data_ingestion()
    pipeline.data_preprocessing()
    assert pipeline.X_train is not None
    assert pipeline.X_test is not None
    assert pipeline.y_train is not None
    assert pipeline.y_test is not None
    assert pipeline.X_train_transformed is not None
    assert pipeline.X_test_transformed is not None

def test_data_resampling(pipeline):
    pipeline.data_ingestion()
    pipeline.data_preprocessing()
    pipeline.data_resampling()
    assert pipeline.X_resampled is not None
    assert pipeline.y_resampled is not None

def test_train_model(pipeline):
    pipeline.data_ingestion()
    pipeline.data_preprocessing()
    pipeline.data_resampling()
    pipeline.train_model(n_trials_rf=2, n_trials_xgb=2)
    assert pipeline.trainer is not None

def test_load_model(pipeline):
    pipeline.data_ingestion()
    pipeline.data_preprocessing()
    pipeline.data_resampling()
    pipeline.train_model(n_trials_rf=2, n_trials_xgb=2)
    pipeline.save_model('test_model.bin')
    pipeline.load_model()
    assert pipeline.trainer is not None

def test_evaluate_model(pipeline):
    pipeline.data_ingestion()
    pipeline.data_preprocessing()
    pipeline.data_resampling()
    pipeline.train_model(n_trials_rf=2, n_trials_xgb=2)
    results = pipeline.evaluate_model()
    assert results is not None

if __name__ == '__main__':
    pytest.main()
