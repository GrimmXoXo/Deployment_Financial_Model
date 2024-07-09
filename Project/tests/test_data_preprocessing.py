import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import PreprocessingPipeline

@pytest.fixture(scope="module")
def preprocessed_data():
    np.random.seed(42)
    X = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(0, 1, 100),
        'C': np.random.normal(0, 1, 100),
        'D': np.random.normal(0, 1, 100),
        'E': list([1]*100)
    })
    X.loc[X.sample(10).index, 'B'] = np.nan
    y = np.random.randint(0, 2, 100)
    
    pipeline = PreprocessingPipeline().preprocessing()
    pipeline.fit(X, y)
    X_transformed = pipeline.transform(X)
    
    return X, X_transformed



def test_output_is_dataframe(preprocessed_data):
    _, X_transformed = preprocessed_data
    assert isinstance(X_transformed, pd.DataFrame), "Output should be a DataFrame"

def test_row_count_preserved(preprocessed_data):
    X, X_transformed = preprocessed_data
    assert X_transformed.shape[0] == X.shape[0], "Number of rows should be preserved"

def test_index_preserved(preprocessed_data):
    X, X_transformed = preprocessed_data
    assert (X_transformed.index == X.index).all(), "Index should be preserved"

def test_constant_columns_removed(preprocessed_data):
    _, X_transformed = preprocessed_data
    assert 'E' not in X_transformed.columns, "Constant columns should be removed"

def test_data_scaled(preprocessed_data):
    _, X_transformed = preprocessed_data
    assert -1 <= X_transformed.mean().mean() <= 1, "Data should be scaled (mean check)"
    assert 0 <= X_transformed.std().mean() <= 2, "Data should be scaled (std check)"

def test_feature_selection(preprocessed_data):
    X, X_transformed = preprocessed_data
    assert X_transformed.shape[1] < X.shape[1], "Number of features should be reduced"

if __name__ == "__main__":
    pytest.main()