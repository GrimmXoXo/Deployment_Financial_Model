# Financial Model Taiwan

Financial Model Taiwan is a Python package designed to preprocess, train, and predict financial models. It includes functionalities for data ingestion, preprocessing, resampling, model training, evaluation, and optimization. This package aims to provide a comprehensive solution for financial modeling with support for various machine learning algorithms and techniques.

## Model Architecture

The following image illustrates the model architecture:

![Model Architecture](./data/docs/ModelArchitecture.png)

### Workflow Overview

1. **Data Ingestion**:
   - Load data from CSV files.
   - Split data into training and testing sets.

2. **Preprocessing Pipeline**:
   - Define preprocessing steps.
   - Handle missing values.
   - Standardize/normalize data.
   - Select important features.

3. **Resampling**:
   - Perform data resampling to handle class imbalance.

4. **Initial Model Training**:
   - Train multiple models: Random Forest, Logistic Regression, SVM, Gradient Boosting, XGBoost, KSBBoost, ANN.
   - Evaluate models based on performance metrics.
   - Select the best performing models for stacking.

5. **Optimizing Model for Performance**:
   - Stack the best models (XGBoost and Random Forest).
   - Optimize hyperparameters using Optuna.
   - Tune the stacked model.
   - Adjust thresholds to minimize type I and type II errors.
   - Increase recall and finalize the model.

## Installation

You can install the package via pip:

```sh
pip install financial_model_taiwan
```

# Usage


## Training a New Model

```python

from financial_model_taiwan.Orchestrator import FinModel

pipeline = FinModel(data_path='data/train_data.csv', target_column='target')
pipeline.data_ingestion()
pipeline.data_preprocessing()
pipeline.data_resampling()
pipeline.train_model()
pipeline.save_model('models/trained_model.bin')
evaluation_results = pipeline.evaluate_model()
print(evaluation_results)

```

## Using a Pre-trained Model
```python
from financial_model_taiwan.Orchestrator import FinModel

pipeline = FinModel(data_path='data/train_data.csv', target_column='target', model_path='models/trained_model.bin')
pipeline.data_ingestion()
pipeline.data_preprocessing()
pipeline.load_model()
evaluation_results = pipeline.evaluate_model()
print(evaluation_results)

```
