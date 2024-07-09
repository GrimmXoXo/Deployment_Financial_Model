import os

# Directory structure
project_structure = {
    "config": ["config.yaml"],
    "data": ["raw", "processed"],
    "models": ["model.pkl", "preprocessing_pipeline.pkl"],
    "src": [
        "__init__.py",
        "data_preprocessing.py",
        "feature_engineering.py",
        "model_training.py",
        "model_evaluation.py",
        "Custom_Transformers.py",
        "Model_arch.py",
    ],
    "notebooks": ["exploratory_data_analysis.ipynb"],
    "tests": [
        "test_data_preprocessing.py",
        "test_feature_engineering.py",
        "test_model_training.py",
    ],
}

# Create directories and files
for directory, files in project_structure.items():
    os.makedirs(directory, exist_ok=True)
    for file in files:
        open(os.path.join(directory, file), 'a').close()

# Create remaining files
open('Dockerfile', 'a').close()
open('requirements.txt', 'a').close()
open('README.md', 'a').close()

print("Project structure created successfully.")
