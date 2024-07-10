from setuptools import setup, find_packages

setup(
    name='financial_model_taiwan',
    version='1.0.2',
    description='A financial model package trained on data from Taiwan fiscal year 1999-2009',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Gaurav M',
    author_email='grimmxoxo2001@example.com',
    url='https://github.com/GrimmXoXo/Deployment_Financial_Model',
    # packages=find_packages(include=['src', 'src.*']),
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'financial_model_taiwan': ['config/*.yaml']},
    include_package_data=True,
    install_requires=[
        'pandas',
        'numpy',
        'joblib',
        'optuna',
        'pathlib',
        'imblearn',
        'typing',
        'xgboost',
        'pytest',
        'scikit-learn'
        # Add any other dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
