from sklearn.pipeline import Pipeline
from .Custom_Transformers import (OutlierCapper, DataFrameImputer,DF_cube_root_transform, DataFrameScaler, DF_log_transform, DataFrameVarianceThreshold, FeatureSelector)
from sklearn.impute import SimpleImputer
# import os
# print(os.getcwd())
from .utils import load_config

# Load the configuration
config = load_config('config.yaml')

threshold = config['CustomTransformers']['thresholds']['medium']

class PreprocessingPipeline:
    def __init__(self):
        pass

    def preprocessing(self):
        preprocessing_pipeline_final = Pipeline(steps=[
            ('imputer', DataFrameImputer(strategy='median')), 
            ('constant_filter', DataFrameVarianceThreshold()),
            ('outlier_capper', OutlierCapper()),
            ('cube_root_transform', DF_cube_root_transform()),
            ('scaler', DataFrameScaler()), 
            ('feature_selector', FeatureSelector(threshold=threshold))
        ])
        return preprocessing_pipeline_final



# from sklearn.pipeline import Pipeline
# from Custom_Transformers import (OutlierCapper, DataFrameScaler, DF_log_transform, DataFrameVarianceThreshold, FeatureSelector,DataFrameKeeper)
# from utils import load_config
# from sklearn.base import BaseEstimator, TransformerMixin
# # Load the configuration
# config = load_config('config/config.yaml')
# threshold = config['CustomTransformers']['thresholds']['medium']



# class PreprocessingPipeline:
#     def __init__(self):
#         pass

#     def preprocessing(self):
#         preprocessing_pipeline_final = Pipeline(steps=[
#             ('df_keeper_start', DataFrameKeeper()),
#             ('outlier_capper', OutlierCapper()),
#             ('df_keeper_1', DataFrameKeeper()),
#             ('constant_filter', DataFrameVarianceThreshold()),
#             ('df_keeper_2', DataFrameKeeper()),
#             ('log_transform', DF_log_transform()),
#             ('df_keeper_3', DataFrameKeeper()),
#             ('scaler', DataFrameScaler()),
#             ('df_keeper_4', DataFrameKeeper()),
#             ('feature_selector', FeatureSelector(threshold=threshold)),
#             ('df_keeper_end', DataFrameKeeper())
#         ])
#         return preprocessing_pipeline_final
    


    # def preprocessing(self):
    #     preprocessing_pipeline_final = Pipeline(steps=[
    #         ('outlier_capper', OutlierCapper()),
    #         ('index_print_1', IndexPrinter('outlier_capper')),
    #         ('constant_filter', DataFrameVarianceThreshold()),
    #         ('index_print_2', IndexPrinter('constant_filter')),
    #         ('log_transform', DF_log_transform()),
    #         ('index_print_3', IndexPrinter('log_transform')),
    #         ('scaler', DataFrameScaler()),
    #         ('index_print_4', IndexPrinter('scaler')),
    #         ('feature_selector', FeatureSelector(threshold=threshold)),
    #         ('index_print_5', IndexPrinter('feature_selector'))
    #     ])
    #     return preprocessing_pipeline_final