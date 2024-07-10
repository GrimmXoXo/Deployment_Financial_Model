from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from .utils import load_config

# Load the configuration
config = load_config('config.yaml')
# print(config)

threshold=config['CustomTransformers']['thresholds']['medium']
alpha=config['CustomTransformers']['alpha']



class DataFrameVarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0):
        self.threshold = threshold
        self.variance_threshold = VarianceThreshold(threshold=self.threshold)

    def fit(self, X, y=None):
        self.variance_threshold.fit(X)
        self.columns_ = X.columns[self.variance_threshold.get_support()].tolist()
        return self

    def transform(self, X):
        X_filtered = self.variance_threshold.transform(X)
        return pd.DataFrame(X_filtered, columns=self.columns_, index=X.index)


# class IndexPrinter(BaseEstimator, TransformerMixin):
#     def __init__(self, step_name):
#         self.step_name = step_name

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         print(f"Index after {self.step_name}:")
#         print(X.index)
#         return X

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        imputed_array = self.imputer.transform(X)
        return pd.DataFrame(imputed_array, columns=X.columns, index=X.index)
    

class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, columns=self.columns_, index=X.index)



class DF_log_transform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(np.log1p(X), columns=self.columns_, index=X.index)


class DF_cube_root_transform(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        return pd.DataFrame(np.cbrt(X), columns=self.columns_, index=X.index)



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=alpha, threshold=threshold):
        self.alpha = alpha
        self.threshold = threshold

    # def fit(self, X, y):
    #     self.ridge = Ridge(alpha=self.alpha)
    #     self.ridge.fit(X, y)
    #     self.coefficients = self.ridge.coef_
    #     feature_names = X.columns
    #     self.regularized_features = pd.DataFrame({
    #         'Features': feature_names,
    #         'Coef': self.coefficients
    #     })
    #     self.regularized_features = self.regularized_features.reindex(
    #         self.regularized_features['Coef'].abs().sort_values(ascending=False).index
    #     )
    #     self.selected_features_ = self.select_features(self.regularized_features)['Features'].tolist()
    #     return self
    

    def fit(self, X, y):
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X, y)
        self.coefficients = self.ridge.coef_
        feature_names = X.columns
        self.regularized_features = pd.DataFrame({
            'Features': feature_names,
            'Coef': self.coefficients
        })
        # Sort features by absolute coefficient values
        self.regularized_features = self.regularized_features.iloc[
            self.regularized_features['Coef'].abs().sort_values(ascending=False).index
        ]
        # Select features based on threshold
        self.selected_features_ = self.select_features(self.regularized_features)['Features'].tolist()
        # Ensure selected features are in the original order of X
        self.selected_features_ = [f for f in X.columns if f in self.selected_features_]
        return self

    def transform(self, X, y=None):
        return X[self.selected_features_]

    def select_features(self, df):
        return df[(df['Coef'] >= self.threshold) | (df['Coef'] <= -self.threshold)]





class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        self.lower_bounds = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_bounds = np.percentile(X, self.upper_percentile, axis=0)
        self.columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        X_capped = np.clip(X, self.lower_bounds, self.upper_bounds)
        return pd.DataFrame(X_capped, columns=self.columns_, index=X.index)
    

class DataFrameKeeper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X, index=self.index_, columns=self.columns_)
        self.index_ = X.index
        self.columns_ = X.columns
        return X

# class DataFrameVarianceThreshold(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold=0):
#         self.threshold=threshold
#         self.variance_threshold = VarianceThreshold(threshold=self.threshold)
    
#     def fit(self, X, y):
#         self.variance_threshold.fit(X)
#         return self
    
#     def transform(self, X):
#         X_filtered = self.variance_threshold.transform(X)

#         return pd.DataFrame(X_filtered, columns=X.columns[self.variance_threshold.get_support()],index=X.index)
    
# class DataFrameScaler(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.scaler = StandardScaler()
    
#     def fit(self, X, y):
#         self.scaler.fit(X)
#         return self
    
#     def transform(self, X):
#         X_scaled = self.scaler.transform(X)
#         return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
# class DF_log_transform(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         pass

#     def fit(self, X, y):
#         return self

#     def transform(self, X):
#         # Ensure output is DataFrame with original columns
#         return pd.DataFrame(np.log1p(X), columns=X.columns,index=X.index)


# class FeatureSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, alpha=alpha, threshold=threshold):
#         self.alpha = alpha
#         self.threshold = threshold
        
#     def fit(self, X, y):
#         self.ridge = Ridge(alpha=self.alpha)
#         self.ridge.fit(X, y)
#         self.coefficients = self.ridge.coef_
        
#         feature_names = X.columns
#         self.regularized_features = pd.DataFrame({
#             'Features': feature_names,
#             'Coef': self.coefficients
#         })
#         self.regularized_features = self.regularized_features.reindex(
#             self.regularized_features['Coef'].abs().sort_values(ascending=False).index
#         )
#         return self

#     def transform(self, X, y):  
#         selected_features = self.select_features(self.regularized_features)
#         selected_columns = [col for col in X.columns if col in selected_features['Features'].values]
#         return X[selected_columns]

#     def select_features(self, df):
#         return df[(df['Coef'] >= self.threshold) | (df['Coef'] <= -self.threshold)]
    

# class OutlierCapper(BaseEstimator, TransformerMixin):
#     def __init__(self, lower_percentile=1, upper_percentile=99):
#         self.lower_percentile = lower_percentile
#         self.upper_percentile = upper_percentile

#     def fit(self, X, y):
#         self.lower_bounds = np.percentile(X, self.lower_percentile, axis=0)
#         self.upper_bounds = np.percentile(X, self.upper_percentile, axis=0)
#         return self

#     def transform(self, X):
#         X_capped = np.clip(X, self.lower_bounds, self.upper_bounds)
#         return pd.DataFrame(X_capped, columns=X.columns, index=X.index)



# # # After OutlierCapper
# # assert X.index.equals(X_capped.index), "Index mismatch after OutlierCapper transformation"

# # # After DataFrameVarianceThreshold
# # assert X_capped.index.equals(X_filtered.index), "Index mismatch after DataFrameVarianceThreshold transformation"



# # # After DataFrameScaler
# # assert X_log_transformed.index.equals(X_scaled.index), "Index mismatch after DataFrameScaler transformation"

# # # After FeatureSelector (if applicable)
# # assert X_scaled.index.equals(X_selected_features.index), "Index mismatch after FeatureSelector transformation"
