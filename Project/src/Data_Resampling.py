from imblearn.combine import SMOTEENN
import pandas as pd
from typing import Tuple

class Resample:
    '''
    Class for resampling the data
    '''
 
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def resampling(self) -> Tuple[pd.DataFrame, pd.Series]:  
        # Initialize SMOTEENN
        smote_enn = SMOTEENN()

        # Apply SMOTEENN to the processed training data
        X_resampled, y_resampled = smote_enn.fit_resample(self.X, self.y)
        return X_resampled, y_resampled
