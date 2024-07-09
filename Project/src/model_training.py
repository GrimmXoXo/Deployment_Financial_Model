import optuna
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np


# Suppress Optuna INFO messages
# optuna.logging.set_verbosity(optuna.logging.WARNING)

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.best_rf_params = None
        self.best_xgb_params = None
        self.stacked_model = None



    def rf_objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        max_depth = trial.suggest_int('max_depth', 10, 20)
        threshold = trial.suggest_float('threshold', 0.1, 0.9)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(self.X_train, self.y_train):
            X_train_cv, X_val_cv = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_cv, y_val_cv = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_prob = model.predict_proba(X_val_cv)[:, 1]
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            f1 = f1_score(y_val_cv, y_pred)
            scores.append(f1)
        
        return np.mean(scores)

    def optimize_rf(self, n_trials=50):
        rf_study = optuna.create_study(direction='maximize')
        rf_study.optimize(self.rf_objective, n_trials=n_trials)
        self.best_rf_params = rf_study.best_params




    def xgb_objective(self, trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0,log=True)
        threshold = trial.suggest_float('threshold', 0.1, 0.9)
        
        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(self.X_train, self.y_train):
            X_train_cv, X_val_cv = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_train_cv, y_val_cv = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_prob = model.predict_proba(X_val_cv)[:, 1]
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            f1 = f1_score(y_val_cv, y_pred)
            scores.append(f1)
        
        return np.mean(scores)

    def optimize_xgb(self, n_trials=50):
        xgb_study = optuna.create_study(direction='maximize')
        xgb_study.optimize(self.xgb_objective, n_trials=n_trials)
        self.best_xgb_params = xgb_study.best_params




    def train_stacked_model(self):
        rf_model = RandomForestClassifier(
            n_estimators=self.best_rf_params['n_estimators'],
            max_depth=self.best_rf_params['max_depth'],
            random_state=42
        )

        xgb_model = XGBClassifier(
            n_estimators=self.best_xgb_params['n_estimators'],
            learning_rate=self.best_xgb_params['learning_rate'],
            random_state=42
        )

        self.stacked_model = StackingClassifier(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            final_estimator=LogisticRegression()
        )

        self.stacked_model.fit(self.X_train, self.y_train)

    # def evaluate_model(self):
    #     if self.stacked_model is None:
    #         raise Exception("The stacked model has not been trained yet.")
    #     y_pred = self.stacked_model.predict(self.X_test)
    #     f1 = f1_score(self.y_test, y_pred)
    #     return f1

# Usage
# Assuming X_train, y_train, X_test, y_test are defined
# trainer = ModelTrainer(X_train, y_train, X_test, y_test)
# trainer.optimize_rf(n_trials=50)
# trainer.optimize_xgb(n_trials=50)
# trainer.train_stacked_model()
# f1_score = trainer.evaluate_model()
# print(f"F1 Score: {f1_score}")
