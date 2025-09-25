from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np
from estimator import Estimator

class Trainer:
    def __init__(self, X_train, y_train, X_val, y_val, model: Estimator, param_distributions: dict):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = model
        self.param_distributions = param_distributions
    
    def train(self):
        """Train model with hyperparameter tuning with different combinations using random search."""
        best_auc_score = 0
        
        # Set up RandomizedSearchCV
        self.model = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_distributions,
            n_iter=50,  # Number of parameter combinations to try
            cv=5,       # 5-fold cross-validation
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,  # Use all available cores
            verbose=1
        )
            
        self.model.fit(self.X_train, self.y_train)
        y_pred_proba = self.model.predict_proba(self.X_val)[:, 1]
        best_auc_score = roc_auc_score(self.y_val, y_pred_proba)
        
        print(f"Best Params: {self.model.best_params_} -> Best AUC: {best_auc_score:.4f}")
        
        return self.model