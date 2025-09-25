from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

class Evaluator:
    def __init__(self):
        pass
    

    def evaluate_model(model, X_val, y_val):
        """Evaluate model performance on validation set"""
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Predict classes
        y_pred = model.predict(X_val)
        
        # Calculate AUC
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        # Additional metrics
        print(f"\n{'='*50}")
        print(f"Validation AUC Score: {auc_score:.4f}")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        
        return auc_score, y_pred_proba
    
def evaluate_feature_importance(model, feature_names):
    """Evaluate and print feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        # See which features are most important (sorted)
        sorted_indices = np.argsort(feature_importance)[::-1]  # Descending order
        print("\nTop 5 most important features:")
        for i, idx in enumerate(sorted_indices[:5]):
            print(f"Feature {idx}: {feature_importance[idx]:.4f}")

        # Create a DataFrame for better visualization (if you have feature names)
        feature_names = feature_names
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        print("\nFeature importance DataFrame:")
        print(importance_df.head())                                                              