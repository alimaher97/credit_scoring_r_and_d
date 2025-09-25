from typing import Protocol, Any, Type, Self
import numpy as np
import pandas as pd

class Estimator(Protocol):
    """
    A Protocol defining the required interface for a machine learning model.
    """
    def __init__(self):
        pass
    
    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> Self:
        """Trains the model on the provided data."""
        ...

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.Series:
        """Generates predictions for the provided data."""
        ...
    
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Generates probability estimates for the provided data."""
        ...

    def save(self, filepath: str) -> None:
        """Saves the trained model to a file."""
        ...

    @classmethod
    def load(cls: Type[Self], filepath: str) -> Self:
        """Loads a trained model from a file."""
        ...
    