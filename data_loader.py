import pandas as pd
from pandas import DataFrame
from typing import Tuple

class CreditDataset:
    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.df: DataFrame | None = None
    
    def load(self) -> None:
        """Loads the dataset from CSV file."""
        self.df = pd.read_csv(self.filepath)
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
    
    def head(self, n: int = 5) -> DataFrame: 
        """Returns the first n rows of the dataset."""
        if self.df is not None:
            return self.df.head(n)
        else:
            raise ValueError("Dataset not loaded. Call .load() first.")
        
    def describe(self) -> DataFrame:
        """Returns summary statistics of the dataset."""
        if self.df is not None:
            return self.df.describe()
        else:
            raise ValueError("Dataset not loaded. Call .load() first.")

    def get_features_and_target(self) -> Tuple[DataFrame, pd.Series]:
        """Returns feature matrix X and target vector y."""
        if self.df is not None:
            X = self.df.drop(columns=["approved"])
            y = self.df["approved"]

            # Normalize features using Z-score: (x - mean) / std
            X = (X - X.mean()) / X.std()

            return X, y
        else:
            raise ValueError("Dataset not loaded. Call .load() first.")

