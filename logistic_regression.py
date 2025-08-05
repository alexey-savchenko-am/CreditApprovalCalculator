import numpy as np
from typing import Tuple

class LogisticRegressionGD: 
    def __init__(self, learning_rate: float = 0.01, n_iter: int = 1000) -> None:
        """
        Initialize the logistic regression model.

        :param learning_rate: Step size for gradient descent
        :param n_iter: Number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        
        """
            Train the model using gradient descent.
            
            :param X: feature matrix (n_samples, n_features)
            :param y: Target vector (n_samples)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iter):
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)

            #compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for input samples.

        :param X: Feature Matrix
        :return: Probabilities between 0 and 1
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.array) -> np.array:
         """
         Predict binary class labels (0 or 1).

        :param X: Feature matrix
        :return: Predicted labels
        """
         y_pred = self.predict_proba(X)
         return (y_pred >= 0.5).astype(int)
    
    def get_params(self) -> Tuple[np.ndarray, float]:
        """
        Get learned parameters of the model.

        :return: Tuple of weights and bias
        """
        return self.weights, self.bias
        
