# logistic.py
# Manual implementation of Logistic Regression

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionManual:
    """
    Manual implementation of Logistic Regression using Gradient Descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for gradient descent updates
    n_iterations : int, default=1000
        Number of training iterations
    verbose : bool, default=True
        Whether to print training progress
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_loss_history = []
        
    def sigmoid(self, z):
        """
        Sigmoid activation function.
        
        Parameters:
        -----------
        z : numpy array
            Linear combination of inputs and weights
            
        Returns:
        --------
        numpy array : Probabilities between 0 and 1
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Parameters:
        -----------
        y_true : numpy array
            True labels (0 or 1)
        y_pred : numpy array
            Predicted probabilities
            
        Returns:
        --------
        float : Binary cross-entropy loss
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train logistic regression using gradient descent.
        
        Parameters:
        -----------
        X_train : numpy array of shape (n_samples, n_features)
            Training features
        y_train : numpy array of shape (n_samples,)
            Training labels
        X_val : numpy array, optional
            Validation features
        y_val : numpy array, optional
            Validation labels
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples, n_features = X_train.shape
        
        # Initialize parameters to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        print(f"\n    Starting training...")
        print(f"    Learning rate: {self.learning_rate}")
        print(f"    Iterations: {self.n_iterations}")
        print(f"    Training samples: {n_samples}")
        print(f"    Features: {n_features}")
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X_train, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute training loss
            train_loss = self.compute_loss(y_train, y_pred)
            self.loss_history.append(train_loss)
            
            # Compute validation loss if provided
            if X_val is not None and y_val is not None:
                val_pred = self.predict_proba(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                self.val_loss_history.append(val_loss)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_train.T, (y_pred - y_train))
            db = (1 / n_samples) * np.sum(y_pred - y_train)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if self.verbose and (i + 1) % 100 == 0:
                if X_val is not None:
                    print(f"    Iteration {i+1}/{self.n_iterations} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"    Iteration {i+1}/{self.n_iterations} - Train Loss: {train_loss:.4f}")
        
        print(f"    Training completed! Final loss: {self.loss_history[-1]:.4f}")
        return self
    
    def predict_proba(self, X):
        """
        Predict probability of class 1.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        numpy array : Probabilities of class 1
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Input features
        threshold : float, default=0.5
            Decision threshold for classification
            
        Returns:
        --------
        numpy array : Predicted labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_loss_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss', linewidth=2)
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='Validation Loss', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Binary Cross-Entropy Loss')
        plt.title('Loss Curves During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_parameters(self):
        """
        Get model parameters.
        
        Returns:
        --------
        tuple : (weights, bias)
        """
        return self.weights.copy(), self.bias
    
    def save_model(self, filepath):
        """
        Save model parameters to file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'learning_rate': self.learning_rate,
                'n_iterations': self.n_iterations,
                'loss_history': self.loss_history,
                'val_loss_history': self.val_loss_history
            }, f)
        print(f"    Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model parameters from file.
        
        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        import pickle
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.weights = params['weights']
        self.bias = params['bias']
        self.learning_rate = params['learning_rate']
        self.n_iterations = params['n_iterations']
        self.loss_history = params['loss_history']
        self.val_loss_history = params['val_loss_history']
        print(f"    Model loaded from {filepath}")


# For testing the logistic regression module independently
if __name__ == "__main__":
    print("Testing Logistic Regression Module...")
    
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = (np.random.rand(100) > 0.5).astype(int)
    X_test = np.random.randn(20, 10)
    
    # Train model
    model = LogisticRegressionManual(learning_rate=0.1, n_iterations=500, verbose=True)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    print(f"\n    Predictions shape: {y_pred.shape}")
    print(f"    Unique predictions: {np.unique(y_pred)}")
    
    # Plot loss curve
    model.plot_loss_curves()
    
    print("\n✅ Logistic Regression module working correctly!")