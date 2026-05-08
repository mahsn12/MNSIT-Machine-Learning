# perceptron.py
# Phase 2: Perceptron algorithm for 10-class MNIST classification
# Using One-vs-All strategy with Perceptron criterion

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time


class PerceptronClassifier:
    """
    Perceptron algorithm for multi-class classification using One-vs-All strategy.
    
    The Perceptron criterion minimizes the misclassification error:
        Loss = max(0, -y * (w·x + b))
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Step size for weight updates
    n_iterations : int, default=1000
        Number of training iterations
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=True
        Print training progress
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.verbose = verbose
        self.weights = None  # Shape: (n_classes, n_features)
        self.bias = None     # Shape: (n_classes,)
        self.n_classes = None
        self.n_features = None
        self.loss_history = []
        self.val_acc_history = []
        self.train_acc_history = []
        
    def _initialize_weights(self, n_features, n_classes):
        """Initialize weights with small random values."""
        np.random.seed(self.random_state)
        self.weights = np.random.randn(n_classes, n_features) * 0.01
        self.bias = np.zeros(n_classes)
        
    def _perceptron_loss(self, X, y, class_idx):
        """
        Compute Perceptron loss for a single class (One-vs-All).
        
        Loss = max(0, -y_true * (w·x + b))
        where y_true is +1 for target class, -1 for others
        """
        # Convert to binary labels: +1 for target class, -1 for others
        y_binary = np.where(y == class_idx, 1, -1)
        
        # Compute linear scores
        scores = np.dot(X, self.weights[class_idx]) + self.bias[class_idx]
        
        # Perceptron loss: max(0, -y * score)
        loss = np.maximum(0, -y_binary * scores)
        
        return np.mean(loss)
    
    def _compute_gradients(self, X, y, class_idx):
        """
        Compute gradients for Perceptron.
        
        For misclassified samples (y * score <= 0):
            dw = -y * x
            db = -y
        For correctly classified samples:
            dw = 0, db = 0
        """
        # Convert to binary labels
        y_binary = np.where(y == class_idx, 1, -1)
        
        # Compute scores
        scores = np.dot(X, self.weights[class_idx]) + self.bias[class_idx]
        
        # Find misclassified samples (where y * score <= 0)
        misclassified = y_binary * scores <= 0
        
        # Initialize gradients
        dw = np.zeros(self.n_features)
        db = 0.0
        
        # Update only for misclassified samples
        if np.any(misclassified):
            X_mis = X[misclassified]
            y_mis = y_binary[misclassified]
            
            dw = -np.dot(y_mis, X_mis) / len(y_mis)
            db = -np.mean(y_mis)
        
        return dw, db
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train Perceptron classifier using One-vs-All strategy.
        
        Parameters:
        -----------
        X_train : numpy array of shape (n_samples, n_features)
            Training features
        y_train : numpy array of shape (n_samples,)
            Training labels (0-9)
        X_val : numpy array, optional
            Validation features
        y_val : numpy array, optional
            Validation labels
        """
        n_samples, self.n_features = X_train.shape
        self.n_classes = len(np.unique(y_train))
        
        print(f"\n{'='*60}")
        print(f"PERCEPTRON CLASSIFIER - One-vs-All Strategy")
        print(f"{'='*60}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Features: {self.n_features}")
        print(f"  Training samples: {n_samples}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.n_iterations}")
        print(f"{'='*60}")
        
        # Initialize weights
        self._initialize_weights(self.n_features, self.n_classes)
        
        # Training loop
        for epoch in range(self.n_iterations):
            epoch_loss = 0
            
            # Train each class separately (One-vs-All)
            for class_idx in range(self.n_classes):
                # Compute gradients for this class
                dw, db = self._compute_gradients(X_train, y_train, class_idx)
                
                # Update weights and bias
                self.weights[class_idx] -= self.learning_rate * dw
                self.bias[class_idx] -= self.learning_rate * db
                
                # Calculate loss
                loss = self._perceptron_loss(X_train, y_train, class_idx)
                epoch_loss += loss
            
            # Average loss across classes
            avg_loss = epoch_loss / self.n_classes
            self.loss_history.append(avg_loss)
            
            # Calculate training accuracy
            y_train_pred = self.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            self.train_acc_history.append(train_acc)
            
            # Calculate validation accuracy
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)
                self.val_acc_history.append(val_acc)
            
            # Print progress
            if self.verbose and (epoch + 1) % 100 == 0:
                if X_val is not None:
                    print(f"  Epoch {epoch+1}/{self.n_iterations} | Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                else:
                    print(f"  Epoch {epoch+1}/{self.n_iterations} | Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f}")
        
        print(f"\nTraining completed!")
        print(f"  Final training accuracy: {train_acc:.4f}")
        if X_val is not None:
            print(f"  Final validation accuracy: {val_acc:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for input data.
        
        For One-vs-All, the predicted class is the one with highest score.
        """
        scores = np.dot(X, self.weights.T) + self.bias
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X):
        """
        Get decision function values (not true probabilities).
        Higher values indicate more confidence.
        """
        return np.dot(X, self.weights.T) + self.bias
    
    def plot_training_curves(self):
        """Plot training loss and accuracy curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(self.loss_history, 'b-', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Perceptron Loss')
        axes[0].set_title('Training Loss Curve')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(self.train_acc_history, 'g-', linewidth=2, label='Training')
        if self.val_acc_history:
            axes[1].plot(self.val_acc_history, 'r-', linewidth=2, label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Perceptron Training Progress', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        classes = [str(i) for i in range(self.n_classes)]
        plt.xticks(np.arange(self.n_classes), classes)
        plt.yticks(np.arange(self.n_classes), classes)
        
        # Add text annotations
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.title('Confusion Matrix - Perceptron Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """Get feature importance (absolute weights averaged across classes)."""
        importance = np.mean(np.abs(self.weights), axis=0)
        return importance
    
    def visualize_weights(self, feature_dim=100):
        """Visualize learned weights for each digit."""
        if feature_dim > 100:
            feature_dim = 100
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        for i in range(self.n_classes):
            row = i // 5
            col = i % 5
            # Reshape weights to original image size approximation
            weight_img = self.weights[i][:feature_dim].reshape(10, 10)
            axes[row, col].imshow(weight_img, cmap='RdBu', interpolation='nearest')
            axes[row, col].set_title(f'Digit {i} Weights')
            axes[row, col].axis('off')
        
        plt.suptitle('Perceptron Weights Visualization (First 100 features)')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='perceptron_model.pkl'):
        """Save model parameters."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'bias': self.bias,
                'n_classes': self.n_classes,
                'n_features': self.n_features,
                'learning_rate': self.learning_rate,
                'n_iterations': self.n_iterations,
                'loss_history': self.loss_history,
                'train_acc_history': self.train_acc_history,
                'val_acc_history': self.val_acc_history
            }, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='perceptron_model.pkl'):
        """Load model parameters."""
        import pickle
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.weights = params['weights']
        self.bias = params['bias']
        self.n_classes = params['n_classes']
        self.n_features = params['n_features']
        self.learning_rate = params['learning_rate']
        self.n_iterations = params['n_iterations']
        self.loss_history = params.get('loss_history', [])
        self.train_acc_history = params.get('train_acc_history', [])
        self.val_acc_history = params.get('val_acc_history', [])
        print(f"Model loaded from {filepath}")


class AveragedPerceptron(PerceptronClassifier):
    """
    Averaged Perceptron: Keeps average of weights across all iterations.
    Often performs better than standard Perceptron.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42, verbose=True):
        super().__init__(learning_rate, n_iterations, random_state, verbose)
        self.avg_weights = None
        self.avg_bias = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train Averaged Perceptron."""
        n_samples, self.n_features = X_train.shape
        self.n_classes = len(np.unique(y_train))
        
        print(f"\n{'='*60}")
        print(f"AVERAGED PERCEPTRON - One-vs-All Strategy")
        print(f"{'='*60}")
        print(f"  Classes: {self.n_classes}")
        print(f"  Features: {self.n_features}")
        print(f"  Training samples: {n_samples}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Iterations: {self.n_iterations}")
        print(f"{'='*60}")
        
        # Initialize weights
        self._initialize_weights(self.n_features, self.n_classes)
        self.avg_weights = np.zeros_like(self.weights)
        self.avg_bias = np.zeros_like(self.bias)
        
        # Training loop
        for epoch in range(self.n_iterations):
            epoch_loss = 0
            
            for class_idx in range(self.n_classes):
                dw, db = self._compute_gradients(X_train, y_train, class_idx)
                
                # Update weights
                self.weights[class_idx] -= self.learning_rate * dw
                self.bias[class_idx] -= self.learning_rate * db
                
                # Accumulate average weights
                self.avg_weights[class_idx] += self.weights[class_idx]
                self.avg_bias[class_idx] += self.bias[class_idx]
                
                loss = self._perceptron_loss(X_train, y_train, class_idx)
                epoch_loss += loss
            
            avg_loss = epoch_loss / self.n_classes
            self.loss_history.append(avg_loss)
            
            y_train_pred = self.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            self.train_acc_history.append(train_acc)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_acc = accuracy_score(y_val, y_val_pred)
                self.val_acc_history.append(val_acc)
            
            if self.verbose and (epoch + 1) % 100 == 0:
                if X_val is not None:
                    print(f"  Epoch {epoch+1}/{self.n_iterations} | Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
                else:
                    print(f"  Epoch {epoch+1}/{self.n_iterations} | Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.4f}")
        
        # Use averaged weights for prediction
        self.weights = self.avg_weights / self.n_iterations
        self.bias = self.avg_bias / self.n_iterations
        
        print(f"\n Averaged Perceptron training completed!")
        print(f"  Final training accuracy: {train_acc:.4f}")
        if X_val is not None:
            print(f"  Final validation accuracy: {val_acc:.4f}")
        
        return self


