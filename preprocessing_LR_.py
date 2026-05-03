# preprocessing.py
# Data processing pipeline for MNIST binary classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Complete data processing pipeline for MNIST binary classification."""
    
    def __init__(self):
        self.scalers = {}
        self.pca = None
        self.features_dict = None
        self.processed_data = None
        
    def load_and_prepare_data(self):
        """Load MNIST and select classes 4 and 9."""
        print("\n" + "="*70)
        print("STEP 1: LOADING AND PREPARING DATA")
        print("="*70)
        
        print("\n[1.1] Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.values.astype(np.float64)
        y = mnist.target.values.astype(int)
        
        # Select classes 4 and 9
        mask = (y == 4) | (y == 9)
        X = X[mask]
        y = y[mask]
        
        # Convert labels: 4 -> 0, 9 -> 1
        y = np.where(y == 4, 0, 1)
        
        print(f"    Selected classes: Digit 4 (class 0) and Digit 9 (class 1)")
        print(f"    Total samples: {len(X)}")
        print(f"    Class distribution:")
        print(f"        Class 0 (digit 4): {np.sum(y == 0)} samples")
        print(f"        Class 1 (digit 9): {np.sum(y == 1)} samples")
        
        return X, y
    
    def display_sample_images(self, X, y):
        """Display sample images from each class."""
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i in range(5):
            # Class 0 samples
            idx0 = np.where(y == 0)[0][i]
            axes[0, i].imshow(X[idx0].reshape(28, 28), cmap='gray')
            axes[0, i].set_title(f"Digit 4 #{i+1}")
            axes[0, i].axis('off')
            
            # Class 1 samples
            idx1 = np.where(y == 1)[0][i]
            axes[1, i].imshow(X[idx1].reshape(28, 28), cmap='gray')
            axes[1, i].set_title(f"Digit 9 #{i+1}")
            axes[1, i].axis('off')
        
        plt.suptitle("Sample Images from Each Class", fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def normalize_images(self, X):
        """Normalize pixel values to [0, 1]."""
        print("\n[1.2] Normalizing images...")
        print(f"    Before normalization: [{X.min():.2f}, {X.max():.2f}]")
        X_normalized = X / 255.0
        print(f"    After normalization: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
        return X_normalized
    
    def extract_flatten_features(self, X):
        """Extract flatten features (baseline)."""
        print("\n[2.1] Extracting Flatten features...")
        features = X  # Already flattened from MNIST
        print(f"    Shape: {features.shape}")
        print(f"    Features dimension: {features.shape[1]}")
        return features
    
    def extract_pca_features(self, X, n_components=50):
        """Extract PCA features."""
        print("\n[2.2] Extracting PCA features...")
        self.pca = PCA(n_components=n_components, random_state=42)
        features = self.pca.fit_transform(X)
        explained_variance = self.pca.explained_variance_ratio_.sum()
        print(f"    Shape: {features.shape}")
        print(f"    Features dimension: {features.shape[1]}")
        print(f"    Explained variance: {explained_variance:.2%}")
        return features
    
    def extract_hog_features(self, X, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
        """Extract HOG features from images."""
        print("\n[2.3] Extracting HOG features...")
        n_samples = X.shape[0]
        hog_features = []
        
        for i in range(n_samples):
            img = X[i].reshape(28, 28)
            features = hog(img, 
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          orientations=9,
                          block_norm='L2-Hys')
            hog_features.append(features)
        
        features = np.array(hog_features)
        print(f"    Shape: {features.shape}")
        print(f"    Features dimension: {features.shape[1]}")
        return features
    
    def visualize_hog(self, image):
        """Visualize HOG features for a sample image."""
        from skimage.feature import hog as hog_visualize
        
        img = image.reshape(28, 28)
        _, hog_image = hog_visualize(img, 
                                     pixels_per_cell=(4, 4),
                                     cells_per_block=(2, 2),
                                     orientations=9,
                                     visualize=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(img, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        ax2.imshow(hog_image, cmap='gray')
        ax2.set_title("HOG Visualization")
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
    
    def extract_all_features(self, X):
        """Extract all three feature types."""
        print("\n" + "="*70)
        print("STEP 2: FEATURE EXTRACTION")
        print("="*70)
        
        features_dict = {}
        
        # Flatten features
        features_dict['Flatten'] = self.extract_flatten_features(X)
        
        # PCA features
        features_dict['PCA'] = self.extract_pca_features(X)
        
        # HOG features
        features_dict['HOG'] = self.extract_hog_features(X)
        
        # Visualize HOG for first sample
        self.visualize_hog(X[0])
        
        self.features_dict = features_dict
        return features_dict
    
    def handle_imbalance(self, X, y):
        """Handle class imbalance if necessary."""
        print("\n" + "="*70)
        print("STEP 3: HANDLING CLASS IMBALANCE")
        print("="*70)
        
        print(f"\n[3.1] Original distribution:")
        print(f"    Class 0: {np.sum(y == 0)} samples")
        print(f"    Class 1: {np.sum(y == 1)} samples")
        
        class_counts = [np.sum(y == 0), np.sum(y == 1)]
        imbalance_ratio = max(class_counts) / min(class_counts)
        
        print(f"    Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 1.2:
            print("\n[3.2] Applying oversampling to balance classes...")
            
            # Separate classes
            X_class0 = X[y == 0]
            X_class1 = X[y == 1]
            
            if len(X_class0) < len(X_class1):
                minority_X = X_class0
                minority_y = np.zeros(len(minority_X))
                majority_X = X_class1
                majority_y = np.ones(len(majority_X))
            else:
                minority_X = X_class1
                minority_y = np.ones(len(minority_X))
                majority_X = X_class0
                majority_y = np.zeros(len(majority_X))
            
            # Oversample minority class
            minority_upsampled = resample(minority_X,
                                         replace=True,
                                         n_samples=len(majority_X),
                                         random_state=42)
            minority_y_upsampled = resample(minority_y,
                                           replace=True,
                                           n_samples=len(majority_X),
                                           random_state=42)
            
            # Combine balanced dataset
            X_balanced = np.vstack([majority_X, minority_upsampled])
            y_balanced = np.hstack([majority_y, minority_y_upsampled])
            
            print(f"    After balancing:")
            print(f"    Class 0: {np.sum(y_balanced == 0)} samples")
            print(f"    Class 1: {np.sum(y_balanced == 1)} samples")
            
            return X_balanced, y_balanced
        else:
            print("\n[3.2] No significant imbalance detected, proceeding with original data.")
            return X, y
    
    def split_and_normalize(self, features_dict, y):
        """Split data into train/val/test and normalize features."""
        print("\n" + "="*70)
        print("STEP 4: TRAIN/VAL/TEST SPLIT & FEATURE NORMALIZATION")
        print("="*70)
        
        processed_data = {}
        
        for feature_name, X_features in features_dict.items():
            print(f"\n[4.{list(features_dict.keys()).index(feature_name)+1}] Processing {feature_name} features...")
            
            # First split: separate test set (20%)
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_features, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Second split: separate validation (25% of temp = 20% of total)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
            
            print(f"    Before normalization:")
            print(f"        Train: {X_train.shape}")
            print(f"        Validation: {X_val.shape}")
            print(f"        Test: {X_test.shape}")
            
            # Normalize features
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_val_norm = scaler.transform(X_val)
            X_test_norm = scaler.transform(X_test)
            
            print(f"    After normalization (mean≈0, std≈1):")
            print(f"        Train mean: {X_train_norm.mean():.2e}, std: {X_train_norm.std():.2f}")
            print(f"        Validation mean: {X_val_norm.mean():.2e}, std: {X_val_norm.std():.2f}")
            print(f"        Test mean: {X_test_norm.mean():.2e}, std: {X_test_norm.std():.2f}")
            
            processed_data[feature_name] = {
                'X_train': X_train_norm,
                'X_val': X_val_norm,
                'X_test': X_test_norm,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'scaler': scaler
            }
        
        self.processed_data = processed_data
        return processed_data
    
    def run_complete_pipeline(self):
        """Run the complete preprocessing pipeline."""
        # Load data
        X, y = self.load_and_prepare_data()
        
        # Display sample images
        self.display_sample_images(X, y)
        
        # Normalize images
        X_normalized = self.normalize_images(X)
        
        # Extract features
        features_dict = self.extract_all_features(X_normalized)
        
        # Handle imbalance
        X_balanced, y_balanced = self.handle_imbalance(X_normalized, y)
        
        # If balancing was applied, re-extract features
        if len(X_balanced) != len(X_normalized):
            print("\n[4.0] Re-extracting features with balanced data...")
            features_dict = self.extract_all_features(X_balanced)
            y = y_balanced
        
        # Split and normalize
        processed_data = self.split_and_normalize(features_dict, y)
        
        return processed_data


# For testing the preprocessing module independently
if __name__ == "__main__":
    print("Testing Preprocessing Module...")
    processor = DataProcessor()
    data = processor.run_complete_pipeline()
    print("\n✅ Preprocessing completed successfully!")
    print(f"Available feature sets: {list(data.keys())}")