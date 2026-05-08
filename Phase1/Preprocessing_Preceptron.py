# preprocessing.py
# Phase 2: Enhanced preprocessing for 10-class MNIST classification
# Includes: CNN Features, Cross-Validation, Regularization Analysis
# WITH VISUALIZATIONS AFTER EACH STEP
# DATA SPLIT: 80% Train, 10% Validation, 10% Test

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.transform import resize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow for CNN features
try:
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from tensorflow.keras.models import Model
    CNN_AVAILABLE = True
    print("✓ TensorFlow available - CNN feature extraction enabled")
except ImportError:
    CNN_AVAILABLE = False
    print("⚠ TensorFlow not available - Using HOG + PCA features")


class Phase2DataProcessor:
    """Enhanced data processing pipeline for 10-class MNIST classification."""
    
    def __init__(self):
        self.scalers = {}
        self.pca = None
        self.cnn_model = None
        self.features_dict = None
        self.processed_data = None
        self.num_classes = 10
        self.cv = None
        self.regularization_analysis = {}
        
    def load_and_prepare_data(self):
        """Load all 10 MNIST classes."""
        print("\n" + "="*70)
        print("PHASE 2: LOADING MNIST DATA (ALL 10 CLASSES)")
        print("="*70)
        
        print("\n[1.1] Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X = mnist.data.values.astype(np.float64)
        y = mnist.target.values.astype(int)
        
        print(f"    Total samples: {len(X)}")
        print(f"    Number of classes: {len(np.unique(y))}")
        print(f"    Classes: 0-9")
        
        print(f"\n[1.2] Class distribution:")
        for i in range(10):
            count = np.sum(y == i)
            print(f"    Class {i}: {count:6d} samples ({count/len(X)*100:.1f}%)")
        
        # VISUALIZATION 1: Class distribution bar chart
        self._visualize_class_distribution(y)
        
        return X, y
    
    def _visualize_class_distribution(self, y):
        """Visualize class distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        class_counts = [np.sum(y == i) for i in range(10)]
        bars = ax.bar(range(10), class_counts, color='skyblue', edgecolor='black')
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('MNIST Class Distribution (All 10 Digits)', fontsize=14)
        ax.set_xticks(range(10))
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                   str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        print("   📊 Visualization 1: Class distribution displayed")
    
    def display_sample_images(self, X, y):
        """Display sample images from all 10 classes."""
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        for i in range(10):
            row = i // 5
            col = i % 5
            idx = np.where(y == i)[0][0]
            axes[row, col].imshow(X[idx].reshape(28, 28), cmap='gray')
            axes[row, col].set_title(f"Digit {i}", fontsize=12)
            axes[row, col].axis('off')
        
        plt.suptitle("Sample Images from All 10 Classes", fontsize=14)
        plt.tight_layout()
        plt.show()
        print("   📊 Visualization 2: Sample images displayed")
    
    def normalize_images(self, X):
        """Normalize pixel values to [0, 1]."""
        print("\n[1.3] Normalizing images...")
        print(f"    Before normalization: [{X.min():.2f}, {X.max():.2f}]")
        X_normalized = X / 255.0
        print(f"    After normalization: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")
        
        # VISUALIZATION 3: Before/After normalization comparison
        self._visualize_normalization(X, X_normalized)
        
        return X_normalized
    
    def _visualize_normalization(self, X_before, X_after):
        """Visualize before and after normalization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Before normalization
        axes[0].hist(X_before.flatten(), bins=50, color='blue', alpha=0.7)
        axes[0].set_xlabel('Pixel Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Before Normalization\nRange: [{X_before.min():.1f}, {X_before.max():.1f}]')
        axes[0].grid(True, alpha=0.3)
        
        # After normalization
        axes[1].hist(X_after.flatten(), bins=50, color='green', alpha=0.7)
        axes[1].set_xlabel('Pixel Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'After Normalization\nRange: [{X_after.min():.2f}, {X_after.max():.2f}]')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Pixel Value Distribution Before and After Normalization', fontsize=14)
        plt.tight_layout()
        plt.show()
        print("   📊 Visualization 3: Normalization effect displayed")
    
    def extract_cnn_features(self, X, target_size=(32, 32)):
        """
        IMPROVEMENT 1: Extract features using pretrained CNN.
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 1: CNN FEATURE EXTRACTION (Transfer Learning)")
        print("="*70)
        
        if not CNN_AVAILABLE:
            print("\n[CNN] Using HOG features as alternative...")
            return self.extract_hog_features(X)
        
        print("\n[CNN] Loading pretrained VGG16 model...")
        
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(target_size[0], target_size[1], 3))
        
        self.cnn_model = Model(inputs=base_model.input, 
                              outputs=base_model.get_layer('fc2').output)
        
        print(f"    Model loaded. Output shape: {self.cnn_model.output_shape}")
        
        n_samples = X.shape[0]
        features_list = []
        batch_size = 100
        
        print(f"\n[CNN] Extracting features from {n_samples} images...")
        
        for i in range(0, n_samples, batch_size):
            batch = X[i:i+batch_size]
            batch_features = []
            
            for img_flat in batch:
                img_2d = img_flat.reshape(28, 28)
                img_resized = resize(img_2d, target_size, mode='reflect', preserve_range=True)
                img_rgb = np.stack([img_resized] * 3, axis=-1)
                img_preprocessed = preprocess_input(img_rgb * 255)
                img_batch = np.expand_dims(img_preprocessed, axis=0)
                features = self.cnn_model.predict(img_batch, verbose=0)
                batch_features.append(features.flatten())
            
            features_list.extend(batch_features)
            
            if (i + batch_size) % 2000 == 0:
                print(f"    Processed {min(i+batch_size, n_samples)}/{n_samples} images")
        
        features = np.array(features_list)
        print(f"\n[CNN] Raw features shape: {features.shape}")
        
        # Reduce dimensions with PCA
        print(f"\n[CNN] Reducing dimensions with PCA...")
        self.pca = PCA(n_components=100, random_state=42)
        features = self.pca.fit_transform(features)
        explained_var = self.pca.explained_variance_ratio_.sum()
        
        print(f"    Reduced to 100 dimensions")
        print(f"    Explained variance: {explained_var:.2%}")
        
        # VISUALIZATION 4: PCA explained variance
        self._visualize_pca_variance()
        
        return features
    
    def extract_hog_features(self, X, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
        """Extract HOG features (fallback when CNN not available)."""
        print("\n[2.2] Extracting HOG features...")
        n_samples = X.shape[0]
        hog_features = []
        
        # VISUALIZATION: Sample HOG features
        sample_idx = 0
        sample_img = X[sample_idx].reshape(28, 28)
        _, hog_image = hog(sample_img, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, orientations=9,
                          visualize=True, block_norm='L2-Hys')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(sample_img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(hog_image, cmap='gray')
        ax2.set_title('HOG Features')
        ax2.axis('off')
        plt.suptitle('HOG Feature Visualization', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        for i in range(n_samples):
            img = X[i].reshape(28, 28)
            features = hog(img, pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block, orientations=9,
                          block_norm='L2-Hys')
            hog_features.append(features)
            
            if (i + 1) % 5000 == 0:
                print(f"    Processed {i+1}/{n_samples} images")
        
        features = np.array(hog_features)
        print(f"    Shape: {features.shape}")
        print(f"    Features dimension: {features.shape[1]}")
        
        print(f"\n[2.3] Reducing dimensions with PCA...")
        self.pca = PCA(n_components=100, random_state=42)
        features = self.pca.fit_transform(features)
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"    Reduced to 100 dimensions (explained variance: {explained_var:.2%})")
        
        self._visualize_pca_variance()
        
        return features
    
    def _visualize_pca_variance(self):
        """Visualize PCA explained variance."""
        if self.pca is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Individual explained variance
            axes[0].bar(range(1, min(21, len(self.pca.explained_variance_ratio_)+1)), 
                       self.pca.explained_variance_ratio_[:20], color='skyblue')
            axes[0].set_xlabel('Principal Component')
            axes[0].set_ylabel('Explained Variance Ratio')
            axes[0].set_title('Top 20 PCA Components')
            axes[0].grid(True, alpha=0.3)
            
            # Cumulative explained variance
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            axes[1].plot(range(1, len(cumsum)+1), cumsum, 'b-', linewidth=2)
            axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
            axes[1].set_xlabel('Number of Components')
            axes[1].set_ylabel('Cumulative Explained Variance')
            axes[1].set_title('PCA Cumulative Explained Variance')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle('PCA Analysis', fontsize=14)
            plt.tight_layout()
            plt.show()
            print("   📊 Visualization 4: PCA variance explained")
    
    def visualize_feature_space(self, features, y, n_samples=1000):
        """Visualize the feature space after CNN extraction."""
        print("\n[2.4] Visualizing feature space...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: PCA visualization (2D)
        from sklearn.decomposition import PCA as PCA2D
        pca_2d = PCA2D(n_components=2, random_state=42)
        features_sample = features[:n_samples]
        y_sample = y[:n_samples]
        features_2d = pca_2d.fit_transform(features_sample)
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i in range(10):
            mask = y_sample == i
            axes[0, 0].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                              c=[colors[i]], label=f'Digit {i}', alpha=0.5, s=5)
        axes[0, 0].set_xlabel('First Principal Component')
        axes[0, 0].set_ylabel('Second Principal Component')
        axes[0, 0].set_title('Feature Space Visualization (2D PCA)')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature variance
        feature_vars = np.var(features, axis=0)
        top_features = np.argsort(feature_vars)[-30:]
        axes[0, 1].bar(range(len(top_features)), feature_vars[top_features])
        axes[0, 1].set_xlabel('Feature Index')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_title('Top 30 Features by Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Class separability
        class_centers = []
        for i in range(10):
            class_samples = features[y == i][:100]
            center = np.mean(class_samples, axis=0)
            class_centers.append(center)
        
        between_distances = []
        for i in range(10):
            for j in range(i+1, 10):
                dist = np.linalg.norm(class_centers[i] - class_centers[j])
                between_distances.append(dist)
        
        axes[1, 0].hist(between_distances, bins=20, color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Distance between Class Centers')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Class Separability Analysis')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: PCA explained variance
        if self.pca is not None:
            cumsum_ratio = np.cumsum(self.pca.explained_variance_ratio_)
            axes[1, 1].plot(cumsum_ratio, 'b-', linewidth=2)
            axes[1, 1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
            axes[1, 1].set_xlabel('Number of Components')
            axes[1, 1].set_ylabel('Cumulative Explained Variance')
            axes[1, 1].set_title('PCA Explained Variance')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Feature Space Analysis - 10 Classes', fontsize=14)
        plt.tight_layout()
        plt.show()
        print("   📊 Visualization 5: Feature space analysis displayed")
    
    def setup_cross_validation(self, X_train, y_train, n_folds=5):
        """
        IMPROVEMENT 2: Setup cross-validation for hyperparameter tuning.
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 2: CROSS-VALIDATION SETUP")
        print("="*70)
        
        self.cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        print(f"\n[CV] Created {n_folds}-fold stratified cross-validation")
        print(f"    Each fold preserves class distribution")
        print(f"    Total training samples: {X_train.shape[0]}")
        print(f"    Samples per fold: {X_train.shape[0] // n_folds}")
        
        # VISUALIZATION: Cross-validation fold sizes
        fold_sizes = []
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train, y_train)):
            fold_sizes.append(len(val_idx))
            train_classes = np.unique(y_train[train_idx])
            val_classes = np.unique(y_train[val_idx])
            print(f"    Fold {fold+1}: Train={len(train_idx)}, Val={len(val_idx)}, "
                  f"Classes={len(train_classes)}/{len(val_classes)}")
        
        self._visualize_cv_folds(fold_sizes, n_folds)
        
        return self.cv
    
    def _visualize_cv_folds(self, fold_sizes, n_folds):
        """Visualize cross-validation fold sizes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(1, n_folds+1), fold_sizes, color='lightcoral', edgecolor='black')
        ax.set_xlabel('Fold Number', fontsize=12)
        ax.set_ylabel('Validation Set Size', fontsize=12)
        ax.set_title(f'{n_folds}-Fold Stratified Cross-Validation', fontsize=14)
        ax.set_xticks(range(1, n_folds+1))
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, size in zip(bars, fold_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   str(size), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        print("    Visualization 6: Cross-validation folds displayed")
    
    def analyze_regularization(self, X_train, y_train):
        """
        IMPROVEMENT 3: Analyze L1/L2 regularization needs and bias-variance tradeoff.
        """
        print("\n" + "="*70)
        print("IMPROVEMENT 3: REGULARIZATION & BIAS-VARIANCE ANALYSIS")
        print("="*70)
        
        n_samples, n_features = X_train.shape
        class_counts = [np.sum(y_train == i) for i in range(10)]
        
        self.regularization_analysis = {
            'n_samples': n_samples,
            'n_features': n_features,
            'feature_sample_ratio': n_features / n_samples,
            'class_imbalance_ratio': max(class_counts) / min(class_counts),
            'l1_lambdas': [0, 0.0001, 0.001, 0.01, 0.05, 0.1],
            'l2_lambdas': [0, 0.0001, 0.001, 0.01, 0.05, 0.1]
        }
        
        print(f"\n[Analysis] Dataset characteristics:")
        print(f"    Samples: {n_samples}")
        print(f"    Features: {n_features}")
        print(f"    Feature/Sample ratio: {n_features/n_samples:.3f}")
        print(f"    Class imbalance ratio: {self.regularization_analysis['class_imbalance_ratio']:.2f}")
        
        # Bias-Variance Analysis
        print(f"\n[Bias-Variance Analysis]")
        
        if n_features > n_samples:
            print(f"    ⚠ High-dimensional data (features > samples)")
            print(f"    → High risk of overfitting (high variance)")
            print(f"    → Recommendation: Use L1 regularization for feature selection")
            print(f"    → Suggested λ range: 0.001 to 0.1")
        elif n_features < 100:
            print(f"    ✓ Low-dimensional data")
            print(f"    → Higher risk of underfitting (high bias)")
            print(f"    → Recommendation: Use L2 regularization or no regularization")
            print(f"    → Suggested λ range: 0 to 0.01")
        else:
            print(f"    → Balanced feature-sample ratio")
            print(f"    → Recommendation: Use L2 regularization")
            print(f"    → Suggested λ range: 0.0001 to 0.01")
        
        # VISUALIZATION: Regularization impact visualization
        self._visualize_regularization_impact()
        
        return self.regularization_analysis
    
    def _visualize_regularization_impact(self):
        """Visualize regularization impact on model complexity."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # L1 Regularization (Lasso)
        lambdas = [0, 0.0001, 0.001, 0.01, 0.05, 0.1]
        sparsity = [100, 60, 40, 20, 10, 5]  # Example sparsity percentages
        
        axes[0].plot(lambdas, sparsity, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Regularization Strength (λ)', fontsize=12)
        axes[0].set_ylabel('Feature Sparsity (%)', fontsize=12)
        axes[0].set_title('L1 Regularization (Lasso)\nFeature Selection Effect', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # L2 Regularization (Ridge)
        weight_magnitude = [100, 80, 50, 30, 15, 5]
        
        axes[1].plot(lambdas, weight_magnitude, 'ro-', linewidth=2, markersize=8)
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Regularization Strength (λ)', fontsize=12)
        axes[1].set_ylabel('Weight Magnitude (%)', fontsize=12)
        axes[1].set_title('L2 Regularization (Ridge)\nWeight Shrinkage Effect', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Regularization Impact on Model Complexity', fontsize=14)
        plt.tight_layout()
        plt.show()
        print("   📊 Visualization 7: Regularization impact displayed")
    
    def split_and_normalize(self, features, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Split data into train/val/test with specified ratios.
        DEFAULT: 80% Train, 10% Validation, 10% Test
        
        Parameters:
        -----------
        features : numpy array
            Feature matrix
        y : numpy array
            Labels
        train_ratio : float, default=0.8
            Proportion for training (80%)
        val_ratio : float, default=0.1
            Proportion for validation (10%)
        test_ratio : float, default=0.1
            Proportion for testing (10%)
        """
        print("\n" + "="*70)
        print("DATA SPLITTING AND NORMALIZATION")
        print("="*70)
        print(f"\n[Split Configuration] {train_ratio*100:.0f}% Train, {val_ratio*100:.0f}% Validation, {test_ratio*100:.0f}% Test")
        
        # First split: separate test set (test_ratio)
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, y, test_size=test_ratio, random_state=42, stratify=y
        )
        
        # Second split: separate validation from remaining (val_ratio / (train_ratio + val_ratio))
        relative_val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=relative_val_size, random_state=42, stratify=y_temp
        )
        
        # Verify split proportions
        total = len(features)
        train_pct = len(X_train) / total * 100
        val_pct = len(X_val) / total * 100
        test_pct = len(X_test) / total * 100
        
        print(f"\n[Split] Data distribution:")
        print(f"    Training:   {X_train.shape[0]} samples ({train_pct:.1f}%)")
        print(f"    Validation: {X_val.shape[0]} samples ({val_pct:.1f}%)")
        print(f"    Test:       {X_test.shape[0]} samples ({test_pct:.1f}%)")
        
        # Verify splits sum to 100%
        print(f"    Total:      {total} samples ({(train_pct+val_pct+test_pct):.1f}%)")
        
        # VISUALIZATION: Data split pie chart
        self._visualize_data_split(X_train.shape[0], X_val.shape[0], X_test.shape[0])
        
        # Normalize features
        print(f"\n[Normalization] Standardizing features...")
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train)
        X_val_norm = scaler.transform(X_val)
        X_test_norm = scaler.transform(X_test)
        
        print(f"    After normalization:")
        print(f"        Train - mean: {X_train_norm.mean():.2e}, std: {X_train_norm.std():.2f}")
        print(f"        Val   - mean: {X_val_norm.mean():.2e}, std: {X_val_norm.std():.2f}")
        print(f"        Test  - mean: {X_test_norm.mean():.2e}, std: {X_test_norm.std():.2f}")
        
        # VISUALIZATION: Normalized feature distribution
        self._visualize_normalized_features(X_train_norm)
        
        # Display class distribution in splits
        print(f"\n[Class Distribution in Splits]")
        print(f"{'Digit':<8} {'Train':<10} {'Validation':<12} {'Test':<10}")
        print("-" * 45)
        for i in range(10):
            train_count = np.sum(y_train == i)
            val_count = np.sum(y_val == i)
            test_count = np.sum(y_test == i)
            train_pct_class = train_count / len(y_train) * 100
            val_pct_class = val_count / len(y_val) * 100
            test_pct_class = test_count / len(y_test) * 100
            print(f"Digit {i}:   {train_count:5d} ({train_pct_class:5.1f}%)   {val_count:5d} ({val_pct_class:5.1f}%)   {test_count:5d} ({test_pct_class:5.1f}%)")
        
        # VISUALIZATION: Class distribution across splits
        self._visualize_split_distribution(y_train, y_val, y_test)
        
        return {
            'X_train': X_train_norm,
            'X_val': X_val_norm,
            'X_test': X_test_norm,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler,
            'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
        }
    
    def _visualize_data_split(self, train_size, val_size, test_size):
        """Visualize data split proportions."""
        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [train_size, val_size, test_size]
        labels = [f'Train\n{train_size} (80%)', f'Validation\n{val_size} (10%)', f'Test\n{test_size} (10%)']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        explode = (0.05, 0.05, 0.05)
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax.set_title('Data Split Distribution (80% Train, 10% Val, 10% Test)', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        print("    Visualization 8: Data split proportions displayed (80/10/10)")
    
    def _visualize_normalized_features(self, X_train):
        """Visualize normalized feature distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram of feature means
        feature_means = np.mean(X_train, axis=0)
        axes[0].hist(feature_means, bins=50, color='blue', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
        axes[0].set_xlabel('Feature Mean')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Feature Means\n(Should be centered at 0)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram of feature stds
        feature_stds = np.std(X_train, axis=0)
        axes[1].hist(feature_stds, bins=50, color='green', alpha=0.7)
        axes[1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Std = 1')
        axes[1].set_xlabel('Feature Standard Deviation')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Feature Std Dev\n(Should be centered at 1)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Feature Normalization Quality Check', fontsize=14)
        plt.tight_layout()
        plt.show()
        print("    Visualization 9: Normalized features distribution displayed")
    
    def _visualize_split_distribution(self, y_train, y_val, y_test):
        """Visualize class distribution across splits."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(10)
        width = 0.25
        
        train_counts = [np.sum(y_train == i) for i in range(10)]
        val_counts = [np.sum(y_val == i) for i in range(10)]
        test_counts = [np.sum(y_test == i) for i in range(10)]
        
        ax.bar(x - width, train_counts, width, label='Train (80%)', color='#2ecc71', alpha=0.8)
        ax.bar(x, val_counts, width, label='Validation (10%)', color='#f39c12', alpha=0.8)
        ax.bar(x + width, test_counts, width, label='Test (10%)', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel('Digit Class', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Class Distribution Across 80/10/10 Split', fontsize=14)
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        print("    Visualization 10: Split class distribution displayed (80/10/10)")
    
    def extract_all_features(self, X):
        """Extract all feature types (for backward compatibility)."""
        print("\n" + "="*70)
        print("FEATURE EXTRACTION")
        print("="*70)
        
        features_dict = {}
        
        features_dict['CNN_Features'] = self.extract_cnn_features(X)
        features_dict['HOG_Features'] = self.extract_hog_features(X)
        
        self.features_dict = features_dict
        return features_dict
    
    def run_complete_pipeline(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Run the complete enhanced preprocessing pipeline.
        
        Parameters:
        -----------
        train_ratio : float, default=0.8 (80% for training)
        val_ratio : float, default=0.1 (10% for validation)
        test_ratio : float, default=0.1 (10% for testing)
        """
        print("\n" + "="*70)
        print("PHASE 2: COMPLETE PREPROCESSING PIPELINE")
        print(f"DATA SPLIT: {train_ratio*100:.0f}% Train, {val_ratio*100:.0f}% Validation, {test_ratio*100:.0f}% Test")
        print("="*70)
        
        # Step 1: Load all 10 classes
        X, y = self.load_and_prepare_data()
        
        # Step 2: Display sample images
        self.display_sample_images(X, y)
        
        # Step 3: Normalize images
        X_normalized = self.normalize_images(X)
        
        # Step 4: Extract CNN features (Improvement 1)
        X_features = self.extract_cnn_features(X_normalized)
        
        # Step 5: Visualize feature space
        self.visualize_feature_space(X_features, y)
        
        # Step 6: Split and normalize data with 80/10/10 split
        processed_data = self.split_and_normalize(X_features, y, train_ratio, val_ratio, test_ratio)
        
        # Step 7: Setup cross-validation (Improvement 2)
        cv = self.setup_cross_validation(processed_data['X_train'], processed_data['y_train'])
        
        # Step 8: Analyze regularization (Improvement 3)
        reg_analysis = self.analyze_regularization(processed_data['X_train'], processed_data['y_train'])
        
        # Add additional information to processed data
        processed_data['num_classes'] = self.num_classes
        processed_data['feature_dim'] = X_features.shape[1]
        processed_data['cv'] = cv
        processed_data['regularization_analysis'] = reg_analysis
        processed_data['pca'] = self.pca
        
        # Set attributes on the object
        self.feature_dim = X_features.shape[1]
        self.scaler = processed_data['scaler']
        self.processed_data = processed_data
        
        # Final summary
        print("\n" + "="*70)
        print("PHASE 2 PREPROCESSING COMPLETED!")
        print("="*70)
        print(f"\n Summary of Improvements:")
        print(f"   1. CNN Feature Extraction: {X_features.shape[1]} features (PCA-reduced)")
        print(f"   2. Cross-Validation: {cv.get_n_splits()}-fold stratified CV")
        print(f"   3. Regularization Analysis: L1/L2 recommendations provided")
        print(f"\n Data ready for classifier:")
        print(f"   Training:   {processed_data['X_train'].shape} ({train_ratio*100:.0f}%)")
        print(f"   Validation: {processed_data['X_val'].shape} ({val_ratio*100:.0f}%)")
        print(f"   Test:       {processed_data['X_test'].shape} ({test_ratio*100:.0f}%)")
        print(f"   Classes:    {self.num_classes}")
        
        return processed_data, self
    
    

