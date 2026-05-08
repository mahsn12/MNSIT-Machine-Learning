# main.py (version without seaborn)
# Main script to run Phase 1: Binary Classification with Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import our modules
from preprocessing import DataProcessor
from LogisticRegression_phase1 import LogisticRegressionManual


class ModelTrainer:
    """Train and evaluate models across different feature types."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_model(self, X_train, y_train, X_val, y_val, feature_name, 
                   learning_rate=0.1, n_iterations=1000):
        """Train logistic regression model."""
        print(f"\n    Training model on {feature_name} features...")
        
        model = LogisticRegressionManual(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            verbose=True
        )
        
        model.fit(X_train, y_train, X_val, y_val)
        self.models[feature_name] = model
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, feature_name):
        """Evaluate model and return metrics."""
        print(f"\n    Evaluating {feature_name} model...")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"        Accuracy:  {metrics['accuracy']:.4f}")
        print(f"        Precision: {metrics['precision']:.4f}")
        print(f"        Recall:    {metrics['recall']:.4f}")
        print(f"        F1-Score:  {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, cm, feature_name):
        """Plot confusion matrix using matplotlib (no seaborn)."""
        plt.figure(figsize=(6, 5))
        
        # Create heatmap-like plot with imshow
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        # Add labels
        classes = ['Digit 4', 'Digit 9']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        
        plt.title(f'Confusion Matrix - {feature_name} Features')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compare all models and display results."""
        print("\n" + "="*70)
        print("STEP 6: MODEL COMPARISON")
        print("="*70)
        
        print("\n" + "-"*70)
        print(f"{'Feature Type':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*70)
        
        for feature_name, metrics in self.results.items():
            print(f"{feature_name:<15} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
        
        print("-"*70)
        
        # Find best model
        best_feature = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\n🏆 Best Model: {best_feature[0]} features with accuracy = {best_feature[1]['accuracy']:.4f}")
        
        # Plot comparison
        self.plot_comparison()
    
    def plot_comparison(self):
        """Plot bar chart comparing models."""
        features = list(self.results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_values = {metric: [self.results[f][metric] for f in features] for metric in metrics_names}
        
        x = np.arange(len(features))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, values) in enumerate(metrics_values.items()):
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[i])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Feature Type')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison Across Feature Types')
        ax.set_xticks(x)
        ax.set_xticklabels(features)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the complete Phase 1 pipeline."""
    
    print("="*70)
    print(" " * 15 + "CSE382 - PHASE 1: BINARY CLASSIFICATION")
    print(" " * 10 + "MNIST Digit Classification (4 vs 9)")
    print(" " * 15 + "Logistic Regression Implementation")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize components
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Run preprocessing pipeline
    processed_data = processor.run_complete_pipeline()
    
    # Train and evaluate models
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING AND EVALUATION")
    print("="*70)
    
    for feature_name, data in processed_data.items():
        print(f"\n{'='*70}")
        print(f"Processing {feature_name.upper()} Features")
        print(f"{'='*70}")
        
        # Train model
        model = trainer.train_model(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            feature_name,
            learning_rate=0.1,
            n_iterations=1000
        )
        
        # Plot loss curves
        model.plot_loss_curves()
        
        # Evaluate on test set
        metrics = trainer.evaluate_model(model, data['X_test'], data['y_test'], feature_name)
        
        # Plot confusion matrix
        trainer.plot_confusion_matrix(metrics['confusion_matrix'], feature_name)
        
        # Store results
        trainer.results[feature_name] = metrics
    
    # Compare all models
    trainer.compare_models()
    
    # Save results
    print("\n" + "="*70)
    print("STEP 7: SAVING RESULTS")
    print("="*70)
    
    # Save models and results
    with open('phase1_results.pkl', 'wb') as f:
        pickle.dump({
            'models': trainer.models,
            'results': trainer.results,
            'processed_data': processed_data
        }, f)
    print("\n✅ Results saved to 'phase1_results.pkl'")
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total execution time: {elapsed_time:.2f} seconds")
    print(f"   Features tested: {', '.join(trainer.results.keys())}")
    print(f"   Best performing: {max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[0]}")
    print(f"   Best accuracy: {max(trainer.results.items(), key=lambda x: x[1]['accuracy'])[1]['accuracy']:.4f}")
    print("\n📁 Deliverables:")
    print("   1. Processed data and models saved to 'phase1_results.pkl'")
    print("   2. All visualizations displayed")
    print("   3. Complete training logs shown above")
    


# Run the main function
if __name__ == "__main__":
    main()