# =========================================================
# PHASE 2 - MULTINOMIAL NB FEATURE COMPARISON
# FULL SCRATCH MODEL + EVALUATION
# NO SKLEARN IN MODEL/EVALUATION
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# =========================================================
# SCRATCH METRICS
# =========================================================

def scratch_confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def scratch_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def scratch_precision_recall_f1(y_true, y_pred, n_classes=10):
    cm = scratch_confusion_matrix(y_true, y_pred, n_classes)
    precisions = []
    recalls = []
    f1_scores = []

    for c in range(n_classes):
        TP = cm[c, c]
        FP = np.sum(cm[:, c]) - TP
        FN = np.sum(cm[c, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)


def scratch_evaluate(y_true, y_pred, n_classes=10):
    precision, recall, f1 = scratch_precision_recall_f1(y_true, y_pred, n_classes)
    return {
        "Accuracy": scratch_accuracy(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def scratch_classification_report(y_true, y_pred, n_classes=10):
    cm = scratch_confusion_matrix(y_true, y_pred, n_classes)
    print("\nClassification Report")
    print("-" * 60)
    print("Class\tPrecision\tRecall\t\tF1-score")

    for c in range(n_classes):
        TP = cm[c, c]
        FP = np.sum(cm[:, c]) - TP
        FN = np.sum(cm[c, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        print(f"{c}\t{precision:.4f}\t\t{recall:.4f}\t\t{f1:.4f}")


def scratch_variance_threshold(X_train, X_val, X_test, threshold=0.001):
    variances = np.var(X_train, axis=0)
    selected = variances > threshold
    return X_train[:, selected], X_val[:, selected], X_test[:, selected]


# =========================================================
# MULTINOMIAL NB SCRATCH
# =========================================================

class MultinomialNBScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_log_priors = {}
        self.feature_log_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_log_priors[c] = np.log(len(X_c) / len(X))
            feature_counts = np.sum(X_c, axis=0) + self.alpha
            total_count = np.sum(feature_counts)
            self.feature_log_probs[c] = np.log(feature_counts / total_count)

    def predict(self, X):
        predictions = []
        for sample in X:
            scores = {}
            for c in self.classes:
                scores[c] = self.class_log_priors[c] + np.sum(sample * self.feature_log_probs[c])
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def make_non_negative(X_train, X_val, X_test):
    min_val = X_train.min()
    X_train_new = X_train - min_val + 1e-6
    X_val_new = X_val - min_val + 1e-6
    X_test_new = X_test - min_val + 1e-6
    return X_train_new, X_val_new, X_test_new


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def learning_curve(X_train, y_train, X_val, y_val, alpha, feature_name):
    train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    train_scores = []
    val_scores = []

    for size in train_sizes:
        n = int(len(X_train) * size)
        X_part = X_train[:n]
        y_part = y_train[:n]

        model = MultinomialNBScratch(alpha=alpha)
        model.fit(X_part, y_part)

        y_train_pred = model.predict(X_part)
        y_val_pred = model.predict(X_val)

        train_metrics = scratch_evaluate(y_part, y_train_pred)
        val_metrics = scratch_evaluate(y_val, y_val_pred)

        train_scores.append(train_metrics["F1"])
        val_scores.append(val_metrics["F1"])

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores, marker="o", label="Training F1")
    plt.plot(train_sizes, val_scores, marker="o", label="Validation F1")
    plt.xlabel("Training Data Size")
    plt.ylabel("Macro F1-score")
    plt.title(f"Learning Curve - Multinomial NB using {feature_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# =========================================================
# LOAD DATA
# =========================================================

processor = DataProcessorPhase2()
data = processor.run_complete_pipeline()

feature_types = ["Flatten", "PCA", "HOG"]
results = []


# =========================================================
# FEATURE COMPARISON
# =========================================================

for feature_name in feature_types:
    print("\n" + "=" * 80)
    print(f"MULTINOMIAL NB USING {feature_name} FEATURES")
    print("=" * 80)

    X_train = data[feature_name]["X_train"]
    X_val = data[feature_name]["X_val"]
    X_test = data[feature_name]["X_test"]
    y_train = data[feature_name]["y_train"]
    y_val = data[feature_name]["y_val"]
    y_test = data[feature_name]["y_test"]

    # Feature selection from scratch
    X_train, X_val, X_test = scratch_variance_threshold(X_train, X_val, X_test, threshold=0.001)
    print("After Feature Selection:", X_train.shape)

    # Multinomial NB requires non-negative input
    X_train_mn, X_val_mn, X_test_mn = make_non_negative(X_train, X_val, X_test)

    # Hyperparameter tuning
    best_model = None
    best_alpha = None
    best_val_f1 = -1
    best_val_acc = -1
    best_train_time = None

    for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        model = MultinomialNBScratch(alpha=alpha)

        start = time.time()
        model.fit(X_train_mn, y_train)
        train_time = time.time() - start

        y_val_pred = model.predict(X_val_mn)
        metrics = scratch_evaluate(y_val, y_val_pred)

        val_acc = metrics["Accuracy"]
        val_f1 = metrics["F1"]

        print(f"alpha={alpha} | Val Accuracy={val_acc:.4f} | Val F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_model = model
            best_alpha = alpha
            best_train_time = train_time

    # Test best model
    y_test_pred = best_model.predict(X_test_mn)
    metrics = scratch_evaluate(y_test, y_test_pred)

    print("\nBest alpha:", best_alpha)
    print(f"Training Time: {best_train_time:.4f} seconds")
    print(f"Validation Accuracy: {best_val_acc:.4f}")
    print(f"Validation F1: {best_val_f1:.4f}")

    print("\nTest Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    scratch_classification_report(y_test, y_test_pred)

    cm = scratch_confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm, f"Multinomial NB - {feature_name}")

    learning_curve(X_train_mn, y_train, X_val_mn, y_val, best_alpha, feature_name)

    results.append({
        "Feature Type": feature_name,
        "Best Alpha": best_alpha,
        "Train Time": best_train_time,
        "Validation Accuracy": best_val_acc,
        "Validation F1": best_val_f1,
        "Test Accuracy": metrics["Accuracy"],
        "Test Precision": metrics["Precision"],
        "Test Recall": metrics["Recall"],
        "Test F1": metrics["F1"]
    })


# =========================================================
# FINAL RESULTS TABLE
# =========================================================

df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("FINAL FEATURE COMPARISON TABLE")
print("=" * 80)
print(df_results.to_string(index=False))

df_results.to_csv("phase2_multinomial_features_results.csv", index=False)


# =========================================================
# BAR CHART
# =========================================================

plt.figure(figsize=(10, 6))
x = np.arange(len(df_results))
width = 0.2

plt.bar(x - width, df_results["Test Accuracy"], width, label="Accuracy")
plt.bar(x, df_results["Test Precision"], width, label="Precision")
plt.bar(x + width, df_results["Test Recall"], width, label="Recall")

plt.xticks(x, df_results["Feature Type"])
plt.ylabel("Score")
plt.title("Multinomial NB Feature Comparison - Phase 2")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# BEST FEATURE TYPE
# =========================================================

best_idx = df_results["Validation F1"].idxmax()
print("\nBEST FEATURE TYPE:")
print(df_results.loc[best_idx])