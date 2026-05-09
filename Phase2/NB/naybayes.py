# =========================================================
# PHASE 2 - NAIVE BAYES TYPES COMPARISON
# FULL SCRATCH IMPLEMENTATION
# Gaussian vs Multinomial vs Bernoulli
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
# GAUSSIAN NB SCRATCH
# =========================================================

class GaussianNBScratch:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = {}
        self.means = {}
        self.vars = {}

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = len(X_c) / len(X)
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + self.var_smoothing

    def predict(self, X):
        predictions = []
        for sample in X:
            scores = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = -0.5 * np.sum(
                    np.log(2 * np.pi * self.vars[c]) +
                    ((sample - self.means[c]) ** 2) / self.vars[c]
                )
                scores[c] = prior + likelihood
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)


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
# BERNOULLI NB SCRATCH
# =========================================================

class BernoulliNBScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_log_priors = {}
        self.feature_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            self.class_log_priors[c] = np.log(len(X_c) / len(X))
            self.feature_probs[c] = (np.sum(X_c, axis=0) + self.alpha) / (len(X_c) + 2 * self.alpha)

    def predict(self, X):
        predictions = []
        for sample in X:
            scores = {}
            for c in self.classes:
                probs = self.feature_probs[c]
                log_prob = self.class_log_priors[c]
                log_prob += np.sum(
                    sample * np.log(probs + 1e-10) +
                    (1 - sample) * np.log(1 - probs + 1e-10)
                )
                scores[c] = log_prob
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)


# =========================================================
# LOAD DATA
# =========================================================

processor = DataProcessorPhase2()
data = processor.run_complete_pipeline()

feature_name = "HOG"

X_train = data[feature_name]["X_train"]
X_val = data[feature_name]["X_val"]
X_test = data[feature_name]["X_test"]
y_train = data[feature_name]["y_train"]
y_val = data[feature_name]["y_val"]
y_test = data[feature_name]["y_test"]


# =========================================================
# FEATURE SELECTION
# =========================================================

X_train, X_val, X_test = scratch_variance_threshold(X_train, X_val, X_test, threshold=0.001)
print("\nAfter Feature Selection:", X_train.shape)

results = []


# =========================================================
# GAUSSIAN NB
# =========================================================

print("\n" + "=" * 70)
print("GAUSSIAN NAIVE BAYES")
print("=" * 70)

best_model = None
best_f1 = -1
best_vs = None

for vs in [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
    model = GaussianNBScratch(var_smoothing=vs)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    metrics = scratch_evaluate(y_val, y_val_pred)
    print(f"var_smoothing={vs} | Validation F1={metrics['F1']:.4f}")

    if metrics["F1"] > best_f1:
        best_f1 = metrics["F1"]
        best_model = model
        best_vs = vs

start = time.time()
y_test_pred = best_model.predict(X_test)
test_time = time.time() - start

metrics = scratch_evaluate(y_test, y_test_pred)
metrics["Model"] = "Gaussian NB"
metrics["Best Parameter"] = best_vs
metrics["Test Time"] = test_time
results.append(metrics)

print("\nBest var_smoothing:", best_vs)
scratch_classification_report(y_test, y_test_pred)

cm = scratch_confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Gaussian NB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()


# =========================================================
# MULTINOMIAL NB
# =========================================================

print("\n" + "=" * 70)
print("MULTINOMIAL NAIVE BAYES")
print("=" * 70)

min_val = X_train.min()
X_train_mn = X_train - min_val + 1e-6
X_val_mn = X_val - min_val + 1e-6
X_test_mn = X_test - min_val + 1e-6

best_model = None
best_f1 = -1
best_alpha = None

for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    model = MultinomialNBScratch(alpha=alpha)
    model.fit(X_train_mn, y_train)
    y_val_pred = model.predict(X_val_mn)
    metrics = scratch_evaluate(y_val, y_val_pred)
    print(f"alpha={alpha} | Validation F1={metrics['F1']:.4f}")

    if metrics["F1"] > best_f1:
        best_f1 = metrics["F1"]
        best_model = model
        best_alpha = alpha

start = time.time()
y_test_pred = best_model.predict(X_test_mn)
test_time = time.time() - start

metrics = scratch_evaluate(y_test, y_test_pred)
metrics["Model"] = "Multinomial NB"
metrics["Best Parameter"] = best_alpha
metrics["Test Time"] = test_time
results.append(metrics)

print("\nBest alpha:", best_alpha)
scratch_classification_report(y_test, y_test_pred)

cm = scratch_confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Greens")
plt.title("Multinomial NB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()


# =========================================================
# BERNOULLI NB
# =========================================================

print("\n" + "=" * 70)
print("BERNOULLI NAIVE BAYES")
print("=" * 70)

threshold = np.mean(X_train)
X_train_b = (X_train > threshold).astype(int)
X_val_b = (X_val > threshold).astype(int)
X_test_b = (X_test > threshold).astype(int)

best_model = None
best_f1 = -1
best_alpha = None

for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    model = BernoulliNBScratch(alpha=alpha)
    model.fit(X_train_b, y_train)
    y_val_pred = model.predict(X_val_b)
    metrics = scratch_evaluate(y_val, y_val_pred)
    print(f"alpha={alpha} | Validation F1={metrics['F1']:.4f}")

    if metrics["F1"] > best_f1:
        best_f1 = metrics["F1"]
        best_model = model
        best_alpha = alpha

start = time.time()
y_test_pred = best_model.predict(X_test_b)
test_time = time.time() - start

metrics = scratch_evaluate(y_test, y_test_pred)
metrics["Model"] = "Bernoulli NB"
metrics["Best Parameter"] = best_alpha
metrics["Test Time"] = test_time
results.append(metrics)

print("\nBest alpha:", best_alpha)
scratch_classification_report(y_test, y_test_pred)

cm = scratch_confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Oranges")
plt.title("Bernoulli NB Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.show()


# =========================================================
# FINAL RESULTS
# =========================================================

df_results = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FINAL COMPARISON TABLE")
print("=" * 70)
print(df_results.to_string(index=False))

df_results.to_csv("phase2_nb_types_results.csv", index=False)

plt.figure(figsize=(10, 6))
x = np.arange(len(df_results))
width = 0.2

plt.bar(x - width, df_results["Accuracy"], width, label="Accuracy")
plt.bar(x, df_results["Precision"], width, label="Precision")
plt.bar(x + width, df_results["Recall"], width, label="Recall")

plt.xticks(x, df_results["Model"])
plt.ylabel("Score")
plt.title("Naive Bayes Types Comparison - Phase 2")
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

best_idx = df_results["F1"].idxmax()
print("\nBEST MODEL:")
print(df_results.loc[best_idx])