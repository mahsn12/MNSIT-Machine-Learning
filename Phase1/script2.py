import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1.  Perceptron (from scratch)
# =============================================================================

class Perceptron:
    """Binary Perceptron classifier with optional class weights."""

    def __init__(self, learning_rate: float = 0.01,
                 n_epochs: int = 50, random_state: int = 42,
                 class_weight: dict = None):
        self.lr           = learning_rate
        self.n_epochs     = n_epochs
        self.random_state = random_state
        self.class_weight = class_weight
        self.weights_     = None
        self.bias_        = None
        self.errors_      = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, X.shape[1])
        self.bias_    = 0.0
        self.errors_  = []

        # Build per-sample weight vector from class_weight dict
        if self.class_weight is not None:
            sample_weights = np.array([self.class_weight[yi] for yi in y])
        else:
            sample_weights = np.ones(len(y))

        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi, wi in zip(X, y, sample_weights):
                pred   = self._activate(xi)
                update = self.lr * wi * (yi - pred)   # scaled by sample weight
                self.weights_ += update * xi
                self.bias_    += update
                errors        += int(update != 0)
            self.errors_.append(errors)
        return self

    def _activate(self, x: np.ndarray) -> int:
        return int(np.dot(x, self.weights_) + self.bias_ >= 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._activate(xi) for xi in X])


# =============================================================================
# 2.  Load & Prepare Data
# =============================================================================

digits = load_digits()
X_raw, y_raw = digits.data, digits.target   # shape: (1797, 64)

y_binary = (y_raw != 0).astype(int)         # change from multiclass to binary

# Three-way split  →  train / validation / test  (60 / 20 / 20)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_binary, test_size=0.20, random_state=42, stratify=y_binary
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)   # 0.25 × 0.80 = 0.20 of total → validation

# Class-imbalance weight computation
n_total = len(y_train)
n_zero  = (y_train == 0).sum()
n_nzero = (y_train == 1).sum()

# sklearn-style balanced weights: n_samples / (n_classes * class_count)
w0 = n_total / (2 * n_zero)    # weight for 'zero' class  → higher
w1 = n_total / (2 * n_nzero)   # weight for 'not-zero'    → lower
class_weights = {0: w0, 1: w1}

# =============================================================================
# 3.  Feature Extraction Pipelines
# =============================================================================

# ---- PCA --------------------------------------------------------------------
# Scale first — PCA is sensitive to feature magnitudes
scaler_pca  = StandardScaler()
X_train_sc  = scaler_pca.fit_transform(X_train)
X_val_sc    = scaler_pca.transform(X_val)
X_test_sc   = scaler_pca.transform(X_test)

# Fit on train only; retain >= 95 % variance (auto-selects n_components)
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_sc)
X_val_pca   = pca.transform(X_val_sc)
X_test_pca  = pca.transform(X_test_sc)

n_comp        = pca.n_components_
var_explained = pca.explained_variance_ratio_.sum() * 100

# ---- HOG --------------------------------------------------------------------
def extract_hog(X_flat: np.ndarray) -> np.ndarray:
    """Reshape each row to 8x8, extract HOG, return feature matrix."""
    feats = []
    for row in X_flat:
        img = row.reshape(8, 8)
        f = hog(img,
                orientations=8,
                pixels_per_cell=(4, 4),
                cells_per_block=(1, 1),
                feature_vector=True)
        feats.append(f)
    return np.array(feats)

X_train_hog_raw = extract_hog(X_train)
X_val_hog_raw   = extract_hog(X_val)
X_test_hog_raw  = extract_hog(X_test)

scaler_hog  = StandardScaler()
X_train_hog = scaler_hog.fit_transform(X_train_hog_raw)
X_val_hog   = scaler_hog.transform(X_val_hog_raw)
X_test_hog  = scaler_hog.transform(X_test_hog_raw)

hog_dim = X_train_hog.shape[1]

# =============================================================================
# 4.  Train & Compare — PCA vs HOG (both class-weighted)
# =============================================================================

print("\n" + "-" * 60)
print("[4] Training Perceptrons: PCA vs HOG  (class-weighted)")
print("-" * 60)

configs = [
    ("PCA features (class-weighted)", X_train_pca, X_val_pca, X_test_pca, class_weights),
    ("HOG features (class-weighted)", X_train_hog, X_val_hog, X_test_hog, class_weights),
]

results = {}
for name, Xtr, Xvl, Xte, cw in configs:
    pct = Perceptron(learning_rate=0.01, n_epochs=50,
                     random_state=42, class_weight=cw)
    pct.fit(Xtr, y_train)

    val_acc  = accuracy_score(y_val,  pct.predict(Xvl)) * 100
    test_acc = accuracy_score(y_test, pct.predict(Xte)) * 100
    results[name] = {"model": pct, "Xte": Xte,
                     "val_acc": val_acc, "test_acc": test_acc}
    print(f"\n    [{name}]")
    print(f"      Val  accuracy : {val_acc:.2f} %")
    print(f"      Test accuracy : {test_acc:.2f} %")

# Best model by validation accuracy
best_name = max(results, key=lambda k: results[k]["val_acc"])
best      = results[best_name]
best_pct  = best["model"]
y_pred    = best_pct.predict(best["Xte"])

print(f"\n    * Best config (by val acc): {best_name}")
print(f"      Test accuracy : {best['test_acc']:.2f} %")
print("\n    Classification Report (best model) :")
print(classification_report(y_test, y_pred,
                             target_names=["Zero (0)", "Not-Zero (1-9)"]))

# =============================================================================
# 5.  Figure - Results Dashboard
# =============================================================================

rng = np.random.default_rng(42)
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle(
    f"Perceptron - Digit '0' vs 'Not-0'  -  Best: {best_name}",
    fontsize=13, fontweight="bold"
)

# (a) Sample test images
ax = axes[0, 0]
n_show = 20
idxs   = rng.choice(len(X_test), n_show, replace=False)
grid   = np.zeros((8 * 2, 8 * 10))
borders = []
for i, si in enumerate(idxs):
    r, c = divmod(i, 10)
    # Inverse-transform via PCA scaler to recover pixel values for display
    img = scaler_pca.inverse_transform(X_test_sc[si:si+1])[0].reshape(8, 8)
    grid[r*8:(r+1)*8, c*8:(c+1)*8] = img
    borders.append("limegreen" if y_test[si] == y_pred[si] else "red")
ax.imshow(grid, cmap="gray")
ax.set_title("Sample test images  (green=correct, red=wrong)")
ax.axis("off")
for i, col in enumerate(borders):
    r2, c2 = divmod(i, 10)
    rect = plt.Rectangle((c2*8 - 0.5, r2*8 - 0.5), 8, 8,
                          linewidth=2.5, edgecolor=col, facecolor="none")
    ax.add_patch(rect)

# (b) Convergence curves - PCA vs HOG
ax = axes[0, 1]
colors = ["green", "red"]
for (name, *_), col in zip(configs, colors):
    errs = results[name]["model"].errors_
    ep   = range(1, len(errs) + 1)
    ax.plot(ep, errs, label=name, color=col, linewidth=1.8)
ax.set_title("Convergence - Misclassifications per Epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("# Misclassified")
ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

# (c) Confusion matrix - best model
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: Zero", "Pred: Not-Zero"],
            yticklabels=["True: Zero", "True: Not-Zero"])
ax.set_title(f"Confusion Matrix  ({best_name})")

# (d) PCA explained variance
ax = axes[1, 1]
cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
ax.plot(range(1, len(cumvar) + 1), cumvar, marker="o",
        color="purple", linewidth=2, markersize=4)
ax.axhline(95, color="red", linestyle="--", linewidth=1, label="95 % threshold")
ax.axvline(n_comp, color="gray", linestyle=":", linewidth=1,
           label=f"{n_comp} components selected")
ax.set_title("PCA - Cumulative Explained Variance")
ax.set_xlabel("# Components"); ax.set_ylabel("Variance explained (%)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

# (e) HOG feature visualisation
ax = axes[2, 0]
sample_img = X_train[0].reshape(8, 8)
hog_feats, hog_img = hog(sample_img, orientations=8,
                          pixels_per_cell=(4, 4), cells_per_block=(1, 1),
                          feature_vector=True, visualize=True)
ax.imshow(hog_img, cmap="viridis")
ax.set_title("HOG feature visualisation (sample image)")
ax.axis("off")

# (f) Val vs Test accuracy bar chart
ax = axes[2, 1]
labels    = [n.replace(" (", "\n(") for n in results]
val_accs  = [results[n]["val_acc"]  for n in results]
test_accs = [results[n]["test_acc"] for n in results]
x = np.arange(len(labels))
w = 0.35
ax.bar(x - w/2, val_accs,  w, label="Val",  color="steelblue", alpha=0.85)
ax.bar(x + w/2, test_accs, w, label="Test", color="coral",     alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(80, 101)
ax.set_title("Val vs Test Accuracy - PCA vs HOG")
ax.legend(); ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.show()
print("\n[5] Plot displayed successfully")

# =============================================================================
# 6.  Summary
# =============================================================================

tn, fp, fn, tp = cm.ravel()
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Task       : Binary  -  0  vs  Not-0  (digits 1-9)")
print(f"  Best config: {best_name}")
print(f"  Epochs     : {best_pct.n_epochs}   |   LR : {best_pct.lr}")
print(f"  Val  acc   : {best['val_acc']:.2f} %")
print(f"  Test acc   : {best['test_acc']:.2f} %")
print(f"  Zeros correctly identified     : {tn}")
print(f"  Zeros missed  (false positive) : {fp}")
print(f"  Not-Zeros correctly identified : {tp}")
print(f"  Not-Zeros missed (false neg.)  : {fn}")
print(f"\n  Feature dimensions used:")
print(f"    PCA : {n_comp}-dim  ({var_explained:.1f}% variance retained)")
print(f"    HOG : {hog_dim}-dim")
print(f"\n  Class weights : zero={w0:.3f}  |  not-zero={w1:.3f}")
print("=" * 60)
