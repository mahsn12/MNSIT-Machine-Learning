import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1.  Perceptron (from scratch)
# =============================================================================

class Perceptron:
    """Binary Perceptron classifier."""

    def __init__(self, learning_rate: float = 0.01,
                 n_epochs: int = 50, random_state: int = 42):
        self.lr           = learning_rate
        self.n_epochs     = n_epochs
        self.random_state = random_state
        self.weights_     = None
        self.bias_        = None
        self.errors_      = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0, 0.01, X.shape[1])
        self.bias_    = 0.0
        self.errors_  = []
        for _ in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred   = self._activate(xi)
                update = self.lr * (yi - pred)
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

print("=" * 60)
print("  Perceptron - Binary Classification: 0  vs  Not-0")
print("=" * 60)

print("\n[1] Loading Digits dataset (8x8 images) ...")
digits = load_digits()
X_raw, y_raw = digits.data, digits.target

y_binary = (y_raw != 0).astype(int)   # 0=zero, 1=not-zero

print(f"    Total samples : {X_raw.shape[0]}")
print(f"    'zero'     (label 0) : {(y_binary == 0).sum()}")
print(f"    'not-zero' (label 1) : {(y_binary == 1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n    Train : {X_train.shape[0]} samples")
print(f"    Test  : {X_test.shape[0]} samples")


# =============================================================================
# 3.  Train Perceptron
# =============================================================================

print("\n" + "-" * 60)
print("[2] Training Perceptron  (0 vs Not-0)")
print("-" * 60)

pct = Perceptron(learning_rate=0.01, n_epochs=50, random_state=42)
pct.fit(X_train, y_train)

y_pred = pct.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n    Test Accuracy : {acc * 100:.2f} %")
print("\n    Classification Report :")
print(classification_report(y_test, y_pred, target_names=["Zero (0)", "Not-Zero (1-9)"]))


# =============================================================================
# 4.  Figure – Main Results (convergence, confusion, weights)
# =============================================================================

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Perceptron: Digit '0'  vs  'Not 0'  —  Results",
             fontsize=15, fontweight="bold")

# (a) Sample test images
ax = axes[0, 0]
n_show = 20
idxs   = rng.choice(len(X_test), n_show, replace=False)
grid   = np.zeros((8 * 2, 8 * 10))
borders = []
for i, si in enumerate(idxs):
    r, c = divmod(i, 10)
    img  = scaler.inverse_transform(X_test[si:si+1])[0].reshape(8, 8)
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

# (b) Convergence curve
ax = axes[0, 1]
ep = range(1, len(pct.errors_) + 1)
ax.plot(ep, pct.errors_, marker="o", color="steelblue", linewidth=2, markersize=5)
ax.fill_between(ep, pct.errors_, alpha=0.15, color="steelblue")
ax.set_title("Convergence - Misclassifications per Epoch")
ax.set_xlabel("Epoch"); ax.set_ylabel("# Misclassified")
ax.grid(True, alpha=0.4)

# (c) Confusion matrix
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: Zero", "Pred: Not-Zero"],
            yticklabels=["True: Zero", "True: Not-Zero"])
ax.set_title("Confusion Matrix")

# (d) Learned weight map
ax = axes[1, 1]
w_img = pct.weights_.reshape(8, 8)
im = ax.imshow(w_img, cmap="RdBu_r", aspect="auto")
plt.colorbar(im, ax=ax)
ax.set_title("Learned Weights  (blue = push 'Not-Zero',  red = push 'Zero')")
ax.axis("off")

plt.tight_layout()
plt.show()
print("\n[3] Plot displayed successfully")


# =============================================================================
# 5.  Summary
# =============================================================================

tn, fp, fn, tp = cm.ravel()
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Task     : Binary  —  0  vs  Not-0  (digits 1-9)")
print(f"  Epochs   : {pct.n_epochs}   |   LR : {pct.lr}")
print(f"  Accuracy : {acc * 100:.2f} %")
print(f"  Zeros correctly identified     : {tn}")
print(f"  Zeros missed  (false positive) : {fp}")
print(f"  Not-Zeros correctly identified : {tp}")
print(f"  Not-Zeros missed (false neg.)  : {fn}")
print("=" * 60)