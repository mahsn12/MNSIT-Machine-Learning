from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from svm import SVM


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

data = mnist.data
classes = mnist.target.astype(int)

# keep only 0 and 1
mask = (classes == 0) | (classes == 1)
data = data[mask]
classes = classes[mask]


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    data, classes, test_size=0.2, random_state=42
)


# ─────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────
pca = PCA(n_components=256)

X_train = pca.fit_transform(X_train_raw)
X_test = pca.transform(X_test_raw)


# ─────────────────────────────────────────────
# TRAIN SVM
# ─────────────────────────────────────────────
W = SVM(
    D=X_train,
    L=y_train,
    margin=1,
    lr=0.01,
    n_features=256
)


# ─────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict(X, W):
    preds = []

    for x in X:
        x = np.append(x, 1)  # bias
        val = np.dot(x, W)

        if val >= 0:
            preds.append(1)
        else:
            preds.append(0)

    return np.array(preds)


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
y_pred = predict(X_test, W)


# ─────────────────────────────────────────────
# PLOT FIRST 5 TEST IMAGES WITH EXPECTED / PREDICTED CLASS
# ─────────────────────────────────────────────
images_to_show = 5

fig, axes = plt.subplots(1, images_to_show, figsize=(12, 3))

for i in range(images_to_show):
    axes[i].imshow(X_test_raw[i].reshape(28, 28), cmap="gray")
    axes[i].set_title(
        f"Expected: {y_test[i]}\nPredicted: {y_pred[i]}",
        fontsize=9
    )
    axes[i].axis("off")

plt.tight_layout()
plt.show()


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("Confusion Matrix:\n", cm)


# ─────────────────────────────────────────────
# PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ─────────────────────────────────────────────
# CORRELATION MATRIX (reduced for speed)
# ─────────────────────────────────────────────
corr = np.corrcoef(X_train[:, :50], rowvar=False)

plt.figure(figsize=(6,6))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Matrix (first 50 features)")
plt.show()
