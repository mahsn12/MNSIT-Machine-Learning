from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from preprocessing import preprocess, skeletonize_image
from KNN_compute import KNN, build_training_graphs

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

data = mnist.data
classes = mnist.target

print(f"Dataset shape : {data.shape}")
print(f"Label of [0]  : {classes[0]}")
print(f"Label of [2]  : {classes[2]}")

def show_pipeline(indices: list[int]):

    n = len(indices)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    fig.suptitle(
        "Pipeline: Raw → Binarized → Skeleton",
        fontsize=14,
        fontweight="bold",
        y=1.01
    )

    col_titles = [
        "Raw Grayscale",
        "Binarized (Otsu + Centered)",
        "Skeleton (Zhang-Suen)"
    ]

    for col, title in enumerate(col_titles):

        axes[0, col].set_title(
            title,
            fontsize=11,
            pad=8
        )

    for row, idx in enumerate(indices):

        raw = data[idx].reshape(28, 28).astype(np.uint8)

        binary = preprocess(data[idx])

        skel = skeletonize_image(binary)

        label = classes[idx]

        imgs = [
            (raw, "gray"),
            (binary, "gray"),
            (skel, "gray")
        ]

        for col, (img, cmap) in enumerate(imgs):

            ax = axes[row, col]

            ax.imshow(
                img,
                cmap=cmap,
                vmin=0,
                vmax=255
            )

            ax.axis("off")

            if col == 0:

                ax.set_ylabel(
                    f"Label: {label}",
                    fontsize=10,
                    rotation=0,
                    labelpad=45,
                    va="center"
                )

    plt.tight_layout()

    plt.show()

sample_indices = []

seen = set()

for i, lbl in enumerate(classes):

    if lbl not in seen:

        sample_indices.append(i)

        seen.add(lbl)

    if len(seen) == 10:
        break

show_pipeline(sample_indices)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    classes,
    test_size=0.20,
    random_state=42,
    stratify=classes
)

print("Train Size :", len(X_train))

print("Test Size  :", len(X_test))

train_set = []

print("\nPreprocessing full training set...\n")

for idx, img in enumerate(X_train):

    binary = preprocess(img)

    skel = skeletonize_image(binary)

    train_set.append(skel)

    if idx % 1000 == 0:
        print(f"Processed {idx} / {len(X_train)} training images")

print("\nTraining preprocessing complete.\n")

print("Building training graphs...")

train_graphs = build_training_graphs(train_set)

print("Training graphs ready.")

predictions = []

samples_to_test = len(X_test)

print("\nTesting on full test set...\n")

for i in range(samples_to_test):

    binary_test = preprocess(X_test[i])

    skel_test = skeletonize_image(binary_test)

    pred = KNN(
        test_image=skel_test,
        train_graphs=train_graphs,
        train_labels=y_train,
        k=3
    )

    predictions.append(str(pred))

    if i % 500 == 0:
        print(f"Tested {i} / {samples_to_test} images")

true_labels = y_test[:samples_to_test]
acc = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    average="weighted"
)
recall = recall_score(
    true_labels,
    predictions,
    average="weighted"
)
f1 = f1_score(
    true_labels,
    predictions,
    average="weighted"
)

print("\n──────── FINAL RESULTS ────────\n")

print(f"Accuracy  : {acc * 100:.2f}%")

print(f"Precision : {precision * 100:.2f}%")

print(f"Recall    : {recall * 100:.2f}%")

print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predictions
    )
)
cm = confusion_matrix(
    true_labels,
    predictions
)

print("Confusion Matrix:")

print(cm)
plt.figure(figsize=(8, 6))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.xticks(np.arange(10))

plt.yticks(np.arange(10))

for i in range(10):

    for j in range(10):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.colorbar()

plt.tight_layout()

plt.show()