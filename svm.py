import numpy as np


def SVM(D: list, L: list, lr: float = 0.01,
        n_features: int = 256, margin: int = 1):

    X = []

    for image in D:
        row = np.append(np.array(image, dtype=np.float32), 1)
        X.append(row)

    X = np.array(X, dtype=float)

    W = np.zeros(n_features + 1, dtype=float)

    Y = np.array(L, dtype=float)
    Y[Y == 0] = -1

    W = train_SVM(X, Y, W, margin, lr)

    return W


def train_SVM(X, Y, W, margin, lr, lmda=0.01,
              max_iters=1000, tol=1e-4):

    for _ in range(max_iters):

        WX = X @ W
        WX = np.array(WX, dtype=np.float32)

        Lw = []

        for i, wx in enumerate(WX):
            margin_value = Y[i] * wx

            if margin_value > margin:
                Lw.append(np.zeros(len(W)))
            else:
                Lw.append(-Y[i] * X[i])

        Lw = np.array(Lw, dtype=np.float32)

        SumLw = np.sum(Lw, axis=0)

        Wnew = W - lr * (SumLw + lmda * W)

        # ✅ stopping condition
        if np.linalg.norm(Wnew - W) < tol:
            break

        W = Wnew

    return W