import numpy as np
import heapq
from scipy.spatial import cKDTree


def image_to_graph(skeleton: np.ndarray) -> np.ndarray:

    rows, cols = np.where(skeleton > 0)

    points = np.column_stack((cols, rows)).astype(np.float32)

    return points


def one_sided_chamfer(A: np.ndarray, B: np.ndarray) -> float:

    if len(A) == 0 or len(B) == 0:
        return 1e9

    tree = cKDTree(B)

    distances, _ = tree.query(A, k=1)

    return distances.mean()


def Symmetric_chamfer_distance(A, B):

    return (
        one_sided_chamfer(A, B)
        +
        one_sided_chamfer(B, A)
    )

def build_training_graphs(train_set):

    graphs = []

    for img in train_set:

        points = image_to_graph(img)

        graphs.append(points)

    return graphs


def KNN(
    test_image,
    train_graphs,
    train_labels,
    k=3
):

    test_points = image_to_graph(test_image)

    heap = []

    for i, train_points in enumerate(train_graphs):

        dist = Symmetric_chamfer_distance(
            test_points,
            train_points
        )

        if len(heap) < k:

            heapq.heappush(
                heap,
                (-dist, train_labels[i])
            )

        else:

            worst = -heap[0][0]

            if dist < worst:

                heapq.heapreplace(
                    heap,
                    (-dist, train_labels[i])
                )

    votes = [0] * 10

    for d, c in heap:

        votes[int(c)] += 1

    prediction = votes.index(max(votes))

    return prediction