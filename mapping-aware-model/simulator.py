from collections import defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd


def generate_common_cm(L: int, h: float, gamma: float) -> np.ndarray:
    """
    Generates the L x L common confusion matrix using the heterogeneity factor, h and
    the lower bound on accuracy, gamma. The first L/2 labels map to the first decision, while the
    remaining L/2 labels map to the second decision.

    We generate the common error matrix M_error, and then mix it with the identify matrix
    using gamma.

    When h = 0, every row of M_error is [1, 1, ..., 1] normalized. When h = 1, every row of M_error
    is [1*1, 2*2, ..., L/2 * L/2, 1*1, 2*2, ..., L/2*L/2] noramlized. For any h in between,
    we simply take a convex combination of the two.

    For example: suppose that L = 4 and gamma = 0.8. Then:

    no_heterogeneity = [0.25, 0.25, 0.25, 0.25]
    max_heterogeneity = [0.1, 0.4, 0.1, 0.4]

    """
    no_heterogeneity = np.ones(int(L))
    no_heterogeneity = no_heterogeneity / sum(no_heterogeneity)
    max_heterogeneity = np.array(
        [x**1.5 for x in range(1, int(L / 2) + 1)]
        + [x**1.5 for x in range(1, int(L / 2) + 1)]
    )
    max_heterogeneity = max_heterogeneity / sum(max_heterogeneity)

    M_error_row = no_heterogeneity * (1 - h) + max_heterogeneity * h
    M_error = np.array([M_error_row for _ in range(L)])

    return np.array(gamma * np.identity(L) + (1 - gamma) * M_error)


def generate_random_matrix_with_lb(L: int, gamma: float) -> np.ndarray:
    M = np.random.rand(L, L)
    M = M / M.sum(axis=1)[:, np.newaxis]
    return np.array(gamma * np.identity(L) + (1 - gamma) * M)


def generate_reviewer_cm(
    a: float,  # a = 0 is the first reviewer, a = 1 is the last.
    L: int,  # size of confusion matrix
    gamma: float,  # lower bound on diagonal entries
) -> np.ndarray:
    """
    For each of the A reviewers, their confusion matrix is diagonal with uniform off-diagonal entries
    with mean equal to mean_acc.
    """
    mean_acc = gamma * (1 - a) + 1 * a
    accuracy = [
        np.random.uniform(mean_acc - 0.1, min(1, mean_acc + 0.1)) for _ in range(L)
    ]

    matrix = []
    for idx in range(L):  # constructing each row of the matrix
        row = [
            (1 - accuracy[idx]) / (L - 1) for _ in range(L)
        ]  # non-diagonal entries are uniform
        row[idx] = accuracy[idx]  # diagonal entry given by
        matrix.append(row)

    return np.array(matrix)


def generate_data(
    L: int,  # Number of labels
    I: int,  # Number of items
    A: int,  # Number of reviewers
    M_final: List[np.ndarray],  # LxL confusion matrix for each of the A reviewers.
    label_to_decision: Dict[str, str],  # map from labels to decision
    num_reviews_per_content=3,  # number of reviews to draw for each of the I items
):
    data = defaultdict(list)

    for _ in range(I):
        y = np.random.choice(
            range(L),
        )

        reviews = []
        labels = []
        labels_binary = []
        for _ in range(num_reviews_per_content):
            reviewer = np.random.choice(range(A))
            rating = np.random.choice(range(L), p=M_final[reviewer][y])

            reviews.append(str(reviewer))
            labels.append(str(rating))
            labels_binary.append(label_to_decision[str(rating)])

        data["labelers"].append(reviews)
        data["ratings"].append(labels)
        data["ratings_binary"].append(labels_binary)
        data["first_rating_binary"].append(labels_binary[0])
        data["true_label"].append(y)
        data["true_decision"].append(label_to_decision[str(y)])
        data["first_decision_correct"].append(
            labels_binary[0] == label_to_decision[str(y)]
        )

    return pd.DataFrame(data)


class SimulationDataInstance(NamedTuple):
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    decision_to_ratings_map: Dict[str, List[str]]
    common_confusion: np.ndarray
    reviewer_confusions: List[np.ndarray]


def generate_simulation_data(
    I: int,  # number of items
    L: int,  # number of labels, the first int(L/2) map to decision A and the remaining to decision B.
    A: int,  # number of annotators/reviewers
    gamma: float,  # lower bound on the any reviewer's accuracy
    h: float,  # heterogeneity factor, how different L/2 labels in each decision vary in terms of reliability
    mixing_factor: float,  # weight between individual CM and population CM
) -> SimulationDataInstance:
    assert (
        L % 2 == 0
    ), "L must be even so it can be partioned into two equal sets of decisions"

    M_common = generate_common_cm(L, h, gamma)

    M_final = []
    for a in range(A):
        M_personal = generate_reviewer_cm((float(a) + 1) / (A + 1), L, gamma)
        M_final.append(M_personal * mixing_factor + M_common * (1 - mixing_factor))

    labels_to_decisions = {str(i): "A" if i < L / 2 else "B" for i in range(L)}
    decision_to_ratings_map = {
        "A": [str(i) for i in range(L) if i < L / 2],
        "B": [str(i) for i in range(L) if i >= L / 2],
    }

    train_data = generate_data(L, I, A, M_final, labels_to_decisions, 3)
    test_data = generate_data(L, max(I, 20000), A, M_final, labels_to_decisions, 3)

    return SimulationDataInstance(
        df_train=train_data,
        df_test=test_data,
        decision_to_ratings_map=decision_to_ratings_map,
        common_confusion=M_common,
        reviewer_confusions=M_final,
    )

