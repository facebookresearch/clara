#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def generate_score(
    true_ratings: np.array, score_means: np.array, score_stdvs: np.array
):
    num_items = len(true_ratings)
    num_ones = np.sum(true_ratings)
    num_zeros = num_items - num_ones

    logger.info(f"num_items = {num_items}")
    logger.info(f"num_ones = {num_ones}")
    logger.info(f"num_zeros = {num_zeros}")

    logger.info(f"score_means = {score_means}")
    logger.info(f"score_stdvs = {score_stdvs}")

    score_zeros = np.random.normal(score_means[0], score_stdvs[0], num_zeros)
    score_ones = np.random.normal(score_means[1], score_stdvs[1], num_ones)

    df = pd.DataFrame({"true_rating": true_ratings.tolist()})
    df["score"] = 0
    df["score"] = df["score"].astype(float)
    df.loc[df.true_rating == 0, "score"] = score_zeros
    df.loc[df.true_rating == 1, "score"] = score_ones

    scores = np.empty((num_items, 2), dtype=float)
    scores[:, 1] = np.exp(df["score"]) / (1 + np.exp(df["score"]))
    scores[:, 0] = 1.0 - scores[:, 1]

    return scores


def generate_dataset_tiebreaking(
    dataset_id: int, theta: np.array, psi: np.array, num_items: int
) -> pd.DataFrame:
    """
    Function to generate one dataset for a particular dataset_id which has
        - 2 labels for each item where these labels agree,
        and 3 labels if there is a disagreement
        - one single confusion matrix

    Args:
        dataset_id: a numeric id for this dataset
        theta: the true prevalence
        psi: the true confusion matrix shared by all labelers
        num_items: number of items in the dataset
    """
    # Set dataset_ids and item ids to be {dataset_id}_{item_num}
    ids = ["{}_{}".format(str(dataset_id), str(x)) for x in range(num_items)]
    dids = [str(dataset_id) for i in range(num_items)]

    # Randomly choose item labels and set labelers to be the same for each dataset_id
    ys = np.random.choice(len(theta), size=num_items, p=theta)

    # For each item generate random labels based on the psi confusion matrix
    ratings = []
    labelers = []
    for i in range(num_items):
        item_rating = np.random.choice(len(theta), size=3, p=psi[ys[i]])
        if item_rating[0] == item_rating[1]:
            ratings.append([item_rating[0], item_rating[1]])
        else:
            ratings.append(list(item_rating))
        labelers.append([dataset_id] * len(ratings[i]))

    data = [dids, ids, labelers, ratings, ys.tolist()]
    columns = ["dataset", "id", "labelers", "ratings", "true_rating"]
    df = pd.DataFrame(data=data).transpose()
    df.columns = columns

    return df


def generate_dataset_tiebreaking_with_scores(
    dataset_id: int, theta: np.array, psi: np.array, num_items: int
) -> pd.DataFrame:

    df = generate_dataset_tiebreaking(dataset_id, theta, psi, num_items)

    C = 2  # number of classifiers
    R = 2  # binary labels
    scores = np.empty((num_items, C, R), dtype=float)

    # generate classifiers cores
    scores[:, 0, :] = generate_score(
        true_ratings=df.true_rating,
        score_means=np.array([-4, 4]),
        score_stdvs=np.array([1, 1]),
    )

    scores[:, 1, :] = generate_score(
        true_ratings=df.true_rating,
        score_means=np.array([-1, 1]),
        score_stdvs=np.array([1, 1]),
    )
    logger.info(f"scores.shape = {scores.shape}")

    df["scores"] = scores.tolist()
    return df


def generate_dataset_tiebreaking_different_labeler_cm(
    dataset_id: int, theta: np.array, psi: np.array, num_items: int
) -> pd.DataFrame:
    """
    Function to generate one dataset for a particular dataset_id which has
        - 2 labels for each item where these labels agree,
        and 3 labels if there is a disagreement
        - a list of confusion matrix
    Args:
        dataset_id: a numeric id for this dataset
        theta: the true prevalence
        psi: the list of 2x2 confusion matrix for different labelers
        num_items: number of items in the dataset
    """
    num_labelers = len(psi)
    if num_labelers < 3:
        raise Exception("Sorry, number of labelers need to be larger than 3!")

    # set dataset_ids and item ids to be {dataset_id}_{item_num}
    ids = ["{}_{}".format(str(dataset_id), str(x)) for x in range(num_items)]
    dids = [str(dataset_id) for i in range(num_items)]

    # randomly choose item labels and set labelers to be the same for each dataset_id
    ys = np.random.choice(len(theta), size=num_items, p=theta)

    item_ratings_all = []
    labelers_all = []
    for i in range(num_items):
        # randomly select labelers for each item
        labelers = np.random.choice(range(0, num_labelers), size=3, replace=False)
        item_ratings = []
        for j in range(3):
            labeler = labelers[j]
            rating = np.random.choice(len(theta), p=psi[labeler][ys[i]])
            item_ratings.append(rating)
        if item_ratings[0] == item_ratings[1]:
            item_ratings_all.append([item_ratings[0], item_ratings[1]])
            labelers_all.append([labelers[0], labelers[1]])
        else:
            item_ratings_all.append(item_ratings)
            labelers_all.append(labelers)

    data = [dids, ids, labelers_all, item_ratings_all, ys.tolist()]
    columns = ["dataset", "id", "labelers", "ratings", "true_rating"]
    df = pd.DataFrame(data=data).transpose()
    df.columns = columns

    return df
