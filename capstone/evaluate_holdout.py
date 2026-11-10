"""Reproducible rating holdout evaluation for books_review_2.csv.

Uses only the Python standard library so the split and reported metrics can be
reproduced without relying on the notebook's saved, in-sample RMSE output.
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {"Title", "review/score", "user_num"}


def _read_ratings(
    path: str | Path,
) -> tuple[list[tuple[str, str, float]], int, int]:
    raw_rows: list[tuple[str, str, float]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            user = (row.get("user_num") or "").strip()
            title = (row.get("Title") or "").strip()
            score = (row.get("review/score") or "").strip()
            if not user and not title and not score:
                continue
            if not user or not title:
                raise ValueError(f"Missing user_num or Title on CSV row {row_number}")
            try:
                rating = float(score)
            except ValueError as exc:
                raise ValueError(f"Invalid review/score on CSV row {row_number}: {score!r}") from exc
            if not 1.0 <= rating <= 5.0:
                raise ValueError(f"Rating outside 1..5 on CSV row {row_number}: {rating}")
            raw_rows.append((user, title, rating))

    if not raw_rows:
        raise ValueError("No rating rows found")

    # The notebooks use pandas.DataFrame.pivot_table, which averages multiple
    # ratings from the same user for the same title. Mirror that before splitting.
    pair_sums: dict[tuple[str, str], float] = defaultdict(float)
    pair_counts: Counter[tuple[str, str]] = Counter()
    for user, title, rating in raw_rows:
        pair_sums[(user, title)] += rating
        pair_counts[(user, title)] += 1
    rows = [
        (user, title, pair_sums[(user, title)] / pair_counts[(user, title)])
        for user, title in sorted(pair_sums)
    ]
    duplicate_pairs = sum(count > 1 for count in pair_counts.values())
    return rows, len(raw_rows), duplicate_pairs


def _user_leave_one_out(
    rows: list[tuple[str, str, float]], seed: int
) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """Hold one rating per eligible user, keeping every item in training."""
    by_user: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    item_counts = Counter(title for _, title, _ in rows)
    for row in rows:
        by_user[row[0]].append(row)

    split_rng = random.Random(seed)
    test: list[tuple[str, str, float]] = []
    train: list[tuple[str, str, float]] = []
    for user in sorted(by_user):
        user_rows = list(by_user[user])
        if len(user_rows) < 2:
            train.extend(user_rows)
            continue

        split_rng.shuffle(user_rows)
        held_out = None
        for candidate in user_rows:
            if item_counts[candidate[1]] > 1:
                held_out = candidate
                break
        if held_out is None:
            train.extend(user_rows)
            continue

        test.append(held_out)
        item_counts[held_out[1]] -= 1
        train.extend(row for row in user_rows if row is not held_out)

    if not test:
        raise ValueError("Could not create a holdout set")
    return train, test


def _errors(actual: Iterable[float], predicted: Iterable[float]) -> tuple[float, float]:
    pairs = list(zip(actual, predicted))
    if not pairs:
        raise ValueError("No predictions to score")
    mse = sum((truth - estimate) ** 2 for truth, estimate in pairs) / len(pairs)
    mae = sum(abs(truth - estimate) for truth, estimate in pairs) / len(pairs)
    return math.sqrt(mse), mae


def _fit_matrix_factorization(
    train: list[tuple[str, str, float]],
    users: list[str],
    items: list[str],
    *,
    factors: int,
    epochs: int,
    learning_rate: float,
    regularization: float,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Fit a small SGD factor model with the notebook's dot-product objective."""
    rng = random.Random(seed)
    scale = 1.0 / factors
    p = {user: [rng.gauss(0.0, scale) for _ in range(factors)] for user in users}
    q = {item: [rng.gauss(0.0, scale) for _ in range(factors)] for item in items}
    for _ in range(epochs):
        for user, item, rating in train:
            pu = p[user]
            qi = q[item]
            estimate = sum(a * b for a, b in zip(pu, qi))
            error = rating - estimate
            # Update user factors first, matching the notebook's SGD order.
            for k in range(factors):
                pu[k] += learning_rate * (error * qi[k] - regularization * pu[k])
            for k in range(factors):
                qi[k] += learning_rate * (error * pu[k] - regularization * qi[k])

    return p, q


def evaluate(
    path: str | Path,
    *,
    seed: int = 42,
    factors: int = 50,
    epochs: int = 200,
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    model_seed: int = 1,
) -> dict[str, float | int | str]:
    """Run deterministic user-wise holdout evaluation and return metrics."""
    rows, source_review_rows, duplicate_pairs = _read_ratings(path)
    train, test = _user_leave_one_out(rows, seed)
    train_users = {user for user, _, _ in train}
    train_items = {item for _, item, _ in train}
    if any(user not in train_users or item not in train_items for user, item, _ in test):
        raise RuntimeError("Holdout split left a cold-start user or item")

    global_mean = sum(rating for _, _, rating in train) / len(train)
    user_sums = Counter()
    user_counts = Counter()
    for user, _, rating in train:
        user_sums[user] += rating
        user_counts[user] += 1

    actual = [rating for _, _, rating in test]
    global_predictions = [global_mean] * len(test)
    user_mean_predictions = [user_sums[user] / user_counts[user] for user, _, _ in test]

    train_for_fit = sorted(train, key=lambda row: (int(row[0]), row[1]))
    p, q = _fit_matrix_factorization(
        train_for_fit,
        sorted(train_users, key=int),
        sorted(train_items),
        factors=factors,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        seed=model_seed,
    )
    mf_raw = [sum(a * b for a, b in zip(p[user], q[item])) for user, item, _ in test]
    mf_clipped = [min(5.0, max(1.0, score)) for score in mf_raw]

    global_rmse, global_mae = _errors(actual, global_predictions)
    user_rmse, user_mae = _errors(actual, user_mean_predictions)
    mf_raw_rmse, mf_raw_mae = _errors(actual, mf_raw)
    mf_clip_rmse, mf_clip_mae = _errors(actual, mf_clipped)

    return {
        "source_review_rows": source_review_rows,
        "user_item_pairs": len(rows),
        "duplicate_user_item_pairs_aggregated": duplicate_pairs,
        "train_rows": len(train),
        "test_rows": len(test),
        "users_in_test": len({user for user, _, _ in test}),
        "items_in_test": len({item for _, item, _ in test}),
        "seed": seed,
        "model_seed": model_seed,
        "factors": factors,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "regularization": regularization,
        "global_mean_rmse": global_rmse,
        "global_mean_mae": global_mae,
        "user_mean_rmse": user_rmse,
        "user_mean_mae": user_mae,
        "mf_raw_rmse": mf_raw_rmse,
        "mf_raw_mae": mf_raw_mae,
        "mf_clipped_rmse": mf_clip_rmse,
        "mf_clipped_mae": mf_clip_mae,
        "mf_raw_prediction_min": min(mf_raw),
        "mf_raw_prediction_max": max(mf_raw),
        "rating_scale": "1..5",
    }


if __name__ == "__main__":
    default_path = Path(__file__).with_name("books_review_2.csv")
    results = evaluate(default_path)
    for key, value in results.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")
