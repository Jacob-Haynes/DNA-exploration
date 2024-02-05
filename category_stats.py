import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    fbeta_score,
    confusion_matrix,
)


def calculate_category_stats(
    data,
):
    y_true = (data["AppointmentStatus"] == "DNA").astype(int)
    y_pred = (data["DNARawProbability"] >= 0.5).astype(int)

    # confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    # Calculate stats
    y_true_count_1 = y_true.sum()
    y_true_count_0 = len(y_true) - y_true_count_1
    y_pred_count_1 = y_pred.sum()
    y_pred_count_0 = len(y_pred) - y_pred_count_1

    precision = (
        cm[
            1,
            1,
        ]
        / (
            cm[
                1,
                1,
            ]
            + cm[
                0,
                1,
            ]
        )
        if (
            cm[
                1,
                1,
            ]
            + cm[
                0,
                1,
            ]
        )
        > 0
        else 0
    )
    actual_probabilities = data["DNARawProbability"]
    auc = roc_auc_score(
        y_true,
        actual_probabilities,
    )
    f0_5 = fbeta_score(
        y_true,
        y_pred,
        beta=0.5,
    )

    return {
        "y_true_count_1": y_true_count_1,
        "y_true_count_0": y_true_count_0,
        "y_pred_count_1": y_pred_count_1,
        "y_pred_count_0": y_pred_count_0,
        "precision": precision,
        "auc": auc,
        "f0_5": f0_5,
        "confusion_matrix": cm,
    }
