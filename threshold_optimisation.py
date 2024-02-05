import numpy as np
from sklearn.metrics import (
    precision_score,
    fbeta_score,
    auc,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)


def optimise_threshold(
    data,
    desired_recall=0.02,
    threshold_range=(
        0.4,
        1,
    ),
    threshold_step=0.01,
    beta=0.5,
):
    y_true = data["AppointmentStatus"] == "DNA"
    predicted_probabilities = data["DNARawProbability"]

    thresholds = np.arange(
        threshold_range[0],
        threshold_range[1] + threshold_step,
        threshold_step,
    )
    best_precision = 0
    best_threshold = threshold_range[0]
    best_recall = None
    best_f0_5 = None
    best_pr_auc = None
    best_roc_auc = None

    for threshold in thresholds:
        y_pred = (predicted_probabilities >= threshold).astype(int)
        precision = precision_score(
            y_true,
            y_pred,
        )
        recall = recall_score(
            y_true,
            y_pred,
        )

        if recall >= desired_recall and precision > best_precision:
            # calc stats
            f0_5 = fbeta_score(
                y_true,
                y_pred,
                beta=beta,
            )
            (
                auc_precision,
                auc_recall,
                _,
            ) = precision_recall_curve(
                y_true,
                predicted_probabilities >= threshold,
            )
            pr_auc = auc(
                auc_recall,
                auc_precision,
            )
            roc_auc = roc_auc_score(
                y_true,
                predicted_probabilities,
            )
            # record stats
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_f0_5 = f0_5
            best_pr_auc = pr_auc
            best_roc_auc = roc_auc

    result_dict = {
        "Threshold": best_threshold,
        "Precision": best_precision,
        "Recall": best_recall,
        "F0_5": best_f0_5,
        "PR_AUC": best_pr_auc,
        "ROC_AUC": best_roc_auc,
    }

    return result_dict
