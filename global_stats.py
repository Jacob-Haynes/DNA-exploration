import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    fbeta_score,
)


def calculate_total_rows(
    data,
):
    return len(data)


def calculate_total_dna_rows(
    data,
):
    return (data["AppointmentStatus"] == "DNA").sum()


def calculate_total_non_dna_rows(
    data,
):
    total_rows = calculate_total_rows(data)
    total_dna_rows = calculate_total_dna_rows(data)
    return total_rows - total_dna_rows


def calculate_dna_rate(
    data,
):
    total_rows = calculate_total_rows(data)
    total_dna_rows = calculate_total_dna_rows(data)
    return (total_dna_rows / total_rows) * 100


def calculate_predicted_dna_rows(
    data,
):
    return (data["DNARawProbability"] >= 0.5).sum()


def calculate_predicted_non_dna_rows(
    data,
):
    total_rows = calculate_total_rows(data)
    predicted_dna_rows = calculate_predicted_dna_rows(data)
    return total_rows - predicted_dna_rows


def calculate_model_dna_rate(
    data,
):
    total_rows = calculate_total_rows(data)
    predicted_dna_rows = calculate_predicted_dna_rows(data)
    return (predicted_dna_rows / total_rows) * 100


def calculate_true_positives(
    data,
):
    return ((data["DNARawProbability"] >= 0.5) & (data["AppointmentStatus"] == "DNA")).sum()


def calculate_true_negatives(
    data,
):
    return ((data["DNARawProbability"] < 0.5) & (data["AppointmentStatus"] != "DNA")).sum()


def calculate_false_positives(
    data,
):
    return ((data["DNARawProbability"] >= 0.5) & (data["AppointmentStatus"] != "DNA")).sum()


def calculate_false_negatives(
    data,
):
    return ((data["DNARawProbability"] < 0.5) & (data["AppointmentStatus"] == "DNA")).sum()


def calculate_global_accuracy(
    data,
):
    total_rows = calculate_total_rows(data)
    true_positives = calculate_true_positives(data)
    true_negatives = calculate_true_negatives(data)
    return (true_positives + true_negatives) / total_rows


def calculate_global_precision(
    data,
):
    true_positives = calculate_true_positives(data)
    false_positives = calculate_false_positives(data)
    return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0


def calculate_global_cm(
    data,
):
    true_positives = calculate_true_positives(data)
    false_negatives = calculate_false_negatives(data)
    false_positives = calculate_false_positives(data)
    true_negatives = calculate_true_negatives(data)
    return np.array(
        [
            [
                true_negatives,
                false_positives,
            ],
            [
                false_negatives,
                true_positives,
            ],
        ]
    )


def calculate_global_auc(
    data,
):
    return roc_auc_score(
        data["AppointmentStatus"] == "DNA",
        data["DNARawProbability"],
    )


def calculate_global_f0_5(
    data,
):
    return fbeta_score(
        data["AppointmentStatus"] == "DNA",
        data["DNARawProbability"] >= 0.5,
        beta=0.5,
    )
