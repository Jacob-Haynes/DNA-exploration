import numpy as np
import pandas as pd
from matplotlib import (
    pyplot as plt,
)

print("Loading data...")
data = pd.read_csv(
    "../data/DNA Prediction Export LUHFT.csv",
    usecols=[
        "AppointmentId",
        "DNARawProbability",
        "DNARank",
        "ContactStatus",
        "AppointmentStatus",
    ],
    skipfooter=1,
    engine="python",
)

# Clean Data
data = data[data["AppointmentStatus"] != "canceled"]


# Clean Data
data = data[data["AppointmentStatus"] != "canceled"]

# Add prediction labels based on the threshold
threshold = 0.7
data["Prediction"] = np.where(
    data["DNARawProbability"] >= threshold,
    "DNA",
    "Attend",
)

# Normalize the DNARawProbability for DNA and Attend separately
dna_data = data[data["Prediction"] == "DNA"]
attended_data = data[data["Prediction"] == "Attend"]

max_dna_prob = dna_data["DNARawProbability"].max()
min_dna_prob = dna_data["DNARawProbability"].min()
max_attended_prob = attended_data["DNARawProbability"].max()
min_attended_prob = attended_data["DNARawProbability"].min()

dna_data["NormalizedProbability"] = (dna_data["DNARawProbability"] - min_dna_prob) / (max_dna_prob - min_dna_prob)
attended_data["NormalizedProbability"] = (attended_data["DNARawProbability"] - min_attended_prob) / (max_attended_prob - min_attended_prob)

# Calculate DNA rate for each decile for DNA data
num_deciles = 10
for i in range(num_deciles):
    decile = i / num_deciles
    dna_decile_data = dna_data[(dna_data["NormalizedProbability"] >= decile) & (dna_data["NormalizedProbability"] < (decile + 0.1))]

    # Calculate true positives, false positives, true negatives, false negatives
    true_positives = len(dna_decile_data[(dna_decile_data["AppointmentStatus"] == "DNA") & (dna_decile_data["Prediction"] == "DNA")])
    false_positives = len(dna_decile_data[(dna_decile_data["AppointmentStatus"] != "DNA") & (dna_decile_data["Prediction"] == "DNA")])
    true_negatives = len(dna_decile_data[(dna_decile_data["AppointmentStatus"] != "DNA") & (dna_decile_data["Prediction"] == "Attend")])
    false_negatives = len(dna_decile_data[(dna_decile_data["AppointmentStatus"] == "DNA") & (dna_decile_data["Prediction"] == "Attend")])

    # Create the confusion matrix
    confusion = np.array(
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

    # Plot the confusion matrix
    plt.figure(
        figsize=(
            4,
            4,
        )
    )
    plt.imshow(
        confusion,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )
    plt.title(f"Confusion Matrix (Decile {i+1})")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(
        tick_marks,
        [
            "Attend",
            "DNA",
        ],
    )
    plt.yticks(
        tick_marks,
        [
            "Attend",
            "DNA",
        ],
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(confusion[i][j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white",
            )
    plt.show()
