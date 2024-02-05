from global_stats import *
from plots import *
from category_stats import *
from threshold_optimisation import *

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

# Global Vars
bins = [
    0,
    0.5,
    0.7,
    0.9,
    1,
]
labels = [
    "Low",
    "Medium",
    "High",
    "Very High",
]

# Calculate statistics
total_rows = calculate_total_rows(data)
total_dna_rows = calculate_total_dna_rows(data)
total_non_dna_rows = calculate_total_non_dna_rows(data)
dna_rate = calculate_dna_rate(data)
predicted_dna_rows = calculate_predicted_dna_rows(data)
predicted_non_dna_rows = calculate_predicted_non_dna_rows(data)
model_dna_rate = calculate_model_dna_rate(data)
global_cm = calculate_global_cm(data)
global_accuracy = calculate_global_accuracy(data)
global_precision = calculate_global_precision(data)
global_auc = calculate_global_auc(data)
global_f0_5 = calculate_global_f0_5(data)

# Print stats
print(f"Total number of appointments: {total_rows}")
print(f"Total number of appointments marked DNA: {total_dna_rows}")
print(f"Total number that are not DNA: {total_non_dna_rows}")
print(f"DNA rate: {dna_rate:.2f}%")
print()
print(f"Number of Predicted DNAs: {predicted_dna_rows}")
print(f"Number of Predicted Non-DNAs: {predicted_non_dna_rows}")
print(f"Model DNA rate: {model_dna_rate:.2f}%")
print()
print(f"Model Accuracy: {global_accuracy:.2f}")
print(f"Model Precision: {global_precision:.2f}")
print(f"Model ROC AUC: {global_auc:.2f}")
print(f"F0.5 score: {global_f0_5:.2f}")
print()

# plot global charts
plt.figure(
    figsize=(
        15,
        10,
    )
)
plt.subplot(
    2,
    2,
    1,
)
plot_data_imbalance(data)
plt.subplot(
    2,
    2,
    2,
)
plot_confusion_matrix(global_cm)
plt.subplot(
    2,
    2,
    3,
)
plot_distribution(data)
plt.subplot(
    2,
    2,
    4,
)
plot_impact_categories(
    data,
    bins,
    labels,
)

# Category Confusion matrices
plt.figure(
    figsize=(
        15,
        5,
    )
)

for (
    i,
    category,
) in enumerate(labels):
    catData = data[(data["ProbabilityCategory"] == category)]
    # Calculate category-specific statistics
    stats = calculate_category_stats(catData)
    # Print stats
    print(f"Category: {category}")
    print(f'y_true - Count of DNAs: {stats["y_true_count_1"]}, Count of Attended: {stats["y_true_count_0"]}')
    print(f'y_pred - Count of Predicted DNAs: {stats["y_pred_count_1"]}, ' f'Count of Predicted Attended: {stats["y_pred_count_0"]}')
    print(f'Precision: {stats["precision"]:.2f}')
    print(f'AUC: {stats["auc"]:.2f}')
    print(f'F0.5: {stats["f0_5"]:.2f}')
    print()

    # Create subplots for each category
    plt.subplot(
        1,
        len(labels),
        i + 1,
    )
    plot_confusion_matrix(
        stats["confusion_matrix"],
        "category",
    )

# Threshold Optimisation
print("Calculating threshold...")
result = optimise_threshold(data)

optimal_cm = confusion_matrix(
    data["AppointmentStatus"] == "DNA",
    (data["DNARawProbability"] >= result["Threshold"]).astype(int),
)

plt.figure(
    figsize=(
        5,
        5,
    )
)
plot_confusion_matrix(optimal_cm)
plt.title("Optimal Confusion Matrix")

print(result)

# Show plot
plt.tight_layout()
plt.show()
