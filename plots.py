import matplotlib.pyplot as plt
import pandas as pd


def plot_data_imbalance(data):
    appointment_status_counts = data["AppointmentStatus"].value_counts()
    plt.pie(
        appointment_status_counts,
        labels=appointment_status_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.axis("equal")
    plt.title("Appointment Status Distribution")


def plot_confusion_matrix(
    global_cm,
    category=None,
):
    plt.imshow(
        global_cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )
    plt.title(f"Confusion Matrix ({category})" if category else "Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{global_cm[i, j]:,}",
                ha="center",
                va="center",
                color="red",
                fontsize=12,
                fontweight="bold",
            )

    plt.xticks(
        [
            0,
            1,
        ],
        [
            "Predicted Attend",
            "Predicted DNA",
        ],
    )
    plt.yticks(
        [
            0,
            1,
        ],
        [
            "Attend",
            "DNA",
        ],
    )


def plot_distribution(
    data,
):
    bins = [i / 10 for i in range(11)]
    plt.hist(
        data["DNARawProbability"],
        bins=bins,
        edgecolor="k",
    )
    plt.xlabel("DNARawProbability")
    plt.ylabel("Number of Appointments")
    plt.title("Histogram of DNARawProbability in Deciles")


def plot_impact_categories(
    data,
    bins,
    labels,
):
    data["ProbabilityCategory"] = pd.cut(
        data["DNARawProbability"],
        bins=bins,
        labels=labels,
    )
    category_counts = data["ProbabilityCategory"].value_counts()
    category_counts = category_counts.reindex(labels)
    category_counts.plot(
        kind="bar",
        edgecolor="k",
    )
    plt.xlabel("Probability Category")
    plt.ylabel("Number of Appointments")
    plt.title("Histogram of Probability Categories")

    category_avg_prob = data.groupby(
        "ProbabilityCategory",
        observed=False,
    )["DNARawProbability"].mean()

    for (
        i,
        v,
    ) in enumerate(category_counts):
        plt.annotate(
            f"{v}\nAvg: {category_avg_prob[labels[i]]:.2f}",
            xy=(
                i,
                v,
            ),
            ha="center",
            va="bottom",
        )
