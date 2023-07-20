import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


# Load data
df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


# Plot single columns
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"].reset_index(drop=True))


# Plot all exercises
for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# Adjust plot settings
mpl.style.use("seaborn-v0_8-talk")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100


# Compare medium vs. heavy sets
category_df = df.query("participant == 'A'").query("label == 'squat'").reset_index()
fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.legend()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")


# Compare participants
participant_df = df.query("label == 'row'").sort_values("participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.legend()
ax.set_xlabel("Samples")
ax.set_ylabel("acc_y")


# Plot multiple axis
label = "dead"
participant = "A"
all_axis_df = (
    df.query(f"participant == '{participant}'")
    .query(f"label == '{label}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]][:100].plot(ax=ax)
ax.set_xlabel("Samples")
ax.set_ylabel("Acceleromete values")
plt.legend()
plt.show()


# Create a loop to plot all combinations per sensor
labels = df["label"].unique()
participants = df.sort_values("participant")["participant"].unique()

# 1. Accelerometer Data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[:100][["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("Acc values")
            plt.title(f"{label}({participant})")
            plt.legend()
            plt.show()

# 2. Gyroscope Data
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[:100][["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_xlabel("Samples")
            ax.set_ylabel("Gyr values")
            plt.title(f"{label}({participant})")
            plt.legend()
            plt.show()


# Combine plots in one figure
label = "squat"
participant = "A"
combined_plot_df = (
    df.query(f"participant == '{participant}'")
    .query(f"label == '{label}'")
    .reset_index()
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=3, fancybox=True, shadow=True
)
ax[1].legend(
    loc="upper center", bbox_to_anchor=(0.5, 1.15), ncols=3, fancybox=True, shadow=True
)
ax[1].set_xlabel("Samples")


# Loop over all combinations and export for both sensors
labels = df["label"].unique()
participants = df.sort_values("participant")["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index(drop=True)
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncols=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncols=3,
                fancybox=True,
                shadow=True,
            )
            ax[1].set_xlabel("Samples")

            plt.savefig(f"../../reports/figures/{label} ({participant}).png")
            plt.show()
