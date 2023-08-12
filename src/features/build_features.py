import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")
df.info()
predictor_columns = list(df.columns[:6])

# Plot settings
mpl.style.use("fivethirtyeight")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

df[df["set"] == 20]["gyr_y"].plot()
# The gaps in the plot represent missing data

for column in predictor_columns:
    df[column] = df[column].interpolate()
df.info()  # No missing values

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

# Medium represents 10 reps and heavy represents 5 reps
df[df["set"] == 13]["acc_y"].plot()  # 5 reps (heavy ohp)
df[df["set"] == 70]["acc_y"].plot()  # 10 reps (medium bench)

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]

sets = list(df.sort_values("set")["set"].unique())
for set in sets:
    start = df[df["set"] == set].index[0]
    end = df[df["set"] == set].index[-1]
    duration = end - start
    df.loc[df["set"] == set, "duration"] = duration.seconds

category_duration = df.groupby("category")["duration"].mean()
category_duration.info()

category_duration["heavy"] / 5
category_duration["medium"] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

# Sample frequency (Number of instances per second(Data in df is sampled by 200 ms))
fs = 1000 / 200
# The higher the cutoff the less smoother is the data
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(
    df_lowpass, "acc_y", fs, cutoff, order=5, phase_shift=True
)
subset = df_lowpass[df_lowpass["set"] == 90]  # Medium row

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="Raw Data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="Butterworth Filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for column in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(
        df_lowpass, column, fs, cutoff, order=5, phase_shift=True
    )
    df_lowpass[column] = df_lowpass[column + "_lowpass"]
    del df_lowpass[column + "_lowpass"]

df_lowpass

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pc_values) + 1), pc_values)
plt.xlabel("Principal Component Numbers")
plt.ylabel("Explained Variance")
plt.legend()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca["set"] == 90]
subset[["pca_1", "pca_2", "pca_3"]].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

subset = df_squared[df_squared["set"] == 28]
subset[["acc_r", "gyr_r"]].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

# Calculates the rolling average for every 1 second
ws = int(1000 / 200)
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

df_temporal = NumAbs.abstract_numerical(df_temporal, predictor_columns, ws, "mean")
df_temporal = NumAbs.abstract_numerical(df_temporal, predictor_columns, ws, "std")

df_temporal.info()

df_temporal = df_squared.copy()
sets = df.sort_values("set")["set"].unique()
df_temporal_list = []
for set in sets:
    subset = df_temporal[df_temporal["set"] == set].copy()
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "mean")
    subset = NumAbs.abstract_numerical(subset, predictor_columns, ws, "std")
    df_temporal_list.append(subset)

df_temporal = pd.concat(df_temporal_list)
df_temporal.info()

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# --------------------------------------------------------------
# Frequency features (Fourier Transformation)
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

# Sample Frequency
fs = int(1000 / 200)

# Window Size (Average time taken for one rep in ms divided by the sampling rate)
ws = int(2800 / 200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

subset = df_freq[df_freq["set"] == 15]
subset[
    [
        "acc_y",
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_0.357_Hz_ws_14",
        "acc_y_freq_0.714_Hz_ws_14",
    ]
].plot()

df_freq_list = []
sets = df.sort_values("set")["set"].unique()

for set in sets:
    print(f"Applying Fourier Transformation to set {set}")
    subset = df_freq[df_freq["set"] == set].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

# Elbow method
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=42)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plotting clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
clusters = df_cluster.sort_values("cluster")["cluster"].unique()
for cluster in clusters:
    subset = df_cluster[df_cluster["cluster"] == cluster]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=cluster)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()

# Comparing accelerometer data across labels
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")
labels = df_cluster["label"].unique()
for label in labels:
    subset = df_cluster.query(f"label == '{label}'")
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=label)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
