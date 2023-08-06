import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction

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
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
