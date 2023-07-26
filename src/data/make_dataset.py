import pandas as pd
from glob import glob

"""
# Read single CSV file
acc_data = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
gyro_data = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)


# List all data in data/raw/MetaMotion
files = glob("../../data/raw/MetaMotion/*.csv")

# Extract features from filename
f = files[1]

data_path = "../../data/raw/MetaMotion\\"

participant = f.split("-")[0].replace(data_path, "")
label = f.split("-")[1]
category = f.split("-")[2].replace("_MetaWear_2019", "").rstrip("123")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category


# Read all files
acc_df = pd.DataFrame()
gyro_df = pd.DataFrame()

acc_set = 1
gyro_set = 1

for f in files:
    participant = f.split("-")[0].replace(data_path, "")
    label = f.split("-")[1]
    category = f.split("-")[2].replace("_MetaWear_2019", "").rstrip("123")

    df = pd.read_csv(f)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in f:
        df["set"] = acc_set
        acc_set += 1
        acc_df = pd.concat([acc_df, df])
    elif "Gyroscope" in f:
        df["set"] = gyro_set
        gyro_set += 1
        gyro_df = pd.concat([gyro_df, df])
    else:
        print(f)
        break


# Working with datetimes
acc_df.info()

pd.to_datetime(acc_df["epoch (ms)"], unit="ms").dt.week

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

acc_df = acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1)
gyro_df = gyro_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1)
"""

# Turn into function
files = glob("../../data/raw/MetaMotion/*.csv")


def get_data_from_files(files):
    acc_df = pd.DataFrame()
    gyro_df = pd.DataFrame()

    acc_set = 1
    gyro_set = 1

    for f in files:
        data_path = "../../data/raw/MetaMotion\\"
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].replace("_MetaWear_2019", "").rstrip("123")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        elif "Gyroscope" in f:
            df["set"] = gyro_set
            gyro_set += 1
            gyro_df = pd.concat([gyro_df, df])

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyro_df.index = pd.to_datetime(gyro_df["epoch (ms)"], unit="ms")

    acc_df = acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1)
    gyro_df = gyro_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis=1)

    return acc_df, gyro_df


acc_df, gyro_df = get_data_from_files(files)


# Merging datasets
data_merged = pd.concat([acc_df.iloc[:, :3], gyro_df], axis=1)
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]


# Resample data (frequency conversion)
# Accelerometer:    12.500HZ (Measures every 1/12.5(0.08) seconds)
# Gyroscope:        25.000Hz (Measures every 1/25(0.04) seconds)
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

data_merged[:1000].resample(rule="200ms").apply(sampling)

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)

data_resampled["set"] = data_resampled["set"].astype("int64")
data_resampled.info()


# Export dataset
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
