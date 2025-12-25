import pandas as pd
import numpy as np

# 1. Load preprocessed dataset
df = pd.read_csv("dataset_preprocessed.csv", sep=";")

temp_int = df["object_temp_C"].values
temp_ext = df["ambient_temp_C"].values

dt = 1.0                 # 1 minute per sample
threshold = -8.0         # temperature threshold
short_limit = 10         # minutes
critical_limit = 30      # minutes

# 2. Feature engineering
df["ambient_slope"] = df["ambient_temp_C"].diff().fillna(0)
df["ambient_rolling_mean"] = (
    df["ambient_temp_C"].rolling(5).mean().fillna(method="bfill")
)


# 3. Label generation (0 / 1 / 2)
labels = []
counter = 0

for temp in temp_int:
    if temp > threshold:
        counter += dt
        if counter < short_limit:
            labels.append(0)      # OK 
        elif counter < critical_limit:
            labels.append(1)      # Rupture courte
        else:
            labels.append(2)      # Rupture critique
    else:
        counter = 0
        labels.append(0)          # OK

df["class"] = labels

# 4. Save dataset for ML
df.to_csv("dataset_features_ml.csv", index=False, sep=";")

print("Feature engineering completed.")
print(df[["ambient_temp_C", "ambient_slope", "ambient_rolling_mean", "class"]].head())

