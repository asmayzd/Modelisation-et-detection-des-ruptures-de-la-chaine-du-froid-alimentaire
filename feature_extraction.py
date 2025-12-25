import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("dataset_preprocessed.csv", sep=";")

time = df["time_min"].values
temp_int = df["object_temp_C"].values
temp_ext = df["ambient_temp_C"].values

# Regulatory threshold (example: -8 °C)
threshold = -8.0

# 1. Max and min internal temperature
temp_max = np.max(temp_int)
temp_min = np.min(temp_int)

# 2. Duration above threshold
dt = 1.0  # 1 minute
duration_out = np.sum(temp_int > threshold) * dt

# 3. Maximum slope (absolute)
slopes = np.diff(temp_int) / dt
max_slope = np.max(np.abs(slopes))

# 4. Mean difference internal / external
mean_gap = np.mean(temp_int - temp_ext)

# Print results
print("Extracted features:")
print(f"Max internal temperature: {temp_max:.2f} °C")
print(f"Min internal temperature: {temp_min:.2f} °C")
print(f"Duration above threshold: {duration_out:.0f} minutes")
print(f"Maximum slope: {max_slope:.2f} °C/min")
print(f"Mean internal-external gap: {mean_gap:.2f} °C")
