import pandas as pd

# Loading the dataset
df = pd.read_csv("sample_temperature-ELM_v1.0_optimizationComp_data.csv")
df.columns = df.columns.str.strip()

# Dropping the temperature in Fahrenheit
df = df.drop(columns=["_18b20_Temp_Fh"])

# Creating the time variable (assuming: 1 serial = 1 minute)
delta_t = 1  # minute
df["time_min"] = df["Serial_Reading"] * delta_t

# Renaming columns 
df = df.rename(columns={
    "Serial_Reading": "serial",
    "_18b20_Temp_C": "ambient_temp_C",
    "Object_Temperature": "object_temp_C",
    "Current_humidity": "humidity",
    "NW_cooling": "cooling_state",
    "Critical_Level": "critical_level"
})

# Checking for missing values
missing_values = df.isnull().sum()
print(missing_values)

# Saving the preprocessed dataset
df.to_csv("dataset_preprocessed.csv", index=False)

# Final preview
print(df.head())
