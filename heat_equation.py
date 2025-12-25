import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Load preprocessed dataset
df = pd.read_csv("dataset_preprocessed.csv", sep=";")

time = df["time_min"].values
ambient_temp = df["ambient_temp_C"].values
object_temp = df["object_temp_C"].values

# 2. Physical and numerical parameters
L = 1.0              # length of the product (arbitrary unit)
nx = 20              # number of spatial points
dx = L / (nx - 1)

dt = 1.0             # time step (1 minute)
r = 0.4           # < 0.5 (stabilité)
alpha = r * dx**2 / dt

# 3. Initial condition
T = np.ones(nx) * object_temp[0]

# 4. Time integration
T_history = []

for n in range(len(time)):
    T_new = T.copy()

    for i in range(1, nx - 1):
        T_new[i] = T[i] + r * (T[i + 1] - 2*T[i] + T[i - 1])

    # Boundary conditions (ambient temperature at both ends)
    T_new[0] = ambient_temp[n]
    T_new[-1] = ambient_temp[n]

    T = T_new
    T_history.append(T.copy())

T_history = np.array(T_history)

# 5. Visualization
plt.figure(figsize=(8, 5))
N = 2000
plt.plot(time[:N], T_history[:N, nx // 2], label="Simulated internal temperature")
plt.plot(time[:N], object_temp[:N], "--", label="Measured internal temperature")
plt.xlabel("Time (minutes)")
plt.ylabel("Temperature (°C)")
plt.title("Heat equation simulation inside the product")
plt.legend()
plt.grid()
plt.show()


# 6. Quantitative comparison

# Temperature simulated at the center
T_sim_center = T_history[:, nx // 2]

# Mean Absolute Error (MAE)
mae = np.mean(np.abs(T_sim_center - object_temp))

print(f"Mean Absolute Error (MAE): {mae:.2f} °C")