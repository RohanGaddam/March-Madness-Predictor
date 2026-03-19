import numpy as np
import matplotlib.pyplot as plt

# Replace these with your real arrays (same length)
#years = np.array([0,   1,     10,     33,      50,     100,    1000,   500_000, 1_000_000])
#co2_ppm = np.array([280.0, 280.55, 280.75, 280.969, 280.95, 280.90, 280.50, 280.00, 280.00])           # atmospheric CO2 (ppm)
#temp_anom = np.array([0.0,  0.0008, 0.004,  0.007,   0.010,  0.010,  0.007,  0.0,     0.0])         # surface temperature anomaly (°C)
years     = [0,   1,     10,     33,      50,     100,    1000,   500_000, 1_000_000]
co2_ppm   = [280.0, 280.55, 280.75, 280.969, 280.95, 280.90, 280.50, 280.00, 280.00]
temp_anom = [0.0,  0.0008, 0.004,  0.007,   0.010,  0.010,  0.007,  0.0,     0.0]

fig, ax1 = plt.subplots(figsize=(9,4))

# CO2 on left y-axis
l1, = ax1.plot(years, co2_ppm, label="CO₂ (ppm)")
ax1.set_xlabel("Year")
ax1.set_ylabel("CO₂ (ppm)")

# Temperature on right y-axis
ax2 = ax1.twinx()
l2, = ax2.plot(years, temp_anom, label="Surface temperature anomaly (°C)")
ax2.set_ylabel("Temperature anomaly (°C)")

# Find and label peaks
i_co2_max = np.argmax(co2_ppm)
i_t_max = np.argmax(temp_anom)

ax1.scatter(years[i_co2_max], co2_ppm[i_co2_max])
ax1.annotate(f"{years[i_co2_max]}: {co2_ppm[i_co2_max]:.2f} ppm",
             (years[i_co2_max], co2_ppm[i_co2_max]),
             xytext=(10,10), textcoords="offset points",
             arrowprops=dict(arrowstyle="->"))

ax2.scatter(years[i_t_max], temp_anom[i_t_max])
ax2.annotate(f"{years[i_t_max]}: {temp_anom[i_t_max]:.3f} °C",
             (years[i_t_max], temp_anom[i_t_max]),
             xytext=(10,-15), textcoords="offset points",
             arrowprops=dict(arrowstyle="->"))

# One combined legend
lines = [l1, l2]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="best")

plt.tight_layout()
plt.show()
