import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from utils.tools import *

root = os.getcwd()
measurement = "Gate Width"
date = "8_18_26"
file_name = "width_0.SPE"
atom = "Nd"
binder = None

N_points = 100

gate_width_start = 10
gate_width_end = 500
N_params = 1000
gate_widths = np.linspace(gate_width_start, gate_width_end, N_params)


fix_spectrum = True
degree = 1

data_path = os.path.join(root, measurement)
data_path = os.path.join(data_path, date)
file_path = os.path.join(data_path, file_name)

print(f"Reading File: {file_name}")
wavelength, spectrum = read_file(file_path, fix_spectrum = fix_spectrum, degree = degree)
N, M = spectrum.shape
idx = N // 2
atom_lines_df = read_reference_lines(atom, wavelength)

lamb0 = select_peak(wavelength, spectrum[idx], atom_lines_df)

pixel0 = np.argmin(np.abs(wavelength - lamb0))
start = pixel0 - N_points // 2
end = pixel0 + N_points // 2 + 1

x = wavelength[start : end]
y = spectrum[N // 2, start : end].astype(float)
point1, point2 = two_point_selection(x, y, lamb0, N_points = N_points)
plot_area(x, y, lamb0, point1, point2)

areas = []
snrs = []
for data in spectrum:
    y = data[start : end].astype(float)
    area = compute_area(x, y, point1, point2)
    snr = compute_snr(x, y, point1, point2)
    areas.append(area)
    snrs.append(snr)

areas = np.array(areas)
snrs = np.array(snrs)


fig, ax = plt.subplots(1)
ax.set_title("Gate Width Optimization")
ax.plot(gate_widths, areas, color = "black")

ax.set_xlabel("Gate Width (ns)")
ax.set_ylabel("Peak Area")

ax.grid(True)
plt.show()

fig, ax = plt.subplots(1)
ax.set_title("Gate Width Optimization")
ax.plot(gate_widths, snrs, color = "black")

ax.set_xlabel("Gate Width (ns)")
ax.set_ylabel("SNR")

ax.grid(True)
plt.show()