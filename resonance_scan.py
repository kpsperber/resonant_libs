import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.optimize import curve_fit

from utils.tools import *

root = os.getcwd()
measurement = "Resonance Scan"
date = "8_20_26"
file_name = "1300_resonance_magnet_492_005.SPE"
laser_file_name = "laser_stats.csv"
atom = "Nd"
binder = None

N_points = 101

fix_spectrum = True
degree = 1

central_wavelength = 492.45
N_wavelengths = 51
spacing = 0.125
N_shots = 10
pixels = np.arange(0, N_wavelengths)
scan_wavelengths = central_wavelength + (pixels - (N_wavelengths - 1) // 2) * spacing

data_path = os.path.join(root, measurement)
data_path = os.path.join(data_path, date)
file_path = os.path.join(data_path, file_name)
laser_path = os.path.join(data_path, laser_file_name)

print(f"Reading File: {file_name}")
wavelength, spectrum = read_file(file_path, fix_spectrum = fix_spectrum, degree = degree)
N, M = spectrum.shape
idx = N_wavelengths // 2

spectrum = spectrum.reshape((N_wavelengths, N_shots, M))

waterfall_plot(wavelength, scan_wavelengths, spectrum.mean(axis = 1))

laser_df = pd.read_csv(laser_path)
atom_lines_df = read_reference_lines(atom, wavelength)

lamb0 = select_peak(wavelength, spectrum[idx, 0], atom_lines_df)

pixel0 = np.argmin(np.abs(wavelength - lamb0))
start = pixel0 - N_points // 2
end = pixel0 + N_points // 2 + 1

x = wavelength[start : end]
y = spectrum[idx, 0, start : end].astype(float)
point1, point2 = two_point_selection(x, y, lamb0, N_points = N_points)
plot_area(x, y, lamb0, point1, point2)

results = {}
for i in range(N_wavelengths):
    shot_areas = []
    laser_power = laser_df["pump_mj"][i]

    for j in range(N_shots):
        total_area = np.trapezoid(spectrum[i, j], wavelength)
        y = spectrum[i, j, start:end].astype(float) / total_area

        area = compute_area(x, y, point1, point2)

        shot_areas.append(area)

    shot_areas = np.asarray(shot_areas)

    mean_area = np.mean(shot_areas)
    std_area = np.std(shot_areas, ddof=1)

    if std_area > 0:
        snr = mean_area / std_area
    else:
        snr = np.nan

    results[i] = {
        "wavelength": laser_df["requested_nm"][i],
        "areas": shot_areas,
        "mean_area": mean_area,
        "std_area": std_area,
        "snr": snr
    }

wavelengths = np.array([results[i]["wavelength"] for i in results.keys()])
areas = np.array([results[i]["mean_area"] for i in results.keys()])
areas_std = np.array([results[i]["std_area"] for i in results.keys()])

window = wavelengths.max() - wavelengths.min()

idx = np.argmax(areas)
resonance_guess = wavelengths[idx]

p0 = [
    areas.max(),
    resonance_guess,
    window / 2
]

lower = [
    1e-10,
    wavelengths.min(),
    1e-3
]

upper = [
    np.inf,
    wavelengths.max(),
    np.inf
]

bounds = (lower, upper)

params, _ = curve_fit(gaussian_fit, wavelengths, areas, p0 = p0, bounds = bounds)

y_fit = gaussian_fit(wavelengths, *params)


print(f"Fit Parameters")
print(f"Amplitude: {params[0]}")
print(f"Resonant Wavelength: {params[1]} nm")

fig, ax = plt.subplots(1)

ax.errorbar(wavelengths, areas, areas_std, fmt = "o", capsize = 4, color = "black")
ax.vlines(params[1], ymin = 0, ymax = 1.1 * areas.max(), color = "green", linestyle = "--", label = f"Resonant Wavelength: {params[1]}")
ax.plot(wavelengths, y_fit, color = "red")
ax.grid(True)
ax.set_xlabel("Laser Wavelength (nm)")
ax.set_ylabel("Area (a.u.)")
ax.set_ylim(0, 1.1 * areas.max())
ax.legend()

plt.tight_layout()
plt.show()
print()
