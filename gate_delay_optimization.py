import numpy as np
import matplotlib.pyplot as plt
import os

from utils.tools import *

root = os.getcwd()
measurement = "Gate Delay"
date = "8_18_26"
file_name = "gate_0.SPE"
atom = "Nd"
binder = None
lamb0 = 492.45
N_points = 100

fix_spectrum = True
degree = 1
debug = True

data_path = os.path.join(root, measurement)
data_path = os.path.join(data_path, date)
file_path = os.path.join(data_path, file_name)

print(f"Reading File: {file_name}")
wavelength, spectrum = read_file(file_path, fix_spectrum = fix_spectrum, degree = degree)
N, M = spectrum.shape
idx = N // 2
atom_lines_df = read_reference_lines(atom, wavelength)

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(wavelength, spectrum[idx], color = "black")
    ax.vlines(atom_lines_df["wavelength"].values, ymin = spectrum[idx].min(), ymax = spectrum[idx].max(), color = "green", linestyles = "--")

    for wl in atom_lines_df["wavelength"].values:
        ax.text(wl, 1.05 * spectrum[idx].max(), str(round(wl, 2)), rotation = 90)

    ax.grid(True)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Counts (a.u.)")
    ax.set_ylim(spectrum[idx].min(), 1.2 * spectrum[idx].max())
    plt.show()

pixel0 = np.argmin(np.abs(wavelength - lamb0))
start = pixel0 - N_points // 2
end = pixel0 + N_points // 2 + 1

x = wavelength[start : end]
y = spectrum[idx, start : end]

if debug:
    fig, ax = plt.subplots(1)
    ax.plot(x, y, color = "black")
    ax.vlines(lamb0, ymin = y.min(), ymax = y.max(), linestyles = "--", color = "green")

    ax.grid(True)

    plt.show()

print()
