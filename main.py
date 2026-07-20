import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from spexread import read_spe_file
import re

def concentration_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    concentration_text = stem.split("_")[0]
    return float(concentration_text)

def natural_sort_key(text):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]

def read_spe(file_path, roi_name="ROI 0"):
    spe = read_spe_file(file_path)

    if roi_name not in spe:
        raise KeyError(
            f"{roi_name!r} was not found. "
            f"Available ROIs: {list(spe.data_vars)}"
        )

    roi = spe[roi_name]

    if "wavelength" not in roi.coords:
        raise ValueError(
            "No wavelength calibration was found in the SPE file."
        )

    lamb = roi["wavelength"].to_numpy()

    # Average all dimensions except the wavelength dimension.
    wavelength_dim = roi["wavelength"].dims[0]
    average_dims = [
        dim for dim in roi.dims
        if dim != wavelength_dim
    ]

    if average_dims:
        spectrum = roi.mean(dim=average_dims).to_numpy()
    else:
        spectrum = roi.to_numpy()

    return np.asarray(lamb), np.asarray(spectrum)

def profile(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def linear(x, m, b):
    return m * x + b 

def extract_spectrum_data(lamb, spectrum, lamb_range = None):
    if lamb_range is not None:
        idx = np.where((lamb > lamb_range[0]) & (lamb < lamb_range[1]))
        lamb = lamb[idx]
        spectrum = spectrum[idx]

    A0 = np.max(spectrum)
    mu0 = np.sum(lamb * spectrum) / np.sum(spectrum)
    sigma0 = np.sqrt(np.sum(spectrum * (lamb - mu0) ** 2) / np.sum(spectrum))
    p0 = [A0, mu0, sigma0]

    popt, pcov = curve_fit(profile, lamb, spectrum, p0)
    return popt, pcov

def analysis(amplitudes, concentrations):
    m0 = (concentrations[-1] - concentrations[0]) / (amplitudes[-1] - amplitudes[0])
    b0 = 0
    p0 = [m0, b0]

    lower_bounds = [
        0.0,
        0.0
    ]

    upper_bounds = [
        np.inf,
        np.inf
    ]

    popt, pcov = curve_fit(linear, amplitudes, concentrations, p0, bounds = (lower_bounds, upper_bounds))
    return popt, pcov

root_folder = "LIBS Data\\Images"
lamb_min = 390.0
lamb_max = 415.0
key_wavelength = 430.36
lamb_range = [lamb_min, lamb_max]
concentrations = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
test = False

if not test:
    records = []

    for file in os.listdir(root_folder):
        if not file.lower().endswith(".spe"):
            continue

        file_path = os.path.join(root_folder, file)
        concentration = concentration_from_filename(file)

        print(
            f"Analyzing: {file}, "
            f"concentration: {concentration}"
        )

        lamb, spectrum = read_spe(file_path)

        lamb = np.asarray(lamb).squeeze()
        spectrum = np.asarray(spectrum).squeeze()

        idx = np.argmin(np.abs(lamb - key_wavelength))
        amplitude = float(spectrum[idx])

        records.append({
            "file": file,
            "concentration": concentration,
            "measured_wavelength": float(lamb[idx]),
            "amplitude": amplitude
        })

    results = pd.DataFrame(records)

    # Sort all measurements by concentration.
    results = results.sort_values(
        ["concentration", "file"]
    ).reset_index(drop=True)

    concentrations = results["concentration"].to_numpy()
    amps = results["amplitude"].to_numpy()

    print(results)

else:
    amps_true = 1e3 * concentrations
    mu = 402.5
    sigma = 10
    lamb = np.linspace(300, 500, 64)
    
    amps = []
    for A in amps_true:
        spectrum = profile(lamb, A, mu, sigma)
        noise = 0.05 * np.random.uniform(size = spectrum.size)
        spectrum = spectrum + noise

        params, covariance = extract_spectrum_data(lamb, spectrum, lamb_range = lamb_range)
        amps.append(params[0])

    amps = np.array(amps)
linear_params, linear_covariance = analysis(amps, concentrations)

fig, ax = plt.subplots(1)
ax.plot(amps, linear(amps, *linear_params))
ax.plot(amps, concentrations, "o")
ax.grid(True)

plt.show()

print()


