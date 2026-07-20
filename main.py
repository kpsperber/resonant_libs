import imageio as iio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

def read_spe(file_name):
    lamb = None
    spectrum = None

    return lamb, spectrum

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

    popt, pcov = curve_fit(linear, amplitudes, concentrations, b0, bounds = (lower_bounds, upper_bounds))
    return popt, pcov

root_folder = "LIBS Data"
lamb_min = 390.0
lamb_max = 415.0
lamb_range = [lamb_min, lamb_max]
concentrations = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
test = True

if not test:
    amp = []
    for file in os.listdir(root_folder):
        print(f"Analyzing: {file}")
        file_path = os.path.join(root_folder, file)

        lamb, spectrum = read_spe(file_path)
        extract_spectrum_data(lamb, spectrum, lamb_range = lamb_range)

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

fig, ax = plt.subplots(1, 2)
ax[0].plot(lamb, profile(lamb, *params))
ax[0].plot(lamb, spectrum, "o")
ax[0].grid(True)

plt.show()

print()


