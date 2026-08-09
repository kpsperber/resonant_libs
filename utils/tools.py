import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from spexread import read_spe_file
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

def read_spe(file_name):
    data = read_spe_file(file_name)
    wavelengths = np.asarray(data["ROI 0"]["wavelength"].values)
    spectrum = np.asarray(data["ROI 0"].squeeze().values)

    return wavelengths, spectrum

def linear_fit(x, a, b):
    return a * x + b

def voigt_fit(x, A, mu, sigma, gamma, a, b):
    background = linear_fit(x, a, b)
    profile = A * voigt_profile(x - mu, sigma, gamma)

    return profile + background

def gaussian_fit(x, A, mu, sigma, a, b):
    background = linear_fit(x, a, b)
    profile = A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return profile + background

def lorentzian_fit(x, A, mu, gamma, a, b):
    background = linear_fit(x, a, b)
    profile = None

    return profile + background

def file_to_concentration(file_name, debug = False):
    concentration_string = file_name.split("_")[1]
    concentration = float(concentration_string) / (10 ** (len(concentration_string) - 1))

    return concentration

def read_referenece_lines(atom_name, wavelength, top_n_lines = 10):
    root = os.getcwd()
    reference_path = os.path.join(root, "Reference Lines")
    file_path = os.path.join(reference_path, atom_name + "_lines.csv")

    df = pd.read_csv(file_path)
    df.rename(columns = {"obs_wl_air(nm)" : "wavelength"}, inplace = True)
    mask = (df["wavelength"] >= wavelength.min()) & (df["wavelength"] <= wavelength.max())
    df = df[mask]

    df["intensity_numeric"] = (df["intens"].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float))
    df["peak_type"] = (df["intens"].astype(str).str.extract(r"^\d+(?:\.\d+)?(.*)$")[0])
    df.drop(columns = ["intens"], inplace = True)

    df = df.dropna(subset = ["wavelength", "intensity_numeric"])
    df = df.sort_values("wavelength").reset_index(drop = True)

    df["delta_left"] = df["wavelength"] - df["wavelength"].shift(1)
    df["delta_right"] = df["wavelength"].shift(-1) - df["wavelength"]

    df["nearest_neighbor"] = df[["delta_left", "delta_right"]].min(axis=1)


    df["distinct_score"] = (df["intensity_numeric"] * df["nearest_neighbor"])

    df = df.nlargest(top_n_lines, "distinct_score")
    df = df.sort_values("wavelength").reset_index(drop = True)

    return df

def initial_guess(x, y, mode = "gaussian"):
    mode = mode.lower()

    A0 = y.max()
    mu0 = np.sum(x * y) / np.sum(y)
    sigma0 = np.sqrt(np.sum(y * (x - mu0) ** 2) / np.sum(y))
    gamma0 = sigma0
    a0 = (y[-1] - y[0]) / (x[-1] - x[0])
    b0 = y.mean() - a0 * x.mean()

    if mode == "gaussian":
        return [A0, mu0, sigma0, a0, b0]

    elif mode == "lorentzian":
        return [A0, mu0, gamma0, a0, b0]

    elif mode == "voigt":
        return [A0, mu0, sigma0, gamma0, a0, b0]

def set_bounds(x, y, p0, mode = "gaussian"):
    mode = mode.lower()

    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    A_lower = 0.0
    A_upper = np.inf

    mu_lower = x_min
    mu_upper = x_max

    width_lower = 1e-6
    width_upper = x_range

    a_lower = -np.inf
    a_upper = np.inf

    b_lower = -np.inf
    b_upper = np.inf

    if mode == "gaussian":
        lower_bound = [
            A_lower,
            mu_lower,
            width_lower,
            a_lower,
            b_lower
        ]

        upper_bound = [
            A_upper,
            mu_upper,
            width_upper,
            a_upper,
            b_upper
        ]

    elif mode == "lorentzian":
        lower_bound = [
            A_lower,
            mu_lower,
            width_lower,
            a_lower,
            b_lower
        ]

        upper_bound = [
            A_upper,
            mu_upper,
            width_upper,
            a_upper,
            b_upper
        ]

    elif mode == "voigt":
        lower_bound = [
            A_lower,
            mu_lower,
            width_lower,
            width_lower,
            a_lower,
            b_lower
        ]

        upper_bound = [
            A_upper,
            mu_upper,
            width_upper,
            width_upper,
            a_upper,
            b_upper
        ]

    return (lower_bound, upper_bound)

def compute_area(x, params, mode = "gaussian"):
    mode = mode.lower()
    a, b = params[-2:]

    if mode == "gaussian":
        fit_func = gaussian_fit

    elif mode == "lorentzian":
        fit_func = lorentzian_fit

    elif mode == "voigt":
        fit_func = voigt_fit

    peak = fit_func(x, *params) - linear_fit(x, a, b)
    peak = np.clip(peak, 0.0, None)
    area = np.trapezoid(peak, x)

    return area
        
def compute_r2(y, y_fit):
    num = np.sum((y - y_fit) ** 2)
    denom = np.sum((y - y.mean()) ** 2)

    return 1 - num / denom
