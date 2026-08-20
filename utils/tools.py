import numpy as np
import os
import pandas as pd
from spexread import read_spe_file

def read_file(file_name, fix_spectrum = False, degree = 5):
    data = read_spe_file(file_name)

    spectrum = data["ROI 0"].values
    spectrum = spectrum.squeeze()

    wavelength = data["wavelength"].values

    if fix_spectrum:
        N_fit = 10
        N = len(wavelength)
        pixels = np.arange(N)

        pixels_fit = pixels[:N_fit]
        wavelength_fit = wavelength[:N_fit]

        coeffs = np.polyfit(pixels_fit, wavelength_fit, degree)
        wavelength = np.polyval(coeffs, pixels)

    return wavelength, spectrum

def read_reference_lines(atom_name, wavelength, top_n_peaks = 5):
    root = os.getcwd()
    reference_folder = "Reference Lines"
    root = os.path.join(root, reference_folder)

    file_name = atom_name + "_lines.csv"
    file_path = os.path.join(root, file_name)

    df = pd.read_csv(file_path)
    df = df.rename(columns = {"obs_wl_air(nm)" : "wavelength"})
    df = df.drop(columns = ["ritz_wl_air(nm)", "element"])

    wavelength_mask = (df["wavelength"].values >= wavelength.min()) & (df["wavelength"].values <= wavelength.max())
    df = df[wavelength_mask]

    species_mask = (df["sp_num"] == 1)
    df = df[species_mask]

    df = df[~df["intens"].astype(str).str.contains(r"[hw]", case=False, na=False)].copy()
    df["intensity"] = (df["intens"].astype(str).str.extract(r"(\d+\.?\d*)"))[0].astype(float)

    wls = df["wavelength"].values

    avg_spacing = []

    for wl in wls:
        distances = np.abs(wls - wl)
        distances = distances[distances > 0]

        avg_spacing.append(np.mean(distances))

    df["avg_spacing"] = avg_spacing

    df["intensity_score"] = df["intensity"] / df["intensity"].max()
    df["spacing_score"] = df["avg_spacing"] / df["avg_spacing"].max()

    df["score"] = df["intensity_score"] * df["spacing_score"]

    df = df.nlargest(top_n_peaks, "score")

    return df

def read_binder_lines(atom_names):
    root = os.getcwd()

    for atom in atom_names:
        file_name = atom + "_lines.csv"
        file_path = os.path.join(root, file_name)
        
    return