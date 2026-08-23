import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
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

def two_point_selection(x, y, lamb0, N_points = 100):

    fig, ax = plt.subplots(1)
    ax.plot(x, y, color = "black")
    ax.vlines(lamb0, ymin=y.min(), ymax=y.max(), linestyles="--", color="green")

    ax.grid(True)
    points = plt.ginput(2, timeout=-1)

    plt.close(fig)

    (x1, _), (x2, _) = points

    point1 = np.argmin(np.abs(x1 - x))
    point2 = np.argmin(np.abs(x2 - x))

    point1, point2 = sorted([point1, point2])

    return point1, point2

def plot_area(x, y, lamb0, point1, point2, time_plot = 2):
    x_peak = x[point1 : point2 + 1]
    y_peak = y[point1 : point2 + 1]

    x1 = x[point1]
    x2 = x[point2]
    y1 = y[point1]
    y2 = y[point2]

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    baseline = slope * x_peak + intercept

    fig, ax = plt.subplots(1)
    ax.plot(x, y, color = "black")
    ax.plot(x_peak, baseline, color = "red")
    ax.fill_between(x_peak, baseline, y_peak, alpha = 0.3)
    ax.vlines(lamb0, ymin=y.min(), ymax=y.max(), linestyles="--", color="green")

    ax.grid(True)

    plt.show(block = False)
    plt.pause(time_plot)
    plt.close()


def select_peak(wavelength, spectrum, atom_lines_df):
    fig, ax = plt.subplots(1)

    ax.plot(wavelength, spectrum, color = "black")

    ax.vlines(
        atom_lines_df["wavelength"].values,
        ymin=spectrum.min(),
        ymax=spectrum.max(),
        color="green",
        linestyles="--"
    )

    for wl in atom_lines_df["wavelength"].values:
        text = ax.text(
            wl,
            1.05 * spectrum.max(),
            f"{wl:.2f}",
            rotation=90,
            ha="center",
            picker=True
        )

        text.wavelength = wl

    ax.grid(True)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Counts (a.u.)")
    ax.set_ylim(
        spectrum.min(),
        1.2 * spectrum.max()
    )

    selected = {"lamb0": None}

    def on_pick(event):
        artist = event.artist

        if hasattr(artist, "wavelength"):
            selected["lamb0"] = artist.wavelength

            print(
                f"Selected wavelength: "
                f"{selected['lamb0']:.3f} nm"
            )

            plt.close(fig)

    fig.canvas.mpl_connect("pick_event", on_pick)

    plt.show()
    plt.close()

    lamb0 = selected["lamb0"]

    return lamb0

def compute_snr(x, y, point1, point2):

    # Baseline endpoints
    x1 = x[point1]
    x2 = x[point2]

    y1 = y[point1]
    y2 = y[point2]

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    baseline = slope * x + intercept

    # Baseline-subtracted spectrum
    y_corrected = y - baseline

    # Signal = integrated peak area
    x_peak = x[point1 : point2 + 1]
    y_peak = y_corrected[point1 : point2 + 1]

    signal = np.trapezoid(y_peak, x_peak)

    # Noise estimated from regions outside the peak
    noise_region = np.concatenate([
        y_corrected[:point1],
        y_corrected[point2 + 1:]
    ])

    noise = np.std(noise_region, ddof=1)

    if noise > 0:
        snr = signal / noise
    else:
        snr = np.nan

    return snr

def compute_area(x, y, point1, point2):
    x_peak = x[point1 : point2]
    y_peak = y[point1 : point2]
    x1 = x[point1]
    x2 = x[point2]

    y1 = y[point1]
    y2 = y[point2]

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    baseline = slope * x_peak + intercept
    area = np.trapezoid(y_peak - baseline, x_peak)

    return area

def gaussian_fit(x, A, mu, sigma):
    u = (x - mu) ** 2 / (2 * sigma ** 2)
    return A * np.exp(-u)

def waterfall_plot(emission_wavelength, laser_wavelength, spectrum):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=25, azim=-30)

    M = len(laser_wavelength)
    N = len(emission_wavelength)

    for i in range(M):
        ax.plot(
            emission_wavelength,
            np.full_like(emission_wavelength, laser_wavelength[i]),
            spectrum[i]
        )

    ax.set_xlabel("Emission Wavelength")
    ax.set_ylabel("Laser Wavelength")
    ax.set_zlabel("Intensity")

    plt.tight_layout()
    plt.show()