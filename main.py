import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from spexread import read_spe_file


# =============================================================================
# Configuration
# =============================================================================

ROOT_FOLDER = os.path.join("LIBS Data", "Images")
RESULTS_FOLDER = "Results"
SPECTRUM_FOLDER = os.path.join(RESULTS_FOLDER, "Spectra")
PEAK_FIT_FOLDER = os.path.join(RESULTS_FOLDER, "Peak Fits")

KEY_WAVELENGTH = 401.25

# Number of wavelength pixels used for the peak fit.
# This must be odd and should include the entire peak plus some baseline.
N_FIT_POINTS = 21

# Skip any concentrations listed here.
SKIP_CONCENTRATIONS = {4.0}

# Set True to use generated test spectra instead of SPE files.
TEST_MODE = False

# Test-mode concentrations.
TEST_CONCENTRATIONS = np.array([
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05
])

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(SPECTRUM_FOLDER, exist_ok=True)
os.makedirs(PEAK_FIT_FOLDER, exist_ok=True)


# =============================================================================
# General utilities
# =============================================================================

def concentration_from_filename(filename):
    """
    Extract concentration from filenames such as:

        0_libs_10accum_1.SPE
        0.01_libs_10accum_1.SPE
        5_libs_10accum_2.SPE

    The concentration is assumed to be before the first underscore.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    concentration_text = stem.split("_")[0]

    try:
        return float(concentration_text)
    except ValueError as error:
        raise ValueError(
            f"Could not extract concentration from {filename!r}. "
            "The filename must begin with a numeric concentration."
        ) from error


def natural_sort_key(text):
    """
    Sort filenames naturally so that file_2 appears before file_10.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def calculate_r2(y_true, y_pred):
    """
    Calculate the coefficient of determination.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual_sum_squares = np.sum(
        (y_true - y_pred) ** 2
    )

    total_sum_squares = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    if total_sum_squares <= 0:
        return np.nan

    return 1.0 - (
        residual_sum_squares
        / total_sum_squares
    )


# =============================================================================
# SPE file reader
# =============================================================================

def read_spe(file_path, roi_name="ROI 0"):
    """
    Read a Princeton Instruments SPE file using spexread.

    All dimensions other than wavelength are averaged. This means that
    detector rows and repeated frames are averaged into one spectrum.
    """
    spe = read_spe_file(file_path)

    if roi_name not in spe:
        raise KeyError(
            f"{roi_name!r} was not found in {file_path!r}. "
            f"Available ROIs: {list(spe.data_vars)}"
        )

    roi = spe[roi_name]

    if "wavelength" not in roi.coords:
        raise ValueError(
            f"No wavelength calibration was found in {file_path!r}."
        )

    lamb = roi["wavelength"].to_numpy()

    wavelength_dims = roi["wavelength"].dims

    if len(wavelength_dims) != 1:
        raise ValueError(
            "Expected wavelength to have exactly one dimension, "
            f"but found {wavelength_dims}."
        )

    wavelength_dim = wavelength_dims[0]

    average_dims = [
        dim
        for dim in roi.dims
        if dim != wavelength_dim
    ]

    if average_dims:
        spectrum = roi.mean(dim=average_dims).to_numpy()
    else:
        spectrum = roi.to_numpy()

    lamb = np.asarray(lamb, dtype=float).squeeze()
    spectrum = np.asarray(spectrum, dtype=float).squeeze()

    if lamb.ndim != 1 or spectrum.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional arrays, but received "
            f"wavelength shape {lamb.shape} and spectrum shape "
            f"{spectrum.shape}."
        )

    if lamb.size != spectrum.size:
        raise ValueError(
            f"Wavelength and spectrum sizes do not match: "
            f"{lamb.size} and {spectrum.size}."
        )

    valid = np.isfinite(lamb) & np.isfinite(spectrum)
    lamb = lamb[valid]
    spectrum = spectrum[valid]

    if lamb.size < 3:
        raise ValueError(
            f"Not enough valid spectral points in {file_path!r}."
        )

    # Keep wavelength and spectrum ordered together.
    order = np.argsort(lamb)
    lamb = lamb[order]
    spectrum = spectrum[order]

    return lamb, spectrum


# =============================================================================
# Peak model and extraction
# =============================================================================

def gaussian_with_baseline(x, area, mu, sigma, baseline_slope, baseline_at_mu):
    """
    Gaussian peak with a linear local baseline.

    The first parameter is integrated Gaussian area rather than peak height.
    """
    gaussian = (
        area
        / (sigma * np.sqrt(2.0 * np.pi))
        * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )
    )

    baseline = (
        baseline_slope * (x - mu)
        + baseline_at_mu
    )

    return gaussian + baseline


def select_centered_window(lamb, spectrum, key_wavelength, n_points):
    """
    Select a fixed number of wavelength pixels centered near the target line.
    """
    if n_points < 7:
        raise ValueError(
            "At least 7 fitting points are recommended for the "
            "five-parameter peak model."
        )

    if n_points % 2 == 0:
        n_points += 1

    if n_points > lamb.size:
        raise ValueError(
            f"Requested {n_points} fitting points, but the spectrum "
            f"contains only {lamb.size} points."
        )

    center_idx = int(
        np.argmin(
            np.abs(lamb - key_wavelength)
        )
    )

    half_points = n_points // 2

    start = center_idx - half_points
    stop = center_idx + half_points + 1

    if start < 0:
        stop -= start
        start = 0

    if stop > lamb.size:
        start -= stop - lamb.size
        stop = lamb.size

    start = max(start, 0)

    x = lamb[start:stop]
    y = spectrum[start:stop]

    if x.size != n_points:
        raise ValueError(
            f"Expected {n_points} fitting points but selected {x.size}."
        )

    return x, y, center_idx


def baseline_corrected_area(x, y):
    """
    Calculate a baseline-corrected integrated area.

    This is used both for initial guesses and as a fallback if curve fitting
    fails.
    """
    n_edge = max(2, x.size // 4)

    edge_x = np.concatenate([
        x[:n_edge],
        x[-n_edge:]
    ])

    edge_y = np.concatenate([
        y[:n_edge],
        y[-n_edge:]
    ])

    baseline_coefficients = np.polyfit(
        edge_x,
        edge_y,
        deg=1
    )

    baseline = np.polyval(
        baseline_coefficients,
        x
    )

    corrected = y - baseline

    area = np.trapezoid(
        corrected,
        x
    )

    height = np.max(corrected)
    center = x[np.argmax(corrected)]

    return {
        "area": float(area),
        "height": float(height),
        "center": float(center),
        "baseline": baseline,
        "corrected": corrected,
        "baseline_coefficients": baseline_coefficients
    }


def extract_peak(
    lamb,
    spectrum,
    key_wavelength,
    n_points=11
):
    """
    Fit a Gaussian plus linear baseline around the target wavelength.

    Returns fitted area, height, center, width, uncertainties, and plotting
    arrays. If the nonlinear fit fails, a baseline-corrected integrated-area
    result is returned instead.
    """
    lamb = np.asarray(lamb, dtype=float).squeeze()
    spectrum = np.asarray(spectrum, dtype=float).squeeze()

    valid = np.isfinite(lamb) & np.isfinite(spectrum)
    lamb = lamb[valid]
    spectrum = spectrum[valid]

    order = np.argsort(lamb)
    lamb = lamb[order]
    spectrum = spectrum[order]

    x, y, center_idx = select_centered_window(
        lamb=lamb,
        spectrum=spectrum,
        key_wavelength=key_wavelength,
        n_points=n_points
    )

    wavelength_step = float(
        np.median(
            np.abs(
                np.diff(x)
            )
        )
    )

    if wavelength_step <= 0:
        raise ValueError(
            "The wavelength array contains repeated or invalid spacing."
        )

    simple_peak = baseline_corrected_area(x, y)

    corrected = simple_peak["corrected"]
    positive = np.clip(corrected, 0.0, None)
    positive_sum = np.sum(positive)

    if positive_sum <= 0:
        raise ValueError(
            f"No positive peak was found near "
            f"{key_wavelength:.4f} nm."
        )

    mu0 = np.sum(
        x * positive
    ) / positive_sum

    variance0 = np.sum(
        positive * (x - mu0) ** 2
    ) / positive_sum

    sigma0 = np.sqrt(
        max(
            variance0,
            wavelength_step**2
        )
    )

    area0 = max(
        np.trapezoid(
            positive,
            x
        ),
        np.finfo(float).eps
    )

    baseline_coefficients = simple_peak[
        "baseline_coefficients"
    ]

    baseline_slope0 = baseline_coefficients[0]

    baseline_at_mu0 = np.polyval(
        baseline_coefficients,
        mu0
    )

    minimum_sigma = wavelength_step / 4.0
    maximum_sigma = max(
        x.max() - x.min(),
        minimum_sigma * 2.0
    )

    lower_bounds = [
        0.0,
        x.min(),
        minimum_sigma,
        -np.inf,
        -np.inf
    ]

    upper_bounds = [
        np.inf,
        x.max(),
        maximum_sigma,
        np.inf,
        np.inf
    ]

    mu0 = np.clip(
        mu0,
        x.min() + np.finfo(float).eps,
        x.max() - np.finfo(float).eps
    )

    sigma0 = np.clip(
        sigma0,
        minimum_sigma * 1.01,
        maximum_sigma * 0.99
    )

    p0 = [
        area0,
        mu0,
        sigma0,
        baseline_slope0,
        baseline_at_mu0
    ]

    try:
        popt, pcov = curve_fit(
            gaussian_with_baseline,
            x,
            y,
            p0=p0,
            bounds=(
                lower_bounds,
                upper_bounds
            ),
            maxfev=50_000
        )

        (
            area,
            mu,
            sigma,
            baseline_slope,
            baseline_at_mu
        ) = popt

        height = area / (
            sigma * np.sqrt(2.0 * np.pi)
        )

        fwhm = (
            2.0
            * np.sqrt(2.0 * np.log(2.0))
            * sigma
        )

        fit = gaussian_with_baseline(
            x,
            *popt
        )

        parameter_variances = np.diag(pcov)
        parameter_errors = np.sqrt(
            np.clip(
                parameter_variances,
                0.0,
                None
            )
        )

        area_error = float(parameter_errors[0])
        center_error = float(parameter_errors[1])
        sigma_error = float(parameter_errors[2])

        fit_success = True
        extraction_method = "gaussian_fit"

    except (RuntimeError, ValueError, FloatingPointError) as error:
        print(
            f"  Peak fit failed: {error}. "
            "Using baseline-corrected integrated area."
        )

        area = simple_peak["area"]
        height = simple_peak["height"]
        mu = simple_peak["center"]

        sigma = np.nan
        fwhm = np.nan
        area_error = np.nan
        center_error = np.nan
        sigma_error = np.nan
        pcov = None

        fit = simple_peak["baseline"]
        popt = None

        fit_success = False
        extraction_method = "integrated_area_fallback"

    return {
        "area": float(area),
        "area_error": float(area_error),
        "height": float(height),
        "center": float(mu),
        "center_error": float(center_error),
        "sigma": float(sigma),
        "sigma_error": float(sigma_error),
        "fwhm": float(fwhm),
        "nearest_wavelength": float(lamb[center_idx]),
        "wavelength_step": wavelength_step,
        "fit_success": fit_success,
        "extraction_method": extraction_method,
        "x": x,
        "y": y,
        "fit": fit,
        "parameters": popt,
        "covariance": pcov
    }


# =============================================================================
# Calibration
# =============================================================================

def linear(x, slope, intercept):
    return slope * x + intercept


def fit_calibration(amplitudes, concentrations):
    """
    Fit concentration = slope * amplitude + intercept.
    """
    amplitudes = np.asarray(
        amplitudes,
        dtype=float
    )

    concentrations = np.asarray(
        concentrations,
        dtype=float
    )

    valid = (
        np.isfinite(amplitudes)
        & np.isfinite(concentrations)
    )

    amplitudes = amplitudes[valid]
    concentrations = concentrations[valid]

    if amplitudes.size < 2:
        raise ValueError(
            "At least two calibration points are required."
        )

    if amplitudes.size != concentrations.size:
        raise ValueError(
            "Amplitude and concentration arrays must have equal length."
        )

    if np.ptp(amplitudes) <= 0:
        raise ValueError(
            "All extracted amplitudes are identical. "
            "A calibration line cannot be fit."
        )

    slope0, intercept0 = np.polyfit(
        amplitudes,
        concentrations,
        deg=1
    )

    # Enforce a positive slope but allow a negative intercept.
    slope0 = max(
        float(slope0),
        np.finfo(float).eps
    )

    popt, pcov = curve_fit(
        linear,
        amplitudes,
        concentrations,
        p0=[
            slope0,
            float(intercept0)
        ],
        bounds=(
            [0.0, -np.inf],
            [np.inf, np.inf]
        ),
        maxfev=20_000
    )

    predicted = linear(
        amplitudes,
        *popt
    )

    r2 = calculate_r2(
        concentrations,
        predicted
    )

    residuals = concentrations - predicted

    return {
        "parameters": popt,
        "covariance": pcov,
        "predicted": predicted,
        "residuals": residuals,
        "r2": r2
    }


# =============================================================================
# Plotting
# =============================================================================

def save_full_spectrum_plot(
    lamb,
    spectrum,
    key_wavelength,
    output_path
):
    fig, ax = plt.subplots()

    ax.plot(
        lamb,
        spectrum
    )

    ax.axvline(
        key_wavelength,
        linestyle="--",
        label=f"Target: {key_wavelength:.3f} nm"
    )

    ax.set_xlabel(r"Wavelength, $\lambda$ (nm)")
    ax.set_ylabel("Counts")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)


def save_peak_fit_plot(
    peak,
    key_wavelength,
    output_path
):
    fig, ax = plt.subplots()

    ax.plot(
        peak["x"],
        peak["y"],
        "o",
        label="Measured"
    )

    ax.plot(
        peak["x"],
        peak["fit"],
        label=peak["extraction_method"]
    )

    ax.axvline(
        key_wavelength,
        linestyle="--",
        label=f"Expected: {key_wavelength:.3f} nm"
    )

    ax.axvline(
        peak["center"],
        linestyle=":",
        label=f"Extracted: {peak['center']:.3f} nm"
    )

    ax.set_xlabel(r"Wavelength, $\lambda$ (nm)")
    ax.set_ylabel("Counts")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)


def plot_calibration(
    amplitudes,
    concentrations,
    calibration,
    output_path
):
    amplitudes = np.asarray(
        amplitudes,
        dtype=float
    )

    concentrations = np.asarray(
        concentrations,
        dtype=float
    )

    slope, intercept = calibration["parameters"]
    r2 = calibration["r2"]

    x_fit = np.linspace(
        amplitudes.min(),
        amplitudes.max(),
        300
    )

    y_fit = linear(
        x_fit,
        slope,
        intercept
    )

    fig, ax = plt.subplots()

    ax.plot(
        x_fit,
        y_fit,
        label="Linear calibration"
    )

    ax.plot(
        amplitudes,
        concentrations,
        "o",
        label="Measurements"
    )

    equation_text = (
        f"Concentration = {slope:.5g} × area "
        f"{intercept:+.5g}\n"
        f"$R^2$ = {r2:.5f}"
    )

    ax.text(
        0.05,
        0.95,
        equation_text,
        transform=ax.transAxes,
        verticalalignment="top"
    )

    ax.set_xlabel(
        f"Peak area near {KEY_WAVELENGTH:.3f} nm"
    )

    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300
    )

    plt.show()


def plot_residuals(
    predicted,
    residuals,
    output_path
):
    fig, ax = plt.subplots()

    ax.plot(
        predicted,
        residuals,
        "o"
    )

    ax.axhline(
        0.0,
        linestyle="--"
    )

    ax.set_xlabel("Predicted concentration")
    ax.set_ylabel("Residual")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)


# =============================================================================
# SPE data analysis
# =============================================================================

def analyze_spe_files():
    records = []

    files = sorted(
        [
            filename
            for filename in os.listdir(ROOT_FOLDER)
            if filename.lower().endswith(".spe")
        ],
        key=natural_sort_key
    )

    if not files:
        raise FileNotFoundError(
            f"No SPE files were found in {ROOT_FOLDER!r}."
        )

    for filename in files:
        file_path = os.path.join(
            ROOT_FOLDER,
            filename
        )

        try:
            concentration = concentration_from_filename(
                filename
            )
        except ValueError as error:
            print(f"Skipping {filename}: {error}")
            continue

        if concentration in SKIP_CONCENTRATIONS:
            print(
                f"Skipping {filename}: concentration "
                f"{concentration} is excluded."
            )
            continue

        print(
            f"Analyzing: {filename}, "
            f"concentration: {concentration}"
        )

        try:
            lamb, spectrum = read_spe(
                file_path
            )

            wavelength_spacing = np.median(
                np.abs(
                    np.diff(lamb)
                )
            )

            print(
                f"  Median wavelength spacing: "
                f"{wavelength_spacing:.4f} nm"
            )

            spectrum_output_path = os.path.join(
                SPECTRUM_FOLDER,
                f"{os.path.splitext(filename)[0]}_spectrum.png"
            )

            save_full_spectrum_plot(
                lamb=lamb,
                spectrum=spectrum,
                key_wavelength=KEY_WAVELENGTH,
                output_path=spectrum_output_path
            )

            peak = extract_peak(
                lamb=lamb,
                spectrum=spectrum,
                key_wavelength=KEY_WAVELENGTH,
                n_points=N_FIT_POINTS
            )

            peak_output_path = os.path.join(
                PEAK_FIT_FOLDER,
                f"{os.path.splitext(filename)[0]}_peak_fit.png"
            )

            save_peak_fit_plot(
                peak=peak,
                key_wavelength=KEY_WAVELENGTH,
                output_path=peak_output_path
            )

            print(
                f"  Nearest wavelength: "
                f"{peak['nearest_wavelength']:.4f} nm"
            )

            print(
                f"  Extracted center: "
                f"{peak['center']:.4f} nm"
            )

            print(
                f"  Peak area: {peak['area']:.6g}"
            )

            print(
                f"  Extraction method: "
                f"{peak['extraction_method']}"
            )

            records.append({
                "file": filename,
                "concentration": concentration,
                "nearest_wavelength": peak[
                    "nearest_wavelength"
                ],
                "peak_center": peak["center"],
                "peak_center_error": peak[
                    "center_error"
                ],
                "wavelength_step": peak[
                    "wavelength_step"
                ],
                "peak_area": peak["area"],
                "peak_area_error": peak[
                    "area_error"
                ],
                "peak_height": peak["height"],
                "peak_sigma": peak["sigma"],
                "peak_sigma_error": peak[
                    "sigma_error"
                ],
                "peak_fwhm": peak["fwhm"],
                "fit_success": peak["fit_success"],
                "extraction_method": peak[
                    "extraction_method"
                ]
            })

        except (
            ValueError,
            RuntimeError,
            KeyError
        ) as error:
            print(
                f"  Failed to analyze {filename}: "
                f"{error}"
            )

    if not records:
        raise RuntimeError(
            "No SPE files were successfully analyzed."
        )

    results_df = pd.DataFrame(records)

    results_df = results_df.sort_values(
        [
            "concentration",
            "file"
        ]
    ).reset_index(drop=True)

    results_path = os.path.join(
        RESULTS_FOLDER,
        "peak_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\nIndividual measurements:")
    print(results_df.to_string(index=False))

    return results_df


# =============================================================================
# Optional summary for repeated concentrations
# =============================================================================

def summarize_replicates(results_df):
    """
    Average peak areas when multiple files have the same concentration.
    """
    summary_df = (
        results_df
        .groupby(
            "concentration",
            as_index=False
        )
        .agg(
            peak_area_mean=("peak_area", "mean"),
            peak_area_std=("peak_area", "std"),
            peak_height_mean=("peak_height", "mean"),
            peak_center_mean=("peak_center", "mean"),
            n=("peak_area", "count")
        )
    )

    summary_df["peak_area_sem"] = (
        summary_df["peak_area_std"]
        / np.sqrt(summary_df["n"])
    )

    # A single replicate has undefined standard deviation and SEM.
    summary_df["peak_area_std"] = (
        summary_df["peak_area_std"]
        .fillna(0.0)
    )

    summary_df["peak_area_sem"] = (
        summary_df["peak_area_sem"]
        .fillna(0.0)
    )

    summary_path = os.path.join(
        RESULTS_FOLDER,
        "concentration_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\nConcentration summary:")
    print(summary_df.to_string(index=False))

    return summary_df


# =============================================================================
# Test data
# =============================================================================

def generate_test_data():
    rng = np.random.default_rng(42)

    lamb = np.linspace(
        390.0,
        415.0,
        256
    )

    areas = []
    records = []

    for concentration in TEST_CONCENTRATIONS:
        true_area = 1.0e4 * concentration

        mu = KEY_WAVELENGTH
        sigma = 0.15

        baseline = (
            100.0
            + 2.0 * (lamb - KEY_WAVELENGTH)
        )

        spectrum = gaussian_with_baseline(
            lamb,
            true_area,
            mu,
            sigma,
            2.0,
            100.0
        )

        spectrum += rng.normal(
            loc=0.0,
            scale=5.0,
            size=lamb.size
        )

        peak = extract_peak(
            lamb=lamb,
            spectrum=spectrum,
            key_wavelength=KEY_WAVELENGTH,
            n_points=N_FIT_POINTS
        )

        areas.append(
            peak["area"]
        )

        records.append({
            "file": f"test_{concentration}.spe",
            "concentration": concentration,
            "peak_area": peak["area"],
            "peak_height": peak["height"],
            "peak_center": peak["center"],
            "peak_fwhm": peak["fwhm"],
            "fit_success": peak["fit_success"],
            "extraction_method": peak[
                "extraction_method"
            ]
        })

    results_df = pd.DataFrame(records)

    return results_df


# =============================================================================
# Main
# =============================================================================

def main():
    if TEST_MODE:
        results_df = generate_test_data()
    else:
        results_df = analyze_spe_files()

    summary_df = summarize_replicates(
        results_df
    )

    # Use the average area for each unique concentration.
    amplitudes = summary_df[
        "peak_area_mean"
    ].to_numpy()

    concentrations = summary_df[
        "concentration"
    ].to_numpy()

    calibration = fit_calibration(
        amplitudes=amplitudes,
        concentrations=concentrations
    )

    slope, intercept = calibration[
        "parameters"
    ]

    parameter_errors = np.sqrt(
        np.clip(
            np.diag(
                calibration["covariance"]
            ),
            0.0,
            None
        )
    )

    slope_error = parameter_errors[0]
    intercept_error = parameter_errors[1]

    print("\nCalibration results")
    print("-------------------")
    print(
        f"Slope: {slope:.8g} ± "
        f"{slope_error:.3g}"
    )

    print(
        f"Intercept: {intercept:.8g} ± "
        f"{intercept_error:.3g}"
    )

    print(
        f"R²: {calibration['r2']:.6f}"
    )

    calibration_results = pd.DataFrame({
        "concentration": concentrations,
        "peak_area": amplitudes,
        "predicted_concentration": calibration[
            "predicted"
        ],
        "residual": calibration[
            "residuals"
        ]
    })

    calibration_results.to_csv(
        os.path.join(
            RESULTS_FOLDER,
            "calibration_results.csv"
        ),
        index=False
    )

    plot_calibration(
        amplitudes=amplitudes,
        concentrations=concentrations,
        calibration=calibration,
        output_path=os.path.join(
            RESULTS_FOLDER,
            "calibration_curve.png"
        )
    )

    plot_residuals(
        predicted=calibration["predicted"],
        residuals=calibration["residuals"],
        output_path=os.path.join(
            RESULTS_FOLDER,
            "calibration_residuals.png"
        )
    )


if __name__ == "__main__":
    main()