#!/usr/bin/env python3
"""Write ffmpeg-ready three-panel frames for the AR C III synthesis."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import numpy as np


INPUT_FILE = Path(
    "/glade/work/mmolnar/sinSPICE/"
    "AR_11158_10kG_I2D_c3_977.0AA_t_400000_502000_dwnsmpl_2.npz"
)
OUTPUT_DIR = Path("AR_CIII_frames")
REST_WAVELENGTH = 977.0       # Angstrom
TIME_START = 400000
TIME_STEP = 2000
C_KMS = 299792.458


def integrate(values, wavelength):
    function = getattr(np, "trapezoid", np.trapz)
    return function(values, x=wavelength, axis=-1)


def moments(spectra, wavelength):
    """Return integrated intensity, COG velocity, and 1-sigma width in km/s."""
    intensity = integrate(spectra, wavelength)
    denominator = np.where(intensity > 0, intensity, np.nan)
    centroid = integrate(spectra * wavelength, wavelength) / denominator
    variance = integrate(
        spectra * (wavelength - centroid[..., None]) ** 2, wavelength
    ) / denominator
    # Positive velocity means motion toward the observer: blueshifts are positive.
    velocity = -C_KMS * (centroid / REST_WAVELENGTH - 1)
    width = C_KMS * np.sqrt(np.maximum(variance, 0)) / REST_WAVELENGTH
    return intensity, velocity, width


def limits(values, low=1, high=99.5, positive=False):
    values = values[np.isfinite(values)]
    if positive:
        values = values[values > 0]
    if not values.size:
        raise ValueError("No finite values available for plotting.")
    lower, upper = np.percentile(values, [low, high])
    if lower == upper:
        padding = abs(lower) * 0.05 or 1.0
        lower, upper = lower - padding, upper + padding
    return float(lower), float(upper)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT_FILE, allow_pickle=False) as data:
        spectra = data["I2D"]
        wavelength = data["wvlGrid"]

    if spectra.ndim != 4 or spectra.shape[-1] != wavelength.size:
        raise ValueError("Expected I2D shape (time, x, y, wavelength).")

    # The original synthesis grid is descending; integration needs increasing x.
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    spectra = spectra[..., order]

    # Use fixed movie-wide limits so colors do not flicker between frames.
    intensity_bounds, width_bounds = [], []
    for frame in spectra:
        intensity, _, width = moments(frame, wavelength)
        intensity_bounds.append(limits(intensity, positive=True))
        width_bounds.append(limits(width))
    intensity_limits = (min(x[0] for x in intensity_bounds),
                        max(x[1] for x in intensity_bounds))
    width_limits = (min(x[0] for x in width_bounds),
                    max(x[1] for x in width_bounds))

    for index, frame in enumerate(spectra):
        intensity, velocity, width = moments(frame, wavelength)
        panels = (
            (intensity, "Total C III intensity", "inferno",
             LogNorm(*intensity_limits), r"erg s$^{-1}$ cm$^{-2}$ sr$^{-1}$"),
            # seismic_r maps positive values to blue and negative values to red.
            (velocity, "Doppler velocity (+ = blueshift)", "seismic_r",
             Normalize(-50, 50), "km/s"),
            (width, r"Doppler width $\sigma_v$", "magma",
             Normalize(*width_limits), "km/s"),
        )
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                                    constrained_layout=True)
        for axis, (values, title, cmap, norm, unit) in zip(axes, panels):
            image = axis.imshow(values.T, origin="lower", cmap=cmap, norm=norm)
            axis.set(title=title, xlabel="x pixel", ylabel="y pixel")
            figure.colorbar(image, ax=axis, label=unit)
        timestep = TIME_START + index * TIME_STEP
        figure.suptitle(f"C III 977 Å — MURaM iteration {timestep:06d}")
        output = OUTPUT_DIR / f"AR_CIII_3panel_{index:05d}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        print(f"[{index + 1}/{len(spectra)}] {output}", flush=True)

    print("\nMovie command:")
    print("ffmpeg -framerate 20 -i AR_CIII_frames/AR_CIII_3panel_%05d.png "
          "-c:v libx264 -pix_fmt yuv420p AR_CIII.mp4")


if __name__ == "__main__":
    main()
