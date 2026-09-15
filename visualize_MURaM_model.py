#!/usr/bin/env python3
"""Create three-panel fixed-boundary maps from MURaM model snapshots."""

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "modules"))

import muram
from utils import ProgressBar


MAGNETIC_FIELD_TO_GAUSS = np.sqrt(4.0 * np.pi)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Make three-panel maps of temperature, electron density, and total "
            "magnetic-field strength at a selected vertical boundary index."
        )
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="MURaM 3D output directory containing Header and result files.",
    )
    parser.add_argument(
        "--t-start",
        type=int,
        required=True,
        help="First MURaM timestep.",
    )
    parser.add_argument(
        "--t-end",
        type=int,
        default=None,
        help="Last MURaM timestep, inclusive. Defaults to --t-start.",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=1,
        help="MURaM timestep spacing.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Horizontal downsampling factor used during synthesis.",
    )
    parser.add_argument(
        "--boundary-index",
        type=int,
        default=150,
        help="Vertical model index at which properties are shown. Defaults to 150.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("MURaM_model_frames"),
        help="Directory for PNG frames. Defaults to MURaM_model_frames.",
    )
    parser.add_argument(
        "--prefix",
        default="MURaM_model_3panel",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_false",
        dest="transpose",
        help="Do not transpose maps before imshow.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG output resolution.")
    parser.add_argument(
        "--cmap-temperature",
        default="inferno",
        help="Matplotlib colormap for temperature.",
    )
    parser.add_argument(
        "--cmap-density",
        default="viridis",
        help="Matplotlib colormap for electron density.",
    )
    parser.add_argument(
        "--cmap-magnetic",
        default="magma",
        help="Matplotlib colormap for magnetic-field strength.",
    )

    args = parser.parse_args()
    if args.t_end is None:
        args.t_end = args.t_start
    if args.t_end < args.t_start:
        parser.error("--t-end must be greater than or equal to --t-start")
    if args.time_step <= 0:
        parser.error("--time-step must be positive")
    if args.downsample <= 0:
        parser.error("--downsample must be positive")
    if args.boundary_index < 0:
        parser.error("--boundary-index must be nonnegative")
    return args


def sample_at_boundary(cube, boundary_index):
    """Sample a (nx, ny, nz) cube at one fixed vertical index."""
    if cube.ndim != 3:
        raise ValueError("Model cube must have shape (nx, ny, nz).")
    if boundary_index >= cube.shape[-1]:
        raise ValueError(
            f"boundary index {boundary_index} is outside a cube with "
            f"{cube.shape[-1]} vertical points"
        )
    return np.asarray(cube[..., boundary_index])


def load_boundary_maps(model_dir, timestep, downsample, boundary_index):
    model = muram.MuramSnap(str(model_dir), timestep)
    missing = [
        name
        for name in ("Temp", "ne", "Bx", "By", "Bz")
        if not hasattr(model, name)
    ]
    if missing:
        missing_files = ", ".join(missing)
        raise FileNotFoundError(
            f"MURaM timestep {timestep} is missing required variables: "
            f"{missing_files}"
        )

    temperature = model.Temp.T[::downsample, ::downsample, :]
    electron_density = model.ne.T[::downsample, ::downsample, :]

    temperature_map = sample_at_boundary(temperature, boundary_index)
    density_map = sample_at_boundary(electron_density, boundary_index)

    bx = model.Bx.T[::downsample, ::downsample, :]
    by = model.By.T[::downsample, ::downsample, :]
    bz = model.Bz.T[::downsample, ::downsample, :]
    bx_map = sample_at_boundary(bx, boundary_index).astype(np.float64)
    by_map = sample_at_boundary(by, boundary_index).astype(np.float64)
    bz_map = sample_at_boundary(bz, boundary_index).astype(np.float64)
    magnetic_map = (
        np.sqrt(bx_map**2 + by_map**2 + bz_map**2)
        * MAGNETIC_FIELD_TO_GAUSS
    )

    return (
        np.asarray(temperature_map, dtype=np.float32),
        np.asarray(density_map, dtype=np.float32),
        np.asarray(magnetic_map, dtype=np.float32),
    )


def percentile_limits(frames, map_index, low=1.0, high=99.5):
    arrays = [
        frame[map_index].ravel()
        for frame in frames
        if np.any(np.isfinite(frame[map_index]))
    ]
    if not arrays:
        return 0.0, 1.0

    values = np.concatenate(arrays)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.nanpercentile(finite, [low, high])
    if vmin == vmax:
        delta = abs(vmin) * 0.05 if vmin else 1.0
        vmin -= delta
        vmax += delta
    return float(vmin), float(vmax)


def map_for_plot(values, transpose):
    return values.T if transpose else values


def save_frame(
    output_file,
    frame_index,
    timestep,
    temperature,
    electron_density,
    magnetic_field,
    limits,
    args,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    panels = [
        (
            axes[0],
            temperature,
            f"Temperature at boundary index {args.boundary_index}",
            args.cmap_temperature,
            limits[0],
            "K",
        ),
        (
            axes[1],
            electron_density,
            f"Electron density at boundary index {args.boundary_index}",
            args.cmap_density,
            limits[1],
            r"cm$^{-3}$",
        ),
        (
            axes[2],
            magnetic_field,
            f"Total magnetic field at boundary index {args.boundary_index}",
            args.cmap_magnetic,
            limits[2],
            "G",
        ),
    ]

    for axis, values, title, cmap, (vmin, vmax), colorbar_label in panels:
        image = axis.imshow(
            map_for_plot(values, args.transpose),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, label=colorbar_label)

    fig.suptitle(f"MURaM model | frame {frame_index} | timestep {timestep}")
    fig.savefig(output_file, dpi=args.dpi)
    plt.close(fig)


def main():
    args = parse_arguments()
    timesteps = list(range(args.t_start, args.t_end + 1, args.time_step))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    with ProgressBar(len(timesteps), label="Loading MURaM boundary maps") as progress:
        for timestep in timesteps:
            maps = load_boundary_maps(
                args.model_dir,
                timestep,
                args.downsample,
                args.boundary_index,
            )
            frames.append(maps)
            progress.update()

    limits = tuple(percentile_limits(frames, index) for index in range(3))

    print(f"Input: {args.model_dir}")
    print(f"Timesteps: {timesteps[0]} to {timesteps[-1]} by {args.time_step}")
    print(f"Boundary index: {args.boundary_index}")
    print(f"Writing frames to: {args.output_dir}")

    with ProgressBar(len(timesteps), label="Writing MURaM frames") as progress:
        for frame_index, (timestep, maps) in enumerate(zip(timesteps, frames)):
            output_file = (
                args.output_dir / f"{args.prefix}_{timestep:06d}.png"
            )
            save_frame(
                output_file,
                frame_index,
                timestep,
                *maps,
                limits,
                args,
            )
            progress.update()

    print("Done.")


if __name__ == "__main__":
    main()
