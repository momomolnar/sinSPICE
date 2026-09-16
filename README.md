# sinSPICE

sinSPICE synthesizes ultraviolet transition-region spectral-line emission from
3D MURaM magnetohydrodynamic simulation snapshots. The project reads MURaM
binary cubes, builds CHIANTI contribution-function tables, integrates line
emission through each vertical atmosphere column, and writes wavelength-resolved
2D intensity maps over a time series.

For a deeper walkthrough of the internals, see
[docs/introduction.md](docs/introduction.md).

To calculate SPICE line optical depths from MURaM using ChiantiPy, use
`compute_spice_optical_depth.py`. The [optical-depth guide](docs/spice_optical_depth.md)
documents the instrument-paper line selection, equations, assumptions, command
examples, and output format. Run `python compute_spice_optical_depth.py --list-lines`
to see the supported lines.
Add `--plot` to save PNG maps of peak column optical depth with horizontal x/y
coordinates in Mm. Plot saved results with
`python visualize_spice_optical_depth.py optical_depth_output/spice_opacity_*.npz`.

## What This Repository Contains

| Path | Purpose |
| --- | --- |
| `example.py` | Hard-coded Ne VIII 770 A synthesis example for a MURaM time series. |
| `example_CIII.py` | Command-line driven synthesis example for C III, Mg IX, O VI, or another supported ion configuration. |
| `modules/muram.py` | Low-level MURaM file readers for full cubes, subdomain cubes, slices, intensities, and snapshots. |
| `modules/dataLoader.py` | Small wrapper that loads the MURaM variables used by the synthesis pipeline. |
| `modules/synthesizeLines.py` | CHIANTI-backed line contribution tables and spectral synthesis routines. |
| `modules/utils.py` | CLI argument parser and Gaussian profile helper. |
| `visualize_CIII_results.py` | Three-panel maps of integrated intensity, Doppler velocity, and wavelength line width from synthesis output. |
| `visualize_MURaM_model.py` | Three-panel maps of temperature, electron density, and total magnetic field at the spectral-integration boundary. |
| `scripts/example.sh` | PBS batch script configured for the NCAR/GLADE environment. |
| `examineSynthesis.ipynb` | Lightweight notebook for loading and inspecting saved synthesis output. |

## Core Workflow

### Emission normalization and observer convention

The original emission pipeline now uses CGS ion masses and weights every cell
by its emission measure `ne * nH * ds`, where
`nH = hydrogen_mass_fraction * rho / (1.00794 * atomic_mass_unit)`.
`MURaMData` exposes `rho` in g cm^-3 and `ds = dX[0]` in cm. Both example
drivers pass them into synthesis. The hydrogen mass fraction defaults to 0.74;
set `hydrogen_mass_fraction` in the driver to match the model EOS composition.

Calls to `compute_spectrum1D` and `compute_2Dspectrum` require keyword arguments
`rho` and `ds`. Both accept `observer="top"` (default) or `"bottom"`.
For the top observer, positive native MURaM `vx` gives a blueshift. The `vz`
argument retains its historical name but contains this native vertical velocity.
Synthesis output is intensity per Angstrom (erg s^-1 cm^-2 sr^-1 Angstrom^-1),
assuming CHIANTI's energy/Angstrom configuration. The wavelength grid remains
descending; reverse both arrays before integrating over wavelength.

These corrections change widths, intensities, and Doppler signs. Regenerate
previous emission products before comparing them to corrected results. The
review in `docs/sinSPICE_overview.tex` and its PDF describe an older code state.

1. A MURaM output directory is selected through `model_dir` or `modeldir`.
2. `MURaMData` opens a timestep with `muram.MuramSnap`.
3. The data loader exposes `Temp`, `ne`, `vz`, `rho`, and vertical cell length `ds`.
4. A `spectrum` object is configured with an ion, ionization level, central
   wavelength, wavelength count, and wavelength spacing.
5. `spectralLine` uses CHIANTI to compute a contribution function table
   `G(T, ne)` and wraps it in a SciPy `RectBivariateSpline`.
6. `compute_2Dspectrum()` clips each atmospheric column below a temperature
   threshold, computes a thermally broadened and Doppler-shifted Gaussian line
   profile at every retained height, and sums the contribution along the line of
   sight.
7. The example scripts parallelize over MURaM timesteps with `joblib.Parallel`.
8. Results are written as `.npz` files containing `I2D` and `wvlGrid`.

The time-series output shape produced by the examples is:

```text
(n_timesteps, nx, ny, n_wavelengths)
```

where `nx` and `ny` include any configured downsampling, and `n_wavelengths`
comes from `numWaves`.

## Dependencies

This repository does not currently include a `requirements.txt`, `pyproject.toml`,
or conda environment file. The imports in the tracked source imply these runtime
dependencies:

- Python 3
- NumPy
- SciPy
- CHIANTI / ChiantiPy
- joblib
- matplotlib
- cmcrameri
- SymPy
- Jupyter, only for `examineSynthesis.ipynb`

The PBS example assumes a conda environment named `chianti`:

```bash
module load conda
conda activate chianti
```

## Input Data Requirements

The synthesis examples expect MURaM output files using the naming conventions in
`modules/muram.py`.

Full-domain cube files include:

- `Header.000000`, `Header.000100`, etc.
- `eosT.<iteration>` for temperature
- `eosne.<iteration>` for electron density, when available
- `result_prim_1.<iteration>`, currently loaded by `MURaMData` into `vz`
- `result_prim_0.<iteration>` for mass density in g cm^-3

The low-level reader also knows many other MURaM variables, including density,
velocity components, magnetic fields, pressure, heating terms, currents,
optical depth, and selected auxiliary fields. Missing files are skipped by
`MuramSnap` and recorded in the `unavailable` list.

If `eosne.<iteration>` is missing but `result_prim_0.<iteration>` is present,
`MuramSnap` supplies a lazy electron-density fallback using
`ne = 1.2 * rho / (1.4271 * atomic_mass)`. This is the fully ionized,
solar-composition approximation appropriate to hot transition-region and coronal
plasma. The snapshot records `ne` in `derived` and sets
`ne_source = "rho_fully_ionized"`; pass `derive_ne=False` to `MuramSnap` to
require an electron-density file.

If both `eosne` and `result_prim_0` are absent but `eosP` and `eosT` exist, the
same fallback first applies the ideal-gas relation with 2.3 total particles per
hydrogen nucleus:

```text
nH  = P / (2.3 k_B T)
rho = 1.4271 atomic_mass nH
ne  = 1.2 nH
```

Here pressure is in dyn cm^-2, temperature is in K, number densities are in
cm^-3, and mass density is in g cm^-3.

## Running The Existing Examples

The examples currently contain hard-coded NCAR/GLADE paths. Before running them
elsewhere, edit:

- `model_dir` or `modeldir`: directory containing MURaM output files
- `workDir`: output directory for `.npz` products and figures
- `tStart`, `tEnd`, `dTimeStep`: timestep range
- `downsample`: spatial downsampling factor
- `Parallel(n_jobs=...)`: CPU count appropriate for the machine
- `GTablesFile`: contribution-table path, if loading saved tables is enabled

Run the configurable example directly:

```bash
python example_CIII.py \
  --ion c \
  --ion_level 3 \
  --lambda0 977 \
  --numWaves 200 \
  --dLambda 0.015
```

Run on a PBS system with the provided batch file:

```bash
qsub scripts/example.sh
```

The batch script currently runs:

```bash
python -i example_CIII.py --ion "c" --ion_level 3 --lambda0 977 --numWaves 200 --dLambda 0.015
```

Alternative Mg IX and O VI invocations are included as comments in
`scripts/example.sh`.

To visualize synthesized spectra with logarithmic integrated intensity:

```bash
python visualize_CIII_results.py synthesis_output.npz \
  --intensity-scale log \
  --velocity-range -30 30
```

Logarithmic intensity scaling is the default. Use `--intensity-scale linear`
to recover linear color normalization.

To visualize the lower boundary used for spectral integration:

```bash
python visualize_MURaM_model.py /path/to/MURaM/3D \
  --t-start 400000 \
  --t-end 502000 \
  --time-step 2000 \
  --downsample 2 \
  --boundary-index 150
```

The boundary is a fixed vertical model index selected with `--boundary-index`;
it defaults to 150. Each frame shows temperature, electron density, and total
magnetic-field strength at that boundary.

## Main Concepts

### MURaM Readers

`modules/muram.py` contains NumPy-backed readers for MURaM data:

- `MuramCube` maps a full 3D cube from disk with `numpy.memmap`.
- `MuramSnap` builds a snapshot object by trying every known cube variable.
- `MuramSubCube` and `MuramSubSnap` do the same for subdomain files.
- `MuramSlice`, `MuramTauSlice`, `MuramXYSlice`, `MuramColumn`, and
  `MuramIntensity` read lower-dimensional MURaM products.

The binary cube reader reverses the file shape and transposes the memmap so the
Python-facing cube is ordered as `(x, y, z)`.

### Data Loader

`modules/dataLoader.py` defines `MURaMData`, the application-level loader used by
the synthesis examples. After `loadModel()`, it exposes:

- `Temp`: transposed MURaM temperature cube
- `ne`: transposed electron density cube
- `vz`: transposed velocity cube used for Doppler shifts

Current implementation note: `vz` is assigned from `muramModel.vx.T`. Verify
that this is the intended line-of-sight velocity component for your simulation
coordinate convention.

### Contribution Functions

`spectralLine` in `modules/synthesizeLines.py` encapsulates one CHIANTI line.
It can compute a table over temperature and electron density, then create a
`RectBivariateSpline` interpolator for `G(T, ne)`.

Default table ranges are:

- `log10(T)`: 4 to 7.5 with 100 samples
- `log10(ne)`: 5 to 12 with 80 samples

When generating a table, the code parallelizes over density samples and writes a
`G_Tmin_max_num_...ne_...npz` file in the current working directory.

### Spectrum Synthesis

`spectrum` combines the ion metadata, wavelength grid, contribution-function
interpolator, and emission integration logic.

For each vertical grid cell retained in a column:

1. It computes a Doppler shift from velocity and `lambda0`.
2. It computes thermal line width from ion mass and temperature.
3. It evaluates a normalized Gaussian profile on `lambdaGrid`.
4. It multiplies the profile by `G(T, ne) * ne * nH * ds`.
5. It sums all height contributions into one emergent spectrum.

`compute_2Dspectrum(Temp, ne, vz, rho=rho, ds=ds, TempClip=2e4)` applies that 1D synthesis to
all `(x, y)` columns. It skips lower atmospheric layers by finding, for each
column, the height closest to `TempClip` after `indexOffset = 150`.

## Output Products

The examples save output with `numpy.savez`. A typical file contains:

- `I2D`: synthesized intensity cube
- `wvlGrid`: wavelength grid used during synthesis

Example loading pattern:

```python
import numpy as np

with np.load("QS_ng_closed_I2D_c3_977AA_t_0_39000_dwnsmpl_2.npz") as data:
    I2D = data["I2D"]
    wvlGrid = data["wvlGrid"]
```

To create an integrated intensity image for one timestep:

```python
image = I2D[0].sum(axis=2)
```

## Known Implementation Caveats

- Paths, timestep ranges, output names, and CPU counts are hard-coded in the
  example scripts.
- `example_CIII.py` is configurable for ion metadata, but not yet for model
  paths, time ranges, downsampling, or output location.
- `MURaMData(verbose=True)` references an undefined variable named `run1`.
- `MURaMData.vz` currently uses `muramModel.vx.T`; confirm velocity-component
  conventions before interpreting Doppler shifts.
- `spectrum.__init__()` calls `spectralLine.def_GTnfn()` positionally. As
  currently written, the default examples compute contribution tables rather
  than loading the provided `GTablesFile`.
- The synthesis loop inside one 2D timestep is serial; only the outer timestep
  loop is parallelized in the examples.
- Generated data and local output directories are not tracked as source files.

## Development Checks

There is no formal test suite in this repository. A basic syntax check is:

```bash
find . -maxdepth 2 -type f -name '*.py' -print | sort | xargs -n1 python -m py_compile
```

This validates Python syntax only; it does not verify CHIANTI data availability,
MURaM file availability, physical correctness, or performance.
