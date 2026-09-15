# Deep Introduction To sinSPICE

sinSPICE is a compact research code for synthesizing ultraviolet spectral-line
emission from 3D MURaM simulation output. It is not packaged as an installable
library yet; the current workflow is script-driven and assumes direct access to
large MURaM binary output on an HPC filesystem.

At a high level, the code does four things:

1. Read selected MURaM variables from binary snapshot files.
2. Use CHIANTI to build an emissivity contribution function for an ion line.
3. Integrate a Doppler-shifted, thermally broadened line profile through every
   vertical column in the model atmosphere.
4. Repeat that synthesis for many timesteps and save the resulting image cube.

## Repository Shape

The tracked source is intentionally small:

```text
sinSPICE/
|-- README.md
|-- docs/
|   `-- introduction.md
|-- example.py
|-- example_CIII.py
|-- examineSynthesis.ipynb
|-- modules/
|   |-- dataLoader.py
|   |-- muram.py
|   |-- synthesizeLines.py
|   `-- utils.py
`-- scripts/
    `-- example.sh
```

The important separation is:

- `modules/muram.py` knows MURaM file formats.
- `modules/dataLoader.py` adapts MURaM snapshots to the variables needed for
  synthesis.
- `modules/synthesizeLines.py` knows line physics and spectral synthesis.
- `example.py`, `example_CIII.py`, and `scripts/example.sh` orchestrate actual
  production runs.

## Execution Flow

The configurable workflow starts in `example_CIII.py`.

The script first appends `modules/` to `sys.path`, then imports:

- `parse_arguments` from `utils`
- `MURaMData` from `dataLoader`
- `spectrum` from `synthesizeLines`
- `Parallel` and `delayed` from `joblib`

It then defines run configuration:

```python
model_dir = "/glade/.../3D/"
model_label = "QS_ng_closed"
workDir = "/glade/work/mmolnar/sinSPICE/"
tStart = 0
tEnd = 39000
dTimeStep = 200
downsample = 2
```

Those values determine which MURaM files are opened, how many timesteps are
synthesized, how much spatial downsampling is applied, and where products are
saved.

The ion metadata comes from CLI arguments:

```python
ion = {
    "ion": args.ion,
    "IonLevel": args.ion_level,
    "lambda0": args.lambda0,
    "numWaves": args.numWaves,
    "dlambda": args.dLambda,
}
```

For example, this command synthesizes C III 977 A:

```bash
python example_CIII.py \
  --ion c \
  --ion_level 3 \
  --lambda0 977 \
  --numWaves 200 \
  --dLambda 0.015
```

After configuration, the script loads the first timestep to infer the spatial
shape after downsampling:

```python
muramModel = MURaMData(model_dir, tStart)
muramModel.loadModel()
nx = ny = muramModel.Temp[::downsample, ::downsample, :].shape[0]
```

Then it creates one `spectrum` object for the ion line. That object owns the
wavelength grid, the ion mass, and the CHIANTI contribution-function
interpolator. Finally, the script builds a list of timesteps and parallelizes
the synthesis over those timesteps:

```python
tSteps = [(tStart + el * dTimeStep) for el in range(nTimeSteps)]

I2D = Parallel(n_jobs=126)(
    delayed(compute_MURAM_tStep)(model_dir, el)
    for el in tSteps
)
I2D = np.asarray(I2D)
```

Each parallel worker loads one MURaM snapshot, computes one synthesized
`(nx, ny, n_wavelengths)` cube, and returns it. The final time-series array has
shape:

```text
(n_timesteps, nx, ny, n_wavelengths)
```

The script saves that array and the wavelength grid with `numpy.savez`.

## Reading MURaM Data

`modules/muram.py` is the lowest layer. It contains a set of readers that map
MURaM naming conventions into Python objects.

### File Naming

Full-domain cubes use:

```text
Header.<iteration>
result_prim_0.<iteration>
result_prim_1.<iteration>
...
eosT.<iteration>
eosne.<iteration>
```

where `<iteration>` is formatted as six digits, such as `000000` or `039000`.

The `MuramCube._varmap` dictionary maps user-facing variable names to MURaM file
prefixes. Examples:

| Attribute | File Prefix | Meaning |
| --- | --- | --- |
| `rho` | `result_prim_0` | density |
| `vx` | `result_prim_1` | velocity component |
| `vy` | `result_prim_2` | velocity component |
| `vz` | `result_prim_3` | velocity component |
| `Temp` | `eosT` | temperature |
| `Pres` | `eosP` | pressure |
| `ne` | `eosne` | electron density |
| `tau` | `tau` | optical depth |

Many additional terms are listed for heating, current, flux, and auxiliary
fields. `MuramSnap` attempts to open every mapped variable and records the
successful names in `available` and missing names in `unavailable`.

### Headers And Shapes

`MuramCube._read_header()` reads `Header.<iteration>` as `float32` text values.
It extracts:

- `shape`: grid dimensions
- `dX`: cell sizes in cm
- `X`: domain size in cm
- `time`: simulation time in seconds
- `dtime`: simulation timestep in seconds
- `vamax`: maximum Alfven velocity value from the header

The MURaM binary cube data is stored in an order that is not the desired
Python-facing order. `MuramCube.__new__()` opens the file as a `numpy.memmap`
with the reversed shape, then transposes it:

```python
np.memmap(..., shape=tuple(shape[::-1])).transpose(2, 1, 0)
```

The resulting object behaves like a NumPy array ordered as:

```text
(x, y, z)
```

This memmap approach matters because full 3D MURaM cubes are large. The code can
slice from disk without immediately loading the entire cube into memory.

### Snapshot Objects

`MuramSnap(datapath, iteration)` is a convenience container. It loops over every
known variable name and tries to create a `MuramCube`. If the file exists, it
attaches that cube to the snapshot:

```python
snap = muram.MuramSnap(model_dir, 0)
snap.Temp
snap.ne
snap.vx
snap.available
```

There is a parallel `MuramSubCube` / `MuramSubSnap` implementation for subdomain
files using `Header_Sub.<iteration>` and `subdomain_*` prefixes.

### 2D Products

The same module also supports lower-dimensional MURaM files:

- `read_hmean()` reads `hmean1D.<iteration>`.
- `read_Iout()` reads `I_out.<iteration>`.
- `read_slice()` reads `xy`, `yz`, or `tau` slice files.
- `MuramIntensity` wraps intensity slices.
- `MuramSlice`, `MuramTauSlice`, and `MuramXYSlice` wrap 2D physical slices.
- `MuramColumn` extracts one vertical column from an XY slice.

The synthesis examples do not use these 2D readers directly, but they are part
of the available codebase.

## Application Data Loader

`modules/dataLoader.py` defines `MURaMData`, a thin wrapper around `MuramSnap`.
It exists so the synthesis scripts do not need to know every detail of
`modules/muram.py`.

Usage:

```python
from dataLoader import MURaMData

model = MURaMData(model_dir, time_step)
model.loadModel()

Temp = model.Temp
ne = model.ne
vz = model.vz
```

Internally, `loadModel()` does this:

```python
self.muramModel = muram.MuramSnap(self.modelDir, timeStep)
self.Temp = self.muramModel.Temp.T
self.ne = self.muramModel.ne.T
self.vz = self.muramModel.vx.T
self.rho = self.muramModel.rho.T
self.ds = float(self.muramModel.Temp.dX[0])
```

The transpose places native vertical x last. `vz` contains native MURaM `vx`,
positive toward the top observer. `rho` is mass density in g cm^-3; `ds` is
the native vertical cell length in cm, including for unequal grid spacings.

## Spectral Synthesis Internals

The synthesis logic is in `modules/synthesizeLines.py`. There are two main
classes:

- `spectralLine`: owns one CHIANTI ion line and its `G(T, ne)` interpolation.
- `spectrum`: owns one wavelength grid and performs 1D/2D spectral synthesis.

### Ion Metadata

The examples pass dictionaries like:

```python
{
    "ion": "c",
    "IonLevel": 3,
    "lambda0": 977.0,
    "numWaves": 200,
    "dlambda": 0.015,
}
```

`spectrum.__init__()` turns that into:

- ion name for CHIANTI, such as `c_3`
- ion mass from a small hard-coded atomic-mass dictionary
- central wavelength `lambda0`
- number of wavelength samples
- wavelength spacing
- wavelength grid

The wavelength grid is:

```python
self.lambdaGrid = (
    self.lambda0
    + (self.numWavelengths // 2 - np.arange(self.numWavelengths)) * self.dlambda
)
```

Because of that formula, the grid is centered on `lambda0` but ordered from
higher wavelength to lower wavelength.

### Contribution Function Table

`spectralLine.computeGtables()` creates logarithmically spaced temperature and
electron-density arrays:

```text
T:  10^4 to 10^7.5 K, 100 samples by default
ne: 10^5 to 10^12 cm^-3, 80 samples by default
```

For each density sample, it creates a CHIANTI ion:

```python
ion = ch.ion(self.Ion, self.temp, self.ne[el])
ion.intensity()
```

It then finds the CHIANTI wavelength closest to the requested line wavelength
and stores the intensity column for that line. After looping over all densities,
the table has shape:

```text
(n_temperature_samples, n_density_samples)
```

`def_GTnfn()` wraps that table in:

```python
RectBivariateSpline(self.temp, self.ne, self.GTne_table)
```

The resulting `self.Gtne(T, ne)` interpolator is used repeatedly during
synthesis.

### One-Column Spectrum

`spectrum.compute_spectrum1D(T, ne, vz, rho=rho, ds=ds)` receives one vertical atmospheric
column. It creates a zeroed intensity vector with one value per wavelength and
then loops over height.

For each height:

1. Doppler shift:

   ```python
   dopplerShift = -vz[el] / self.c * self.lambda0  # observer="top"
   ```

2. Thermal width:

   ```python
   sigmaRadProfile = self.lambda0 / self.c * np.sqrt(
       self.kb * T[el] / self.ionMass
   )  # kb in erg/K, ionMass in grams, c in cm/s
   ```

3. Gaussian profile:

   ```python
   lineProfile = Gaussian(
       self.lambdaGrid,
       self.lambda0 + dopplerShift,
       sigmaRadProfile,
   )
   ```

4. Contribution:

   ```python
   nH = hydrogen_mass_fraction * rho[el] / (1.00794 * 1.66053906660e-24)
   Icontrib = self.spectralLine.Gtne(T[el], ne[el]) * ne[el] * nH * ds * lineProfile
   intensity += Icontrib[0, :]
   ```

The output is one optically thin spectrum per column, in
erg s^-1 cm^-2 sr^-1 Angstrom^-1 with CHIANTI's energy/Angstrom configuration.
The table uses unit emission measure, so multiplying by `ne * nH * ds` restores
the actual cell contribution. Complete cells are summed, including a one-cell
column. `hydrogen_mass_fraction` defaults to 0.74 and should match the model EOS.
`observer="bottom"` reverses the default top-observer Doppler sign.

### Full 2D Spectrum

`spectrum.compute_2Dspectrum(Temp, ne, vz, rho=rho, ds=ds, TempClip=2e4)` receives 3D cubes with
shape:

```text
(nx, ny, nz)
```

It computes one spectrum per horizontal grid point. Before integrating, it
chooses a per-column start index:

```python
indexOffset = 150
zClip = np.argmin(np.abs(Temp[:, :, indexOffset:] - TempClip), axis=-1)
zClip += indexOffset
```

This is intended to skip the lower atmosphere or convection-zone region and
begin integration near a temperature of `2e4 K`.

The function reshapes all 3D arrays into `nAtmos = nx * ny` independent columns,
computes `compute_spectrum1D()` for each column, and stores:

```text
self.I2D.shape == (nx * ny, n_wavelengths)
```

The example scripts reshape that result back to:

```text
(nx, ny, n_wavelengths)
```

## Parallelization Model

The currently active parallelism is over timesteps, not over atmospheric
columns:

```python
I2D = Parallel(n_jobs=126)(
    delayed(compute_MURAM_tStep)(model_dir, el)
    for el in tSteps
)
```

Inside each timestep, `compute_2Dspectrum()` loops over columns serially with a
Python list comprehension. There is commented-out code showing an intended or
previous column-level `joblib` parallelization path, but it is not active.

Contribution-function table generation also uses joblib, with `n_jobs=40` over
density samples.

Practical implication: large runs can consume substantial CPU and memory. Before
running on a smaller workstation, reduce:

- timestep count
- horizontal grid size via `downsample`
- wavelength count via `numWaves`
- `n_jobs`

## Output Semantics

The `.npz` products contain:

- `I2D`: synthesized intensity data
- `wvlGrid`: wavelength grid

For `example_CIII.py`, output naming is assembled from:

- `model_label`
- ion name and level, such as `c3`
- central wavelength
- timestep range
- downsampling factor

A typical analysis pattern is:

```python
import numpy as np

with np.load(path_to_npz) as data:
    I2D = data["I2D"]
    wvlGrid = data["wvlGrid"]

first_timestep = I2D[0]
integrated_image = first_timestep.sum(axis=2)
single_pixel_profile = first_timestep[100, 100, :]
```

`examineSynthesis.ipynb` is a starting point for this kind of inspection.

## How To Adapt The Code

To synthesize a different line:

1. Ensure the ion symbol exists in `mass_dict` inside `spectrum.__init__()`.
2. Choose CHIANTI-compatible ion metadata.
3. Set `lambda0` near the desired line wavelength.
4. Set `numWaves` and `dLambda` wide enough to include expected Doppler shifts.
5. Run `example_CIII.py` with new arguments or add a dedicated script.

Example:

```bash
python example_CIII.py \
  --ion o \
  --ion_level 6 \
  --lambda0 1031.9 \
  --numWaves 100 \
  --dLambda 0.015
```

To run on a different machine:

1. Create an environment with NumPy, SciPy, ChiantiPy, joblib, matplotlib,
   cmcrameri, and SymPy.
2. Ensure CHIANTI data is installed and discoverable by ChiantiPy.
3. Change `model_dir` to a MURaM output directory.
4. Change `workDir` to a writable output directory.
5. Lower `n_jobs` to match available CPUs.
6. Run a short timestep range first to verify paths and output shape.

## Current Limitations And Sharp Edges

This code is useful, but it is still research-script style rather than a
hardened package.

- There is no package metadata, dependency lockfile, or installation command.
- The examples use hard-coded HPC paths and CPU counts.
- Most run configuration is edited in source files instead of passed on the CLI.
- `MURaMData(verbose=True)` references `run1`, which is not defined.
- `MURaMData.vz` uses `muramModel.vx.T`; users must confirm this is the desired
  Doppler velocity component.
- `spectrum.__init__()` currently passes `loadGTables` and `GTablesFile`
  positionally into `spectralLine.def_GTnfn()`, whose first two parameters are
  custom table ranges. In the current examples this means the code computes
  tables by default rather than loading the provided file.
- `compute_2Dspectrum()` assumes `indexOffset = 150` is appropriate for the
  input atmosphere.
- There is no automated physical validation or regression test suite.

## Suggested Next Engineering Steps

The highest-value improvements would be:

1. Add a `requirements.txt` or `environment.yml`.
2. Move paths, timestep ranges, `downsample`, and `n_jobs` into CLI arguments.
3. Fix and test the contribution-table loading path.
4. Rename or verify the velocity component used for Doppler shifts.
5. Add a small synthetic-data test for `compute_spectrum1D()` and
   `compute_2Dspectrum()`.
6. Convert `modules/` into an importable Python package.
