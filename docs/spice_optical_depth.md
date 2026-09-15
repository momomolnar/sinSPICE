# SPICE line optical depths from MURaM

`compute_spice_optical_depth.py` calculates the line optical depth through each
vertical MURaM column, including thermal broadening, optional microturbulence,
the resolved velocity, and stimulated emission. It also synthesizes an optically
thin comparison spectrum from the same atomic populations. The reusable physics
and population-table code is in `modules/spiceOpticalDepth.py`.

This is an **equilibrium, bound-bound opacity diagnostic**. CHIANTI populations
are not recomputed in response to the radiation trapped by an optically thick
line. Consequently, a large optical depth flags a failure of the optically thin
assumption; the saved comparison intensity is not a prediction of the emergent
intensity in that regime.

## Line selection and the instrument paper

The catalogue follows Table 1 of **SPICE Consortium et al. (2020), “The Solar
Orbiter SPICE instrument: An extreme UV imaging spectrometer,” A&A 642, A14**,
[DOI](https://doi.org/10.1051/0004-6361/201935574),
[accessible PDF](https://www2.mps.mpg.de/homes/solanki/publikationen/j532.pdf).
The paper specifies bands of 704–790 and 973–1049 Å. Its table is a selection
of diagnostics, not every transition in those bands. Si XII is a second-order
line: the absorbing photon has wavelength 520.67 Å, while its nominal
first-order-equivalent detector coordinate is 1041.34 Å. All opacity and
emissivity calculations use the physical photon wavelength.

| CLI identifier | Ion | Paper wavelength (Å) | Paper log10 T (K) | Default |
| --- | --- | ---: | ---: | --- |
| `h_1_1025` | H I | 1025.72 | 4.0 | |
| `c_2_1036` | C II | 1036.34 | 4.3 | |
| `c_3_977` | C III | 977.03 | 4.5 | yes |
| `s_5_786` | S V | 786.47 | 5.2 | yes |
| `o_4_787` | O IV | 787.72 | 5.2 | yes |
| `o_5_760` | O V | 760.43 | 5.4 | yes |
| `o_6_1031` | O VI | 1031.93 | 5.5 | yes |
| `o_6_1037` | O VI | 1037.64 | 5.5 | yes |
| `ne_6_1005` | Ne VI | 1005.79 | 5.6 | |
| `si_7_1049` | Si VII | 1049.25 | 5.6 | |
| `ne_8_770` | Ne VIII | 770.42 | 5.8 | yes |
| `mg_8_772` | Mg VIII | 772.31 | 5.9 | |
| `mg_9_706` | Mg IX | 706.02 | 6.0 | yes |
| `fe_10_1028` | Fe X | 1028.04 | 6.0 | |
| `mg_11_997` | Mg XI | 997.44 | 6.2 | |
| `si_12_520` | Si XII, order 2 | 520.67 | 6.3 | |
| `fe_18_974` | Fe XVIII | 974.84 | 6.9 | |
| `fe_20_721` | Fe XX | 721.55 | 7.0 | |

The eight defaults are my practical starting set for this repository's
transition-region/coronal models. They include C III, the O VI doublet, and
Ne VIII, plus intermediate-temperature diagnostics and Mg IX. This default set
is a software choice. The table's approximate temperatures are labels, **not
temperature masks or substitutes for an ionization balance**.

Use `--all-lines` for all 18 entries or `--lines` for a subset. H I is explicitly
labelled an approximation when selected: Ly beta usually needs an NLTE hydrogen
calculation, ideally using the simulation's hydrogen populations when available.
C II and C III also need particular care with ionization equilibrium and
radiative-transfer effects. The script does not automatically interpret the
reader's `H_1`, `HH`, or `HHp` files, whose definitions must first be established
for the particular simulation.

## How this fits the existing repository

The implementation reuses `MuramCube` from `modules/muram.py` and follows the
transpose used by `MURaMData`. It opens only the four required physical cubes.
The full-domain input files are:

| File | Quantity | Required unit |
| --- | --- | --- |
| `Header.<iteration>` | Dimensions, spacings, simulation time | cm for spacings, s for time |
| `eosT.<iteration>` | Temperature | K |
| `eosne.<iteration>` | Electron density | cm^-3 |
| `result_prim_0.<iteration>` | Mass density | g cm^-3 |
| `result_prim_1.<iteration>` | Native vertical velocity `vx` | cm s^-1 |

An iteration is formatted with at least six digits, for example `400000`.
Iteration increments are not physical time intervals; the saved physical time
comes from the header.

The low-level reader documents native axes `(x, y, z)` with **x vertical**.
Applying `.T` gives `(z, y, x)`: two horizontal axes followed by height, as in
the other synthesis scripts. The line-of-sight cell length is therefore
`dX[0]`, including when the three header spacings differ. Positive `vx` points
toward an observer at the upper x boundary, giving a blueshift. `--observer top`
implements this convention; `--observer bottom` reverses the Doppler sign.
Only vertical, axis-aligned rays are implemented. A slanted view requires
resampling along rays and projecting all velocity components.

`--downsample` strides the two horizontal dimensions. It does not average pixels
and does not skip vertical cells or multiply their path lengths. Rectangular
horizontal domains are supported.

The default integrates all vertical cells. `--height-start` and `--height-stop`
select an explicit interval with Python's inclusive-start/exclusive-stop
convention. For a solar observation, choose the lower boundary to exclude the
subsurface convection zone according to your model's geometry. Index 150 in the
existing scripts is not a universal photospheric height. There is no automatic
temperature clipping: a cool foreground structure can absorb radiation from
hotter material behind it, so dropping it changes the question being answered.
The pre-existing MURaM `tau` field is not a SPICE line optical depth and is not
used here.

## Atomic populations and abundances

For element X and ion stage q, define

\[
A_X = n_X/n_H, \qquad F_q(T)=n_{X,q}/n_X, \qquad
p_i(T,n_e)=n_{X,q,i}/n_{X,q}.
\]

The number density of hydrogen nuclei is obtained from mass density:

\[
n_H = X_H\rho/m_H, \qquad
n_i = n_H A_X F_q(T)p_i(T,n_e).
\]

`X_H` defaults to 0.74 and is an explicit assumed composition parameter. Set
`--hydrogen-mass-fraction` to the value consistent with the model EOS. The script
does not infer it from an abundance table. `m_H = 1.00794 m_u` is used; atomic
weights for thermal widths and all other constants are declared in CGS in the
module. No ion mass in kilograms enters a CGS thermal-width expression.

Using rho avoids a fixed `n_H/n_e` conversion, which would be particularly
inappropriate in partially ionized plasma. `n_H` counts nuclei in all hydrogen
ionization states, not only neutral hydrogen or protons.

For each ion, the code calls `ChiantiPy.core.ion(..., temperature=T,
eDensity=ne, abundance=...)`, uses `IoneqOne` for the installed default equilibrium
balance, and calls `populate()` for the fractional level populations.
`Wgfa` supplies transition wavelengths and Einstein A coefficients; `Elvlc`
supplies level IDs and statistical weights `mult = 2J+1`. These interfaces are
documented in the [ChiantiPy source](https://github.com/chianti-atomic/ChiantiPy/blob/master/ChiantiPy/core/Ion.py)
and the [CHIANTI user guide](https://www.chiantidatabase.org/cug.pdf), especially
Sections 3.1 and 6.1–6.5.

The default abundance filename stem is `sun_photospheric_2021_asplund`, present
in the locally installed CHIANTI 11.0.2 database. Choose a different installed
file with `--abundance` for a different composition or coronal FIP enhancement.
Check the actual available filenames: older distributions can differ. The
selected elemental abundance and equilibrium filename are written to metadata.
`populate()` uses its installed default corrections and equilibrium proton
density. Neither the MURaM electron-density cube nor a CHIANTI version number
alone implies that non-equilibrium ionization or the advanced CHIANTI 11
ionization models are being used.

### Transition identification

The default match window is **±0.15 Å around the paper wavelength**. Only
positive (observed-energy) wavelengths and positive A coefficients are accepted.
Theoretical wavelengths, stored as negative numbers in CHIANTI, are excluded.
If no transition matches, the run fails rather than choosing an arbitrarily
distant line. This interpretation of the files follows
[CHIANTI's radiative-data description](https://www.chiantidatabase.org/cug.pdf).

Every matching same-ion component is retained and its opacity/emission is
summed. This handles the closely spaced H I fine-structure components and the
two selected Fe X components at 1028.053 and 1028.082 Å in the local data. The
code prints and saves every selected wavelength, level pair, A coefficient,
statistical weight, and effective oscillator strength. This is a feature
selection window, not a full synthesis of every line falling anywhere within
the final spectral window; other-ion blends are absent.

The local atomic-data check found C III 977.020 Å, O VI 1031.912 and 1037.613 Å,
and Si VII 1049.140 Å. The latter differs from the paper by 0.110 Å and motivated
the 0.15 Å matching default. In the local Si VII file it is the
`2s2 2p4 3P1`–`2s2 2p4 1S0` transition (levels 2–5). All 18 catalogue features
were checked against the installed database. These values are local-data
findings, not changes to the paper's wavelengths.

The nearest selected component defines the exact reference wavelength and
spectral grid center. Both this reference and the original paper wavelength
are saved. For another database release, inspect the printed selections before
increasing `--match-tolerance`.

### Population tables

The stored table is

\[
Q_i(\log_{10}T,\log_{10}n_e)=A_X F_q(T)p_i(T,n_e)=n_i/n_H.
\]

Only the levels needed by the selected transitions are retained. Lines of the
same ion share a solve, including the O VI doublet. Default grids are
`log10(T) = 3..8` with 101 points and `log10(ne) = 6..17` with 45 points.
The interpolation is bilinear in the logarithms of T and ne, applied to the
linear values of Q. It therefore preserves nonnegative corner values and
handles exact zeros without taking their logarithms. Solves are skipped only
where the tabulated equilibrium ion fraction is exactly zero.

The run rejects nonfinite/negative population values, solver errors, and
populations that do not sum to unity. MURaM temperature, density, and electron
density must be positive and finite. Velocities must be finite and below
0.01 c in magnitude. Model values outside the atomic table cause an explicit
error; there is no extrapolation, edge clipping, or silent replacement by zero.
Extend `--log-t`/`--log-ne` to cover your selected atmosphere. Bounds are checked
while processing column blocks.

Population caches are compressed NPZ files named by a signature. The signature
includes grid coordinates, selected transitions, ChiantiPy and database
versions, numerical abundances, the equilibrium array, and hashes of files for
the ion and next higher stage. `--rebuild-tables` forces a rebuild. Cache writes
use an atomic rename. A cache is a performance optimization, not an accuracy
guarantee: compare maps with denser grids, particularly near steep ion-fraction
gradients. Very faint ion-fraction tails can have large relative interpolation
errors even when their absolute opacity contribution is negligible.

## Opacity and spectral synthesis equations

For a transition from lower l to upper u, let \(\lambda_0\) be the physical
rest wavelength in cm and \(\nu_0=c/\lambda_0\). The Doppler width parameter is

\[
b=\sqrt{2k_BT/m_X+\xi^2}, \qquad \Delta\nu_D=\nu_0 b/c.
\]

Here \(\xi\), set by `--microturbulence`, is a **Doppler b parameter** in km/s,
not a Gaussian standard deviation or FWHM. For this convention the Gaussian
velocity standard deviation is \(b/\sqrt{2}\). The temperature is used as both
the electron temperature for populations and the ion temperature for broadening.

With \(v_{\rm red}>0\) away from the observer, the implemented center and profile are

\[
\nu_c=\frac{\nu_0}{1+v_{\rm red}/c}, \qquad
\phi_\nu=\frac{1}{\sqrt{\pi}\Delta\nu_D}
\exp\left[-\left(\frac{\nu-\nu_c}{\Delta\nu_D}\right)^2\right].
\]

This centers the line at \(\lambda_0(1+v_{\rm red}/c)\), consistent with the
nonrelativistic wavelength Doppler convention. Widths use the rest-frame
\(\nu_0\); relativistic transfer factors are neglected. The profile is normalized
over frequency, \(\int\phi_\nu d\nu=1\), and has units Hz^-1.

The extinction coefficient, including stimulated emission, is

\[
\boxed{\alpha_\nu=\frac{\lambda_0^2 A_{ul}}{8\pi}
\left(\frac{g_u}{g_l}n_l-n_u\right)\phi_\nu}
\quad [\mathrm{cm}^{-1}].
\]

Equivalently,

\[
\alpha_\nu=\frac{\pi e^2}{m_e c}f_{lu}
\left(n_l-\frac{g_l}{g_u}n_u\right)\phi_\nu,\qquad
f_{lu}=\frac{m_e c}{8\pi^2 e^2}\frac{g_u}{g_l}\lambda_0^2 A_{ul}.
\]

The implementation uses the A-based expression directly. The effective f
derived by this relation is also recorded; this avoids relying on a gf field
that can be absent or zero for a weak/forbidden transition. In particular, the
lower-level occupation is explicitly solved: it is not assumed to equal the
entire ion population for metastable transitions. Negative extinction from a
population inversion is retained and flagged, rather than clipped away.

For cell-centered model data, the total column optical depth at each frequency is

\[
\boxed{\tau(\nu)=\sum_{k=k_{\min}}^{k_{\max}-1}\alpha_{\nu,k}\,\Delta s_k},
\qquad \Delta s_k=\mathtt{dX[0]}.
\]

Summing complete cells is intentional: a trapezoidal integral over cell centers
would discard half of each boundary cell. A single-cell atmosphere still has
optical depth \(\alpha\Delta s\). The present file format supplies uniform
spacings; a genuinely nonuniform grid needs a per-cell path-length implementation.

The spontaneous emissivity and optically thin comparison intensity are

\[
j_\nu=\frac{h\nu_0}{4\pi}A_{ul}n_u\phi_\nu,\qquad
I_{\lambda,\mathrm{thin}}=\frac{c}{\lambda^2}10^{-8}
\sum_k j_{\nu,k}\Delta s_k.
\]

The factor \(10^{-8}\) converts intensity per cm of wavelength to intensity per
Å. **Optical depth itself gets no wavelength Jacobian**: `tau2D` simply samples
\(\tau_\nu\) at \(\nu=c/\lambda\). Both calculations include the actual column
length and number densities, with no emission-measure normalization hidden in
a contribution function.

The output does not solve \(dI_\nu/ds=-\alpha_\nu I_\nu+j_\nu\). In particular,
multiplying the total thin intensity by `exp(-tau2D)` would incorrectly put all
emission behind a foreground screen. Emission distributed along a ray requires
depth-dependent transfer; resonance scattering additionally changes the source
function and potentially the level populations.

## Running the script

Use an environment with Python 3.9+, NumPy, SciPy, and ChiantiPy. Matplotlib is
needed for `--plot` and the standalone visualization script. `joblib` is
needed only for `--jobs` other than 1. A CHIANTI atomic database is a separate
installation from the Python package; see the
[official download instructions](https://www.chiantidatabase.org/chianti_download.html).

```bash
python -m pip install numpy scipy ChiantiPy joblib matplotlib
export XUVTOP=/path/to/CHIANTI
python compute_spice_optical_depth.py --list-lines
```

The repository's `.venv` had none of these scientific packages during validation.
An existing environment with them was available at
`/Users/memolnar/miniconda3/envs/SPICE_DEMs/bin/python`, with the database at
`/Users/memolnar/Work/CHIANTI`. No environments or database files were modified.

Start with a single C III snapshot and a sparse horizontal sample. The paths and
iteration below follow `example_CIII.py`; adapt them to your machine:

```bash
python compute_spice_optical_depth.py \
  --model-dir /glade/campaign/hao/radmhd/Rempel/Spot_Motion/AR_25Mm_v1000_vz300_disp/3D \
  --iterations 400000 \
  --lines c_3_977 \
  --height-start 150 \
  --downsample 8 \
  --output-dir optical_depth_CIII \
  --jobs 8
```

Here height index 150 is an example borrowed from the repo's existing boundary
choice; establish that it is suitable for this snapshot before interpreting the
column. `--height-stop` defaults to the top of the model. Leaving out
`--height-start` includes the entire input column.

Run all eight default lines over an iteration range:

```bash
python compute_spice_optical_depth.py \
  --model-dir /path/to/MURaM/3D \
  --t-start 400000 --t-end 502000 --time-step 2000 \
  --height-start 150 --downsample 2 \
  --output-dir optical_depth_default --jobs 8
```

Or explicitly choose the resonance diagnostics:

```bash
python compute_spice_optical_depth.py \
  --model-dir /path/to/MURaM/3D --iterations 400000 402000 \
  --lines c_3_977 o_6_1031 o_6_1037 ne_8_770 \
  --height-start 150 --microturbulence 10 \
  --velocity-range 250 --num-waves 1001
```

To compute the whole catalogue, substitute `--all-lines` for `--lines ...`.
For a cooler/denser selected atmosphere or a convergence check, change the
table grids, for example `--log-t 3 8 201 --log-ne 5 18 105`.

Build tables on their own, reusable in subsequent matching runs:

```bash
python compute_spice_optical_depth.py --tables-only \
  --lines c_3_977 o_6_1031 o_6_1037 ne_8_770 \
  --cache-dir opacity_tables --jobs 8
```

Table creation can be expensive for ions with many levels. Processes parallelize
the density samples; snapshots and line features are processed sequentially.
On HPC systems, constrain BLAS threads (for example `OPENBLAS_NUM_THREADS=1`
and `OMP_NUM_THREADS=1`) when using many worker processes, and allocate enough
memory for each worker. `--jobs -1` requests all available CPUs.

Input cubes stay memory mapped. `--chunk-columns` limits the atmospheric arrays
held for integration, and wavelengths are streamed without allocating a full
columns × heights × wavelengths profile array. Output arrays remain in memory
for one line and snapshot: the two large float64 arrays together need about
`16 * n_horizontal_1 * n_horizontal_2 * num_waves` bytes, plus maps, table and
compression overhead. A 512 × 512 field with 401 wavelengths needs about
1.57 GiB for those two arrays alone. Reduce horizontal sampling or wavelength
count when necessary; chunking alone does not reduce output-array memory.

Existing output files are refused unless `--overwrite` is supplied. Different
abundances, height cuts, or broadening choices should normally use different
output directories so they can be compared. Files required for every selected
iteration are checked before building tables.

## Outputs and interpretation

One file is saved per feature and snapshot:
`spice_opacity_c_3_977_400000.npz`, for example.

| Key | Shape | Meaning |
| --- | --- | --- |
| `tau2D` | `(1, native_z, native_y, n_wavelengths)` | Total selected-column spectral optical depth |
| `I2D` | same | Optically thin intensity, erg s^-1 cm^-2 sr^-1 Å^-1 |
| `tau_rest` | `(1, native_z, native_y)` | Optical depth at the exact atomic reference wavelength, including model velocities |
| `tau_peak` | same map shape | Maximum optical depth among the sampled wavelengths |
| `tau_static` | same map shape | Optical depth at the reference wavelength with all resolved velocities set to zero |
| `wvlGrid` | `(n_wavelengths,)` | Increasing physical wavelengths in Å |
| `velocityGrid` | `(n_wavelengths,)` | Offsets relative to the atomic reference in km/s, red positive |
| `detectorWvlGrid` | `(n_wavelengths,)` | `order * wvlGrid`, a coordinate only, not an instrument response |
| `metadata` | scalar JSON string | Atomic selections, units, parameters, versions, file hashes, geometry, diagnostics and command |

The leading dimension is length one for compatibility with the repository's
time-leading `I2D` convention. Horizontal dimensions include any downsampling.
These files can supply `I2D` and `wvlGrid` to the existing visualization script,
but their intensity model must be kept in mind. The metadata loads with
`allow_pickle=False`; no Python object deserialization is required.

`tau_static` is a useful comparison when velocity gradients move absorbing
layers out of resonance. It includes thermal and selected microturbulent
broadening. For a multiplet it is the sum of component opacities at the **shared
reference wavelength**, not a sum of each component's individual peak.

`tau_peak` depends on the chosen finite spectral grid and is not a guaranteed
continuous maximum. The script warns when resolved velocities exceed the
spectral window or when the sample spacing is greater than one third of the
smallest Doppler b. These are basic checks: extend the window enough to capture
the wings and repeat with finer sampling until the maps and integrated
intensities converge. Even in the absence of a warning, a near-edge line can
lose wing flux. The odd default of 401 samples guarantees the exact reference
wavelength is present.

### Matplotlib maps of maximum column optical depth

Add `--plot` to a synthesis command to save a PNG beside every NPZ, for example:

```bash
python compute_spice_optical_depth.py \
  --model-dir /path/to/MURaM/3D --iterations 400000 \
  --lines c_3_977 --height-start 150 --downsample 8 --plot
```

The image is named `spice_opacity_c_3_977_400000_tau_peak.png`. To visualize
already saved results without repeating synthesis:

```bash
python visualize_spice_optical_depth.py optical_depth_output/spice_opacity_*.npz \
  --output-dir optical_depth_maps
```

The standalone command reads only `tau_peak` and `metadata`, and requires NumPy
and Matplotlib. It does not load the spectral cubes, CHIANTI, or MURaM inputs.
Use `--overwrite` to replace existing images. Both commands support
`--plot-scale log|linear`, `--plot-limits MIN MAX`, `--plot-cmap`, and `--plot-dpi`.
Defaults are logarithmic grayscale, fixed optical-depth limits of `1e-3` to
`5`, and 180 DPI. Values at or below `1e-3` are black and values at or above
`5` are white; the colorbar spans those same limits and has the map's height.
Limits are optical depths, not their logarithms. Use `--plot-limits` or
`--plot-cmap` to choose a different display scale.

Each pixel shows `tau_peak = max_wavelength(integral_LOS alpha_lambda ds)`.
For nonnegative extinction, cumulative optical depth is largest through the
complete selected column. The map then takes its maximum over the sampled
wavelengths; it is not a maximum of the local extinction coefficient. It does
not calculate a depthwise maximum of cumulative optical depth for inverted
populations, where extinction can be negative.

The plot labels the two horizontal coordinates **x [Mm]** and **y [Mm]**:
plot x is native MURaM y, plot y is native MURaM z, and native MURaM x remains
the line of sight. Stored maps have row/column order `(native_z, native_y)`,
so they are plotted without transposition. Coordinates are relative to the
domain's horizontal lower boundaries. Sampled cell centers are calculated as
`(native_index + 0.5) * cell_spacing_cm / 1e8`; display edges lie halfway
between sampled columns and end at the domain boundaries. Unequal spacings,
rectangular domains, and downsampling factors that do not divide the domain
size are supported. Colors represent nearest sampled columns, not pixel averages.

A white contour marks `tau_peak = 1` when the finite map spans that value and
both horizontal dimensions have multiple samples. In logarithmic maps, finite
values at or below the lower limit, including zero and negative values, are
black; nonfinite pixels are gray and their count is annotated. Linear colors
retain finite negative optical depths. Rendering uses a noninteractive
Matplotlib canvas, so synthesis on an HPC compute node does not open windows.

Read a result and use the plotting helper from Python:

```python
import json
import numpy as np
from scipy.integrate import trapezoid
from visualize_spice_optical_depth import save_tau_peak_map

with np.load("optical_depth_CIII/spice_opacity_c_3_977_400000.npz",
             allow_pickle=False) as data:
    tau_peak = data["tau_peak"][0]
    tau_static = data["tau_static"][0]
    integrated_thin = trapezoid(data["I2D"][0], data["wvlGrid"], axis=-1)
    metadata = json.loads(str(data["metadata"]))

print("Fraction of sampled columns with tau_peak >= 1:", (tau_peak >= 1).mean())
print("Atomic components:", metadata["atomic"]["components"])
save_tau_peak_map(tau_peak, metadata, "CIII_tau_peak.png", limits=(0.01, 100))
```

For non-inverted populations, `tau << 1` means weak line extinction along the
selected column, while `tau >= 1` indicates potentially consequential opacity.
The total optical depth through a column can exceed the optical depth between
a particular emitting layer and the observer. These products do not locate a
tau-one surface or save cumulative depth-dependent optical depths.

The O VI doublet provides a useful internal comparison. With negligible upper
populations, equal lower-level columns, and the same velocity broadening,
its static peak-depth ratio scales as `f * wavelength`, approximately two.
It is not an assumption enforced by the code, and observed intensity ratios
also depend on excitation and transfer.

## Limits and validation

The calculation assumes a Maxwellian electron distribution, equilibrium ion
fractions, statistical-equilibrium level populations, a single ion/electron
temperature, and Gaussian profiles. It includes neither an external pumping
radiation field nor a self-consistent scattering source function. Natural and
pressure wings, partial redistribution, non-equilibrium ionization, and
instrumental PSFs/effective areas are absent. It is not a calculation of SPICE
detector counts. Absence of line opacity does not by itself establish that an
observed spectrum is described fully by collisional emission.

Continuum opacity is also absent. In particular, neutral-hydrogen bound-free
absorption can affect the short-wavelength band below the Lyman edge near
912 Å and the physical Si XII wavelength. A full foreground absorption model
would need the appropriate neutral populations and continuum cross sections.
The saved optical depths refer exclusively to the selected bound-bound
components.

The test suite uses analytic slabs and artificial MURaM-format files. It checks
absolute line-center normalization, total cell length (including a one-cell
column), stimulated emission and inversions, thermal and microturbulent widths,
Doppler shifts, multiplet addition, profile normalization, and the wavelength
emissivity Jacobian. A nonsquare fixture with unequal grid spacings checks the
reader transpose, vertical path length, height slicing, and observer sign.
Invalid cells, out-of-grid values, missing inputs, and output collisions are
also exercised.

```bash
python -m unittest discover -s tests -v
SINSPICE_TEST_CHIANTI=1 python -m unittest discover -s tests -v
```

The second command additionally checks actual catalogue identifications and
builds a small C III table, compares a tabulated population against a direct
ChiantiPy solve, verifies cache reuse, and runs the CLI workflow on a generated
MURaM-format snapshot, including its PNG output. Plotting tests check the map
orientation and physical extent with unequal spacings and non-divisible
downsampling, zero/negative/nonfinite pixels, and standalone plotting from an
NPZ containing only the map and metadata. All 15 tests passed using ChiantiPy 0.15.2 and CHIANTI
11.0.2. Separate smoke runs built tables for all 18 features on a 2 × 2 grid
and built the full default 101 × 45 C III table with two worker processes.
The tiny smoke-test grids are deliberately insufficient for science; they
exercise the interfaces and solver paths. Artificial fixtures are not solar-model validation.
Production MURaM snapshots at the hard-coded GLADE paths were not available
locally, so no physical opacity maps for those simulations are claimed here.

For a scientific production run, first establish the correct geometry and
composition, then converge the population grids and spectral sampling. Use
non-LTE/NEI calculations where the inferred optical depths or timescales demand
them. Record the saved atomic metadata alongside any reported results.
