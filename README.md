# Immersion-cooled 18650 cell in OpenFOAM: case files, verification suite, result tables

Companion repository of the article

> A. R. Lakshmipathi and V. Dhanapathy, "Conjugate Heat Transfer Analysis of Single-Phase
> Immersion Cooling for an 18650 Cell at Peak Discharge: Anisotropic Interface Coupling
> Pitfalls and Low-Cost Cloud Execution" (manuscript under review; the reference will be
> completed on publication).

Archived release v1.0.0: <https://doi.org/10.5281/zenodo.23344394> (all versions: <https://doi.org/10.5281/zenodo.23344393>).

The article and its Supplementary Material are the description of the model, of the
verification problems and of every result. This repository holds what is needed to repeat
the runs.

| Folder or file | Contents |
|---|---|
| `cases/` | Dictionaries of the 26 full-discharge runs of the cell (`0.orig`, `constant`, `system`), with run scripts. No meshes and no results. |
| `tools/` | Mesh generator, coolant-property generator and the analysis scripts. |
| `benchmarks/interface/` | Verification suite for the coupling of an anisotropic solid: generator of the cases, exact solutions, drivers, the source of two user libraries, and the table of all benchmark runs. |
| `results/`, `numbers_quoted.csv` | Result tables of all runs and the list of the numbers the article quotes, each with its source. `SUPPLEMENTARY_DATA.txt` describes every file. These are the data files of the Supplementary Material, at the same paths. |

## Requirements

- OpenFOAM v2606 (ESI/OpenCFD line, <https://www.openfoam.com>). The commands below use its
  `openfoam2606` wrapper. The cell cases use the unmodified `chtMultiRegionFoam`.
- Python 3 with numpy. The verification suite also needs scipy; `tools/analyze.py` with
  plots and `make_figures.py` need matplotlib.

## The cell cases

```sh
cd cases/r1m_C13
openfoam2606 -c ./Allrun        # mesh, region split, decomposition, solver, reconstruction
```

`Allrun.pre` alone builds the mesh and the initial fields. The number of MPI ranks is
`numberOfSubdomains` in `system/decomposeParDict` (4 as shipped). One discharge is 267.4 s
of physical time; Section VI of the article gives the run times. `Allclean` removes
everything a run produced.

Points that are easy to miss:

- **Outer correctors.** `chtMultiRegionFoam` of v2606 reads `nOuterCorrectors` from the
  top-level `system/fvSolution` only. The entry of the same name in
  `system/fluid/fvSolution` is not read; it is left in the files because they are the files
  of the runs. `nCorrectors` is read from `system/fluid/fvSolution`.
- **Interface option.** `kappaMethod` of the coupled patch in `0.orig/battery/T` selects the
  directional or the scalar option that the article compares.
- **Time step.** `maxDeltaT` is set per mesh (table below). The article explains why.
- **Velocity limiter.** `constant/fluid/fvOptions` holds an inactive `limitVelocity` entry.
  No case here uses it.
- **Expansion-port variants** (`*_portFloor`) carry the initial pressure field of their mesh
  in `0.orig/fluid/p`; `Allrun.pre` creates the port patch.

Meshes (`system/blockMeshDict.<name>`, written by `tools/make_blockMesh.py`):

| mesh | cells | `make_blockMesh.py` arguments |
|---|---|---|
| xcoarse | 27,840 | `--nc 8 --nrs 5 --nrf 20 --nz1 14 --nz2 32 --nz3 14` |
| coarse | 76,626 | `--nc 11 --nrs 7 --nrf 28 --nz1 20 --nz2 46 --nz3 20` |
| medium | 224,576 | none (the defaults) |
| fine | 652,740 | `--nc 23 --nrs 14 --nrf 57 --nz1 40 --nz2 92 --nz3 40` |

`python3 tools/make_blockMesh.py [arguments] --out FILE` reproduces the four dictionaries
byte for byte. `python3 tools/make_thermo.py FILE` (run from the repository root) writes the
coolant property file; its viscosity polynomial is a least-squares fit, and its coefficients
can differ from the shipped file in the ninth significant digit, depending on the platform.

### The runs

Settings as read from the dictionaries in this repository. Supplementary Note S11 of the
article relates these identifiers to the names used in the article.

| run | mesh | pressure correctors | outer passes | maxCo | maxDeltaT [s] | interface option | k (radial, tangential, axial) [W/(m K)] | heat source [W/m3] | cp law (c0, c1) | expansion port |
|---|---|---|---|---|---|---|---|---|---|---|
| `r1x_C60` | xcoarse | 5 | 1 | 1 | 0.06 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_C30` | coarse | 5 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1m_C13` | medium | 5 | 1 | 1 | 0.013 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1f_C6` | fine | 5 | 1 | 1 | 0.006 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_C15` | coarse | 5 | 1 | 0.5 | 0.015 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1m_C065` | medium | 5 | 1 | 0.5 | 0.0065 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_A30` | coarse | 3 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_A15` | coarse | 3 | 1 | 0.5 | 0.015 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1m_A13` | medium | 3 | 1 | 1 | 0.013 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1m_A065` | medium | 3 | 1 | 0.5 | 0.0065 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1f_A6` | fine | 3 | 1 | 1 | 0.006 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_A4_30` | coarse | 4 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_B30` | coarse | 2 | 3 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_C30_def` | coarse | 5 | 1 | 1 | 0.03 | `solidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1m_C13_def` | medium | 5 | 1 | 1 | 0.013 | `solidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_A30_def` | coarse | 3 | 1 | 1 | 0.03 | `solidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1m_A13_def` | medium | 3 | 1 | 1 | 0.013 | `solidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_C30_kr020` | coarse | 5 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.20 0.20 30.4 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_C30_kr220` | coarse | 5 | 1 | 1 | 0.03 | `directionalSolidThermo` | 2.2 2.2 13 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_A30_kr020` | coarse | 3 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.20 0.20 30.4 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_A30_kr220` | coarse | 3 | 1 | 1 | 0.03 | `directionalSolidThermo` | 2.2 2.2 13 | 1.10032e6 | 824.66 3.6 | no |
| `r1c_C30_q20` | coarse | 5 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.20914e+06 | 824.66 3.6 | no |
| `r1c_A30_q20` | coarse | 3 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.20914e+06 | 824.66 3.6 | no |
| `r1c_C30_cpconst` | coarse | 5 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 1934 0 | no |
| `r1c_C30_portFloor` | coarse | 5 | 1 | 1 | 0.03 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | yes |
| `r1m_C13_portFloor` | medium | 5 | 1 | 1 | 0.013 | `directionalSolidThermo` | 0.8 0.8 28 | 1.10032e6 | 824.66 3.6 | yes |

Reading the identifiers: `x`, `c`, `m`, `f` name the mesh. `C` is the setting of the
article's results (five pressure correctors, one outer pass); `A` has three correctors,
`A4` four, and `B` three outer passes with two correctors each. The number is the largest
time step (`30` = 0.03 s, `065` = 0.0065 s). Suffixes: `_def` scalar interface option,
`_kr020` and `_kr220` other conductivities of the cell, `_q20` 20 W heat rate, `_cpconst`
constant heat capacity of the coolant, `_portFloor` expansion port in the floor. The heat
source is given per volume of the cell; the heat-capacity law of the coolant is
cp = c0 + c1 T in J/(kg K), T in K.

## Analysis scripts

Run from the repository root on a finished case directory.

| Script | Purpose |
|---|---|
| `tools/analyze.py --case DIR --no-plots` | Temperatures, heat-transfer coefficient, peak speed and energy balance from the monitors. |
| `tools/run_summary.py DIR ...` | One row per run, with the heat-transfer coefficient at 200 s and 250 s. |
| `tools/pilots/pilot_metrics.py DIR` | Per-time-step instruments: energy budget of the coolant and the test for a velocity that alternates from step to step. `--selftest` checks the arithmetic on synthetic files. |
| `tools/gci_family.py DIR_coarsest ... DIR_finest` | Grid-convergence analysis of a mesh family. |

## Verification suite

```sh
cd benchmarks/interface
openfoam2606 -c 'derivation/coNormalInterface/Allwmake && suite/lib/coNormalCoupling/Allwmake'
export CONORMAL_LIB=$PWD/derivation/coNormalInterface/lib/libcoNormalInterface.so
openfoam2606 -c 'bash suite/scripts/run_smoke.sh p1'                      # patch test, 14 cases
openfoam2606 -c 'python3 suite/scripts/runpts.py $HOME/bc_test x_bc_smoke'   # corrected boundary condition
```

Both libraries are built into a `lib` folder next to their source; nothing is written into
the OpenFOAM installation. `suite/scripts/gencase.py` generates a case, `exact.py` holds the
exact solutions, `post.py` evaluates a run, and `blocks.py` lists the runs of every block of
`results/results.csv` (881 runs; columns in `results/results_columns.txt`).

- `derivation/coNormalInterface`: a function object that supplies the tangential part of the
  co-normal flux to the stock coupled condition, with a guard against the scalar option on an
  anisotropic solid.
- `suite/lib/coNormalCoupling`: the same correction as a boundary condition, a guard that
  stops a run with the scalar option on an anisotropic solid, and a wall-heat-flux object
  that uses the tensor. See `README.txt` there.

**The corrected coupling is conditionally stable.** It is research code for verification,
not a general replacement of the stock boundary condition. The article states the
conditions under which it diverged.

## Reading the comments in the files

The dictionaries are those of the runs and the scripts are those that produced the results;
only comments were edited for publication. Many comments are working notes. Where a comment
and the article differ, the article holds. Comments cite working documents (`DESIGN.md`,
`RESULTS.md`, `DERIVATION.md`, `PILOTS.md`) that are not part of this repository; their
content is in the article and its Supplementary Material. In `pilot_metrics.py`, "pilot"
means a short trial run, and "P0", "as run" and "legacy" refer to the configuration with
one pressure corrector and the velocity limiter (`results/r1_runs/one_corrector_limiter.csv`).

## What was tested from this folder

On 13 October 2026, OpenFOAM v2606, Linux on a four-core ARM computer:

- `cases/r1x_C60` (extra-coarse mesh) was run with `./Allrun` on 4 ranks for the first 2 s of
  the discharge (61 time steps) and compared with the monitors of the run that the article
  reports: same sequence of time steps; temperature monitors within 3e-8 K; plume speed
  within 3e-7 m/s (the flow is just starting, about 4e-5 m/s). `tools/analyze.py` was run on
  the result.
- `cases/r1c_C30_portFloor` (coarse mesh, expansion port) was started with `./Allrun` on
  4 ranks: mesh and port patch were created, `checkMesh` reported no error, and the first
  four time steps have the time-step sizes of the run that the article reports, with
  temperature monitors within 1e-12 K. It was stopped there.
- Both libraries were built. `run_smoke.sh p1` ran its 14 cases, and the block `x_bc_smoke`
  ran its 12 cases, of which 4 are stopped by the guard, as intended. With the corrected
  coupling the patch test at 30 degrees agrees with the exact solution within 2e-11 K.
- `tools/make_blockMesh.py` reproduced the four mesh dictionaries byte for byte, and
  `tools/pilots/pilot_metrics.py --selftest` passed.

Not repeated from this folder: the other 24 cell cases, any full discharge, and the full
matrix of benchmark runs.

## Licence

Case dictionaries, scripts and library source: GNU General Public License, version 3 or
later (`LICENSE`). Result tables and the list of quoted numbers: CC BY 4.0
(`LICENSE-DATA.md`).

This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of
the OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade
marks.

## Contact

The corresponding author named in the article.

The case files, the scripts and this repository were prepared with an AI assistant, Claude Code (Anthropic), under the direction of the authors, who reviewed the content and are responsible for it.
