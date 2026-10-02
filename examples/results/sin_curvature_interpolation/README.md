# SiN Euler-bend interpolation comparison

Run from the repository root using Python 3.9–3.11 and the dependencies in `requirements.txt`:

```bash
python examples/SiN_curvature_interpolation.py --step 1000
python examples/SiN_curvature_interpolation.py --step 5000
python examples/SiN_curvature_interpolation.py --plot-only
```

Run the two dataset steps in separate Python processes because `DataUpdater` imports each folder's metadata as `dataset_info`. The runs use the integrated `sim.EME(..., neff_interpolation=True)` option and repeat the same cases with interpolation disabled. `--plot-only` recreates both figures and the FDTD error summary from the included CSV files without running EME.

The 5 µm wide, 300 nm thick SiN bends have effective radii 40–60 µm in 2 µm increments. Partial Euler and Full Euler are separate structures. Inputs TE0 and TE1 are plotted against output TE0, TE1, and the sum of higher guided TE modes (TE2+). The mode names are assigned by descending real effective index among guided TE modes at each end. Curves use a linear 0–1 output-power axis; solid lines use interpolated phase, dotted lines use the existing phase, and crosses show archived FDTD values. FDTD reference data exist only at 40, 50, and 60 µm.

The compact [FDTD reference CSV](../../data/SiN_w5_curvature_fdtd_reference.csv) contains guided output powers extracted from the prior SiN curvature convergence study. The CSVs here contain the DREME results; no new FDTD simulation was performed. `fdtd_error_summary.csv` reports mean and maximum absolute error in **percentage points** over 18 comparisons per shape and dataset step: 3 radii × 2 input modes × 3 output groups. These are case-specific comparisons, not a guarantee that interpolation improves every geometry or that a coarse grid has converged.

The datasets were generated with 30 trial modes, FDE cross-section spans of 10 × 3.5 µm, a fine mesh of 10 nm in a 6 × 1 µm region, and 25 nm outside. The simulation uses geometry resolution 3000, `EMEStabilityConfig()` defaults, `force_unitary=False`, and `force_passive=False`. Only one dataset parameter (curvature) varies along each bend. The Euler centerline formula in the script is the one used to generate the prior FDTD cases.
