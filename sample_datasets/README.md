# SiN curvature sample datasets

These two precomputed datasets model a 5 µm wide, 300 nm thick, fully etched SiN waveguide with SiO2 cladding at a wavelength of 1.55 µm. Curvature spans 0–50,000 m⁻¹. The folder names encode the curvature grid spacing:

| Directory | Grid spacing | Grid points |
| --- | ---: | ---: |
| `w_5um_curv_01000pm` | 1,000 m⁻¹ | 51 |
| `w_5um_curv_05000pm` | 5,000 m⁻¹ | 11 |

Each folder contains `dataset_info.py`, complex effective indices (`neff.pkl`), TE polarization fractions (`TE_pol.pkl`), bidirectional modal overlaps (`overlap.pkl`), the original `integrity_report.json`, and a copy of the Lumerical cross-section model (`wg_crosssection.lms`). The reports passed: all expected grid points and neighbor overlaps are present, with 60 stored forward/backward modal entries per point. Binary pickle files use Git LFS.

The datasets are ready for propagation with `DataUpdater(path, is_testmode=True)`. They omit the machine-specific generation manifest and notebook. Regenerating the modal dataset requires a compatible Lumerical installation and the original generation workflow; the supplied model alone is insufficient. The sample comparison starts no FDE or FDTD calculation.

Run the [Euler-bend comparison](../examples/SiN_curvature_interpolation.py) to compare the existing phase calculation and `neff_interpolation=True` with both grids and archived FDTD output powers. The [validation guide](../docs/SiN_curvature_interpolation_example.rst) explains the results and their limits.
