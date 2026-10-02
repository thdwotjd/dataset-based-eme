"""SiN curvature-only DREME convergence dataset metadata.

Use this directory in a fresh Python process. DataUpdater imports this
module by the generic name ``dataset_info``.
"""

import numpy as np


class DatasetInfo:
    def __init__(self):
        self.description = {
            "description": "SiN curvature-only DREME convergence dataset",
            "material": "SiN",
            "cladding": "SiO2",
            "structure": "300 nm fully etched rectangular waveguide",
            "fixed_top_width_um": 5,
            "nominal_curvature_step_per_m": 5000,
            "curvature_range_per_m": [0, 50000],
            "fde_y_span_um": 10.0,
            "fde_z_span_um": 3.5,
            "fine_mesh_y_span_um": 6.0,
            "fine_mesh_z_span_um": 1.0,
            "coarse_mesh_nm": 25.0,
            "fine_mesh_nm": 10.0,
            "number_of_trial_modes": 30,
            "wavelength_um": 1.55,
        }
        self.file_structure = {
            "wg_crosssection.lms": "Copy of the SiN FDE setup model",
            "dataset_info.py": "Dataset metadata and parameter grid",
            "overlap.pkl": "Bidirectional overlap dictionary",
            "neff.pkl": "Complex effective-index dictionary",
            "TE_pol.pkl": "TE-polarization dictionary",
            "run_manifest.json": "Configuration and run provenance",
            "integrity_report.json": "Post-run data checks",
        }
        self.FDE_crosssection = {"crosssection_x": "y", "crosssection_y": "z"}
        self.mode_numbers = 30
        self.wavelength = 1.55e-6
        self.cladding_index = 1.44
        self.parameter_names = ["top_width", "curvature"]
        self.parameter_types = {"top_width": "Length", "curvature": "Number"}
        self.parameters = {
            "top_width": np.asarray([5e-6], dtype=float),
            "curvature": np.arange(0, 50000 + 5000, 5000, dtype=float),
        }

    def get_description(self):
        return self.description

    def get_file_structure(self):
        return self.file_structure

    def get_parameter_names(self):
        return self.parameter_names

    def get_parameter_grid(self):
        return self.parameters

    def get_parameter_types(self):
        return self.parameter_types

    def get_mode_numbers(self):
        return self.mode_numbers

    def get_crosssection_x(self):
        return self.FDE_crosssection["crosssection_x"]

    def get_crosssection_y(self):
        return self.FDE_crosssection["crosssection_y"]

    def get_wavelength(self):
        return self.wavelength

    def get_cladding_index(self):
        return self.cladding_index

    def _is_variable_FDE(self):
        return False
