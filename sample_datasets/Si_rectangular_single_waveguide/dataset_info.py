import numpy as np

class DatasetInfo:
    def __init__(self):
        self.description = {
            "Description" :"This dataset contains overlap, neff, TE_pol",
            "material": "Si",
            "structure": "Single Waveguide",
            "corse mesh size": "30nm",
            "fine mesh size": "5nm"
        }
        self.file_structure = {
            "wg_crossection.lms": "Lumerical lms file where the model has parameters of parameter_names",
            "dataset_info.py": "Python class",
            "overlap.pkl": "pickle file of dictionary",
            "neff.pkl": "pickle file",
            "TE_pol.pkl": "pickle file"
        }
        self.FDE_crosssection = {
            "crosssection_x" : "y",
            "crosssection_y" : "z"
        }
        self.mode_numbers = 10
        self.wavelength = 1.55e-6
        self.cladding_index = 1.44
        self.parameter_names = ["top_width", "curvature"]
        self.parameter_types = {
            "top_width" : "Length",
            "curvature" : "Number"
        }
        self.parameters = {
            "top_width": np.round(np.linspace(1e-6, 3e-6, 101), 9),    # m (20nm)
            "curvature": np.round(np.linspace(0, 330000, 166), 0)  # m^-1 R:5um ~ infinity (2000)
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