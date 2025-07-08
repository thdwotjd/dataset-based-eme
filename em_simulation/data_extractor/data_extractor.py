import sys
import gc
import os
import yaml
import importlib.util
from copy import deepcopy

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(project_root, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

path_name_start = config["lumerical_path"]
path_name_start2 = config["ansys_path"]
path_name_end = config["api_path"]
path_name_end2 = config["ansys_api_path"]
version = config["versions"]

is_import_api = False

for i in range(len(version)):
    try:
        sys.path.append(path_name_start + version[i] + path_name_end)
        import lumapi
        is_import_api = True
        break
    except:
        if i == len(version)-1:
            print("failed to load lumerical api (Dataset acquisition) (Lumerical Version)")
        continue

if not is_import_api:
    for i in range(len(version)):
        try:
            sys.path.append(path_name_start2 + version[i] + path_name_end2)
            import lumapi
            is_import_api = True
            break
        except:
            if i == len(version)-1:
                print("failed to load lumerical api (Dataset acquisition) (Ansys version)")
        continue


class DataExtractor:
    """
    Dataset dictionary structures:
    """
    def __init__(self, data_directory, is_testmode = False):
        self.data_directory = data_directory
        

        try:
            dataset_info_path = os.path.join(data_directory, 'dataset_info.py')
            spec = importlib.util.spec_from_file_location('dataset_info', dataset_info_path)
            dataset_info_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_info_module)
            DatasetInfo = dataset_info_module.DatasetInfo
            self.data_info = DatasetInfo()
        except Exception as e:
            print(f"Failed to import DatasetInfo from {dataset_info_path}: {e}")
            return
        
        self.fde_file = data_directory + "/wg_crosssection.lms"
        self.overlap_filename = data_directory + "/overlap.pkl"
        self.neff_filename = data_directory + "/neff.pkl"
        self.TE_pol_filename = data_directory + "/TE_pol.pkl"
        # if not is_testmode:
        self.mode = lumapi.MODE(filename = self.fde_file)

        self.parameter_names = self.get_parameter_names()
        self.parameter_types = self.get_parameter_types()
        self.mode_numbers = self.get_mode_numbers()
        self.crosssection_x = self.get_crosssection_x()
        self.crosssection_y = self.get_crosssection_y()
        self.wavelength = self.get_wavelength()
        self.is_testmode = is_testmode

    #region get simple data from dataset    
    def get_parameter_names(self):
        return self.data_info.get_parameter_names()
    
    def get_parameter_types(self):
        return self.data_info.get_parameter_types()
    
    def get_mode_numbers(self):
        return self.data_info.get_mode_numbers()
    
    def get_crosssection_x(self):
        return self.data_info.get_crosssection_x()
    
    def get_crosssection_y(self):
        return self.data_info.get_crosssection_y()
    
    def get_wavelength(self):
        return self.data_info.get_wavelength()
    
    def get_cladding_index(self):
        return self.data_info.get_cladding_index()
    #endregion get simple data from dataset

    #region FDE sweep
    def set_fde_sweep(self, parameter_points):
        # set sweep name
        self.mode.deletesweep("sweep")
        self.mode.addsweep(0)

        # set sweep parameters
        param_points_num = len(parameter_points)
        self.mode.setsweep("sweep", "type","values")
        self.mode.setsweep("sweep", "number of points", param_points_num)

        for i in range(len(self.parameter_names)):
            parameters = dict()
            param_name = self.parameter_names[i]
            parameters["Name"] = param_name
            parameters["Parameter"] = "::model::" + param_name
            parameters["Type"] = self.parameter_types[param_name]

            for j in range(param_points_num):
                parameters["value_" + str(j+1)] = parameter_points[j][i]
            self.mode.addsweepparameter("sweep", parameters)

        # set sweep results
        neffs_dict = dict()
        Es_dict = dict()
        Hs_dict = dict()
        TE_pols_dict = dict()
        crosssection_x = dict()
        crosssection_y = dict()

        crosssection_x["Name"] = "x"
        crosssection_x["Result"] = "::model::FDE::data::material::" + self.crosssection_x
        crosssection_y["Name"] = "y"
        crosssection_y["Result"] = "::model::FDE::data::material::" + self.crosssection_y
        for i in range(self.mode_numbers):
            neffs_dict["mode" + str(i+1)] = dict()
            neffs_dict["mode" + str(i+1)]["Name"] = "mode" + str(i+1) + "_neff"
            neffs_dict["mode" + str(i+1)]["Result"] = "::model::FDE::data::mode" + str(i+1) + "::neff"

            Es_dict["mode" + str(i+1)] = dict()
            Es_dict["mode" + str(i+1)]["Name"] = "mode" + str(i+1) + "_E"
            Es_dict["mode" + str(i+1)]["Result"] = "::model::FDE::data::mode" + str(i+1) + "::E"

            Hs_dict["mode" + str(i+1)] = dict()
            Hs_dict["mode" + str(i+1)]["Name"] = "mode" + str(i+1) + "_H"
            Hs_dict["mode" + str(i+1)]["Result"] = "::model::FDE::data::mode" + str(i+1) + "::H"

            TE_pols_dict["mode" + str(i+1)] = dict()
            TE_pols_dict["mode" + str(i+1)]["Name"] = "mode" + str(i+1) + "_TEpol"
            TE_pols_dict["mode" + str(i+1)]["Result"] = "::model::FDE::data::mode" + str(i+1) + "::TE polarization fraction"
        
        self.mode.addsweepresult("sweep", crosssection_x)
        self.mode.addsweepresult("sweep", crosssection_y)
        for j in range(self.mode_numbers):
            self.mode.addsweepresult("sweep", neffs_dict["mode" + str(j+1)])
            self.mode.addsweepresult("sweep", Es_dict["mode" + str(j+1)])
            self.mode.addsweepresult("sweep", Hs_dict["mode" + str(j+1)])
            self.mode.addsweepresult("sweep", TE_pols_dict["mode" + str(j+1)])

    def run_fde_sweep(self):
        self.mode.runsweep("sweep")

    def post_process_sweep_data(self, parameter_points):
        param_points_num = len(parameter_points)
        # extract cross section array
        x = self.mode.getsweepresult("sweep", "x")[self.crosssection_x][:,0]
        y = self.mode.getsweepresult("sweep", "y")[self.crosssection_y][:,0]

        # result ndarray initialization
        neffs = np.zeros(shape = (param_points_num, 2*self.mode_numbers,), dtype=np.complex64)
        TE_pols = np.zeros(shape = (param_points_num, 2*self.mode_numbers,), dtype=np.float16)
        
        ## extract sweep data
        # neff & TE pols
        for i in range(self.mode_numbers):
            # forward propagating mode
            neffs[:,i] = self.correct_gain(np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_neff")["neff"]))
            TE_pols[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_TEpol")["TE polarization fraction"])

            # backward propagating mode
            neffs[:,i+self.mode_numbers] = -deepcopy(neffs[:,i])
            TE_pols[:,i+self.mode_numbers] = deepcopy(TE_pols[:,i])
        
        ## overlaps
        # Since the number parameter could be large, 
        # overlaps are calculated from the divided Efields and Hfields sections.
        prop_axis = list({"x", "y", "z"} - {self.crosssection_x, self.crosssection_y})[0]
        prop_axis = 0 if prop_axis == 'x' else 1 if prop_axis == 'y' else 2

        # initiallize overlaps
        overlap_ab = np.zeros(shape=(param_points_num-1, 2*self.mode_numbers, 2*self.mode_numbers), dtype =np.complex64)
        overlap_ba = np.zeros(shape=(param_points_num-1, 2*self.mode_numbers, 2*self.mode_numbers), dtype =np.complex64)

        Efields = np.zeros(shape = (param_points_num, 2*self.mode_numbers, 3, len(x), len(y)), dtype=np.complex64)
        Hfields = np.zeros(shape = (param_points_num, 2*self.mode_numbers, 3, len(x), len(y)), dtype=np.complex64)

        for i in range(self.mode_numbers):
            Efields[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_E")["E"]).transpose(2,3,0,1)   # (len_x, len_y, num_points, 3) -> (num_points, 3, len_x, len_y)
            Hfields[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_H")["H"]).transpose(2,3,0,1)
        Efields, Hfields = self.normalize_field(Efields, Hfields, x, y, prop_axis)
        for i in range(self.mode_numbers):
            # backward propagating mode
            # Since, overlap concerns the transverse field of the field, 
            # the following is valid for reciprocal medium
            Efields[:,i+self.mode_numbers] = np.conjugate(deepcopy(Efields[:,i]))
            Hfields[:,i+self.mode_numbers] = -np.conjugate(deepcopy(Hfields[:,i]))

        for i in range(param_points_num-1):
            overlap_ab[i] = self.calc_overlap(Efields[i], Hfields[i+1], x, y, prop_axis)
            overlap_ba[i] = self.calc_overlap(Efields[i+1], Hfields[i], x, y, prop_axis)

        return neffs, TE_pols, overlap_ab, overlap_ba        
    #endregion FDE sweep

    @staticmethod
    def compute_differences(arr):
        # Initialize an empty array with the same length as the input array
        output = np.empty_like(arr)
        # Calculate the difference between adjacent elements and store in the output array
        output[:-1] = arr[1:] - arr[:-1]
        # Set the last component as the prior component
        output[-1] = output[-2]
        
        return output
    
    def calc_overlap(self, Efield, Hfield, x, y, prop_axis):
        
        # Add new axes for broadcasting
        Efield_expanded = Efield[:, np.newaxis, :, :, :]
        Hfield_expanded = Hfield[np.newaxis, :, :, :, :]
        
        # Compute cross products using broadcasting
        cross_product = np.cross(Efield_expanded, Hfield_expanded, axis=2)

        # del Efield_expanded
        del Hfield_expanded
        gc.collect()

        x = self.compute_differences(x)
        y = self.compute_differences(y)

        weight_mask = np.outer(x, y)

        # The shape of weight_mask should be the same as func
        weight_mask_expanded = weight_mask[np.newaxis, np.newaxis,np.newaxis,:,:]
        
        overlap = (weight_mask_expanded * cross_product).sum(axis=(3,4))
        overlap = overlap[:,:,prop_axis]/2
        return overlap
    
    #region data manipulation
    @staticmethod
    def correct_gain(neffs):
        """
        neff: np array of imaginary values
        It imaginary part of neff has negative sign, it force imaginary part to zero. 
        """
        neff_imag = neffs.imag
        neff_imag = np.where(neff_imag < 0, 0, neff_imag)
        return neffs.real + neff_imag*1j
    
    def normalize_field(self, Efield, Hfield, x, y, prop_axis):
        """
        Normalize electric and magnetic field arrays.

        Parameters:
            Efield (ndarray): Electric fields, shape (n_points, n_modes, 3, len(x), len(y))
            Hfield (ndarray): Magnetic fields, same shape as Efield.
            x (ndarray): 1D array of x mesh points.
            y (ndarray): 1D array of y mesh points.
            prop_axis (int): Propagation axis (0: x, 1: y, 2: z).

        Returns:
            tuple: (normalized_Efields, normalized_Hfields)
                - normalized_Efields (ndarray): Normalized electric fields.
                - normalized_Hfields (ndarray): Normalized magnetic fields.
        """    
        x = self.compute_differences(x)
        y = self.compute_differences(y)

        weight_mask = np.outer(x, y)
        weight_mask_expanded = weight_mask[np.newaxis,np.newaxis,:,:]

        # integrand
        integrand = np.cross(Efield, Hfield, axis=2)
        integrand = integrand[:,:,prop_axis]
        
        # integral to get normalization constant
        normalization_constants = (weight_mask_expanded*integrand/2).sum(axis=(2,3))
        normalization_constants = np.sqrt(normalization_constants)
        normalization_constants_expanded = normalization_constants[:,:,np.newaxis, np.newaxis, np.newaxis]

        # normalize fields
        normalized_Efields = Efield/normalization_constants_expanded
        normalized_Hfields = Hfield/normalization_constants_expanded

        return normalized_Efields, normalized_Hfields
    
    @staticmethod
    def compute_differences(arr):
        # Initialize an empty array with the same length as the input array
        output = np.empty_like(arr)
        # Calculate the difference between adjacent elements and store in the output array
        output[:-1] = arr[1:] - arr[:-1]
        # Set the last component as the prior component
        output[-1] = output[-2]
        
        return output
    #endregion