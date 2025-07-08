import sys
import pickle
from itertools import product
from copy import deepcopy
import gc
import os
import yaml

import numpy as np
from tqdm import tqdm

from . import overlap_calculation_tool as oct

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(project_root, 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# import Lumerical
is_import_api = False

path_name_start = config["lumerical_path"]
path_name_start2 = config["ansys_path"]
path_name_end = config["api_path"]
path_name_end2 = config["ansys_api_path"]
version = config["versions"]


for i in range(len(version)):
    try:
        sys.path.append(path_name_start + version[i] + path_name_end)
        import lumapi
        is_import_api = True
        break
    except:
        if i == len(version)-1:
            print(
                "failed to load lumerical api (Dataset acquisition) "
                "(Lumerical Version)"
            )
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
                print(
                    "failed to load lumerical api (Dataset acquisition)"
                    "(Ansys version)"
                )
        continue


# ray.init(object_store_memory=2e9, ignore_reinit_error=True)
class TestModeError(Exception):
    pass

class DataUpdater:
    """
    Dataset dictionary structures:
        - overlap: {
            keys: parameter point (tuple ordered by ordering of datainfo.parameter_names)
            values: dictionary 
                {
                key: adj parameter point, 
                value : ndarray of <E(prameter point),H(adj_point)> (ndarray with shape (mode_numbers, mode_numbers))
                }
            }
        - neff: {keys: parameter point, value: neffs of ndarray with shape (mode_numbers,)}
        - TE_pol: {keys: parameter point, value: TE_plaraization fraction of ndarray with shape (mode_numbers,)}
    """
    def __init__(self, data_directory, is_testmode = False):
        self.data_directory = data_directory
        self._is_testmode = is_testmode
        
        try:
            sys.path.append(data_directory)
            from dataset_info import DatasetInfo
            self.data_info = DatasetInfo()
        except:
            print("Failed to import Dataset")
            return
        
        file_names = config["file_names"]
        self.FDE_file = os.path.join(data_directory, file_names["fde"])
        self.overlap_filename = os.path.join(data_directory, file_names["overlap"])
        self.neff_filename = os.path.join(data_directory, file_names["neff"])
        self.TE_pol_filename = os.path.join(data_directory, file_names["te_pol"])
        # self.FDE_file = data_directory + "/wg_crosssection.lms"
        # self.overlap_filename = data_directory + "/overlap.pkl"
        # self.neff_filename = data_directory + "/neff.pkl"
        # self.TE_pol_filename = data_directory + "/TE_pol.pkl"

        if not is_testmode:
            self.mode = lumapi.MODE(filename = self.FDE_file)

        self.parameter_names = self.get_parameter_names()
        self.parameter_types = self.get_parameter_types()
        self.parameter_grid = self.get_parameter_grid()
        self.mode_numbers = self.get_mode_numbers()
        self.crosssection_x = self.get_crosssection_x()
        self.crosssection_y = self.get_crosssection_y()
        self.wavelength = self.get_wavelength()
        # self._is_variable_FDE = self.get_if_varaible_FDE()

        self.verify_data()

        self.overlap = self.load_dict_from_pickle(self.overlap_filename)
        self.neff = self.load_dict_from_pickle(self.neff_filename)
        self.TE_pol = self.load_dict_from_pickle(self.TE_pol_filename)

    def update_dataset(self, parameter_names, parameter_values):
        """
        parameter_names: list of str
        parameter_values: list of list which min, max values of the parameter
            e.g) [[0, 50000], [1e-6, 1.5e-6]]
        """
        parameter_names_ref = self.get_parameter_names()
        parameter_grid = self.get_parameter_grid()

        # Validity check
        ## Check len of parameter names and values
        if len(parameter_names) != len(parameter_values):
            print("Number of values and parameter names are different")
            return

        ## Check if parameter name is valid
        param_dict = dict()
        for param_name, parameter_value in zip(parameter_names, parameter_values):
            if param_name not in parameter_names_ref:
                print(param_name, " is not in parameters")
                print("Parameter names are ", *parameter_names_ref)
                return
            param_dict[param_name] = [parameter_value]

        # Find nearest parameter value
        nearest_values = []
        for parameter_name, parameter_value in zip(parameter_names, parameter_values):
            reference = deepcopy(parameter_grid[parameter_name])
            nearest_value = reference[np.argmin(np.abs(reference - parameter_value))]
            nearest_values.append(nearest_value)
            print("nearest point: ", parameter_name, parameter_value)

        # Extract proper values of tuples into list
        ## Find input parameter order in the tuple
        parameter_indices = []
        for parameter_name in parameter_names:
            parameter_indices.append(parameter_names_ref.index(parameter_name))

        ## Initiate tuple list
        param_list = []
        for param_name in parameter_names_ref:
            if param_name not in parameter_names:
                param_list.append(parameter_grid[param_name])
            else:
                param_list.append(param_dict[param_name])

        updating_parameters = list(product(*param_list))

        # Update parameters
        num_point = 0
        for point in tqdm(updating_parameters):
            if not self.check_data_point_in_dataset(point):
                self.calc_data_point(point)
                self.save_data()
                num_point += 1
        self.save_data()
        print("Update", num_point, "Parameters")

        return
    
    def populate_dataframe(self, parameter_names, parameter_ranges):
        """
        parameter_names: list of str
        parameter_ranges: list of list which min, max values of the parameter
            e.g) [[0, 50000], [1e-6, 1.5e-6]]
        """
        parameter_names_ref = self.get_parameter_names()
        parameter_grid = self.get_parameter_grid()

        # Validity check
        ## Check len of parameter names and values
        if len(parameter_names) != len(parameter_ranges):
            print("Number of values and parameter names are different")
            return
        ## Check if the parameter ranges
        for parameter_range in parameter_ranges:
            if len(parameter_range) != 2:
                print("Each range of parmeter should be two values. (min & max)")
                print(parameter_range)
                return
            if parameter_range[0] > parameter_range[1]:
                print("The order of the range should be min, max")
                return
        ## Check if parameter name is valid
        param_dict = dict()
        for param_name, parameter_range in zip(parameter_names, parameter_ranges):
            if param_name not in parameter_names_ref:
                print(param_name, " is not in parameters")
                print("Parameter names are ", *parameter_names_ref)
                return
        
        # add missing parameter dimension (with full parameters)
        missing_params = list(set(parameter_names_ref) - set(parameter_names))
        if len(missing_params): # if there is any missing parmaeters
            for missing_param in missing_params:
                parameter_names.append(missing_param)

                missing_param_values = deepcopy(parameter_grid[missing_param])
                parameter_ranges.append([missing_param_values[0], missing_param_values[-1]])

        # reorder param_names & param_values according to the dataset (parameter_names_ref)
        parameter_ranges_reorder= []
        for param_name in parameter_names_ref:
            param_index = index = parameter_names.index(param_name)
            parameter_ranges_reorder.append(parameter_ranges[param_index])


        # mapping parmater range to values
        value_list_from_range = []
        for parameter_name, parameter_range in zip(parameter_names_ref, parameter_ranges_reorder):
            reference = deepcopy(parameter_grid[parameter_name])
            num_trial_points = 3 * len(reference)   # this number can be adjusted it it does not work well
            trial_value_list = np.linspace(parameter_range[0], parameter_range[1], num_trial_points)
            nearest_values = []
            for trial_value in trial_value_list:
                nearest_values.append(reference[np.argmin(np.abs(reference - trial_value))])
            value_list_from_range.append(np.unique(nearest_values))

        #generate list of tuples (product of dimensions)
        updating_parameters = list(product(*value_list_from_range))

        # Update parameters
        for parameter_name, parameter_range in zip(parameter_names_ref, parameter_ranges_reorder):
            print("Parameter name: ", parameter_name, "&  Parameter ranges: ", parameter_range)
        # print("Sample parameter point : ", updating_parameters[0],  )
        if len(updating_parameters)>3:
            print("Sample parameter point : {}, {}, ... , {}".format(updating_parameters[0], updating_parameters[1], updating_parameters[-1]))
        else:
            print("Sample parameter point : {}, {}".format(updating_parameters[0], updating_parameters[-1]))

        print("The total number of points in the given range : ", len(updating_parameters))
        num_to_update = 0

        parameters_to_process = []
        for point in updating_parameters:
            # if not self.check_data_point_in_dataset(point):
            if not self.deepcheck_data_point_in_dataset(point):
                num_to_update += 1
                parameters_to_process.append(point)
        print("The total number of points to be updated : ", num_to_update)

        user_input = input("Do you want to continue? (y/n): ").strip().lower()
        if user_input == 'y':
            print("Executing the function...")
            # print(f"Function executed with arguments: x={x}, y={y}")
        elif user_input == 'n':
            print("Skipping the function...")
            return
        else:
            print("Invalid input. Proceeding without execution.")
            return

        num_point = 0
        for point in tqdm(parameters_to_process):
            # if not self.check_data_point_in_dataset(point):
            if not self.deepcheck_data_point_in_dataset(point):
                self.calc_data_point_modified(point)
                self.save_data()
                num_point += 1
        self.save_data()
        print("Update", num_point, "Parameters")


    def save_data(self):
        if self._is_testmode:
            return
        self.save_dict_to_pickle(self.overlap, self.overlap_filename)
        self.save_dict_to_pickle(self.neff, self.neff_filename)
        self.save_dict_to_pickle(self.TE_pol, self.TE_pol_filename)
        pass

    def load_data(self):
        self.overlap = self.load_dict_from_pickle(self.overlap_filename)
        self.neff = self.load_dict_from_pickle(self.neff_filename)
        self.TE_pol = self.load_dict_from_pickle(self.TE_pol_filename)

    def check_data_point_in_dataset(self, param_point):
        test1 = (param_point in self.overlap)
        test2 = (param_point in self.neff)
        test3 = (param_point in self.TE_pol)
        return (test1 and  test2 and test3)
    
    def deepcheck_data_point_in_dataset(self, param_point):
        test123 = self.check_data_point_in_dataset(param_point)
        adj_points = self.get_adjacent_points(param_point)
        adj_points = self.filter_out_calculated_points(param_point, adj_points)
        test4 = not len(adj_points)
        return (test123 and test4)

    def verify_data(self):
        param_grid = self.get_parameter_grid()
        param_types = self.get_parameter_types()
        for param_name in self.parameter_names:
            if param_name in param_grid and param_name in param_types:
                continue
            print("Dataset parameter names and parameter gird are not matched")
            return
        
        # check parameters in lms files
        pass

    def calc_data_point(self, parameter_point):
        print("calc_data_point() is deprecated")
        adj_points = self.get_adjacent_points(parameter_point)
        self.set_FDE_sweep(parameter_point, adj_points)
        self.run_FDE_sweep()
        neff, TE_pol, overlaps = self.post_process_sweep_data(adj_points)

        self.overlap[parameter_point] = overlaps
        self.neff[parameter_point] = neff
        self.TE_pol[parameter_point] = TE_pol
    
    def calc_data_point_modified(self, parameter_point):
        is_calculated = 0
        adj_points = self.get_adjacent_points(parameter_point)
        adj_points = self.filter_out_calculated_points(parameter_point, adj_points)
        if len(adj_points) == 0:
            # print("All points are already calculated, parameter point: ", parameter_point)
            return is_calculated
        elif self._is_testmode:
            raise TestModeError("The dataset update was attempted in test mode")
        
        # if _is_variable_FDE:
        self.set_FDE_sweep_variable_FDE_size(parameter_point, adj_points)
        # else:
        # self.set_FDE_sweep_modified(parameter_point, adj_points)
        self.run_FDE_sweep()
        neff, TE_pol, overlaps = self.post_process_sweep_data_modified(parameter_point, adj_points)

        
        if not parameter_point in self.overlap:
            self.overlap[parameter_point] = dict()
            self.neff[parameter_point] = neff[parameter_point]
            self.TE_pol[parameter_point] = TE_pol[parameter_point]
        
        for adj_point in adj_points:
            self.overlap[parameter_point][adj_point] = overlaps[parameter_point][adj_point]
            if not adj_point in self.overlap:
                self.overlap[adj_point] = dict()
                self.neff[adj_point] = neff[adj_point]
                self.TE_pol[adj_point] = TE_pol[adj_point]
            self.overlap[adj_point][parameter_point] = overlaps[adj_point][parameter_point]
        
        is_calculated = 1
        return is_calculated

    def get_adjacent_points(self, parameter_point):
        adj_values = []
        # get adj parameters
        for i in range(len(self.parameter_names)):
            param_name = self.parameter_names[i]
            adj_value = []
            # if error occurs, point is not exact point int param grid (check round operation)
            point_index = np.where(self.parameter_grid[param_name] == parameter_point[i])[0][0]
            if not point_index == 0:
                adj_value.append(self.parameter_grid[param_name][point_index-1])
            if point_index < len(self.parameter_grid[param_name]) - 1:
                adj_value.append(self.parameter_grid[param_name][point_index+1])
            adj_values.append(adj_value)
        
        # get adj one param adj points
        adj_points = []
        for i in range(len(adj_values)):
            temp = list(parameter_point)
            for adj in adj_values[i]:
                temp[i] = adj
                adj_points.append(tuple(temp))

        #adj_points = list(product(*adj_values))
        return adj_points
    
    def filter_out_calculated_points(self, parameter_point, adj_points):
        filtered_adj_points = []
        if not parameter_point in self.overlap:
            return adj_points
        
        for adj_point in adj_points:
            if not adj_point in self.overlap[parameter_point]:
                filtered_adj_points.append(adj_point)
        return filtered_adj_points

    #region get simple data from dataset
    def get_parameter_grid(self):
        return self.data_info.get_parameter_grid()
    
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
    
    # def get_if_varaible_FDE(self):
    #     return self.data_info._is_variable_FDE()
    #endregion get simple data from dataset

    #region get data point from datset (overlap, neff, TE_pol)

    def get_overlap(self, pt1, pt2):
        """
        Parameters:
            -pt1, pt2: parameter point tuple
        Returns:
            overlap_ab (ndarray): <E_a, H_b> 
            overlap_ba (ndarray): <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        if pt1 == pt2:
            overlap_ab = self.same_mode_overlap_matrix()
            overlap_ba = self.same_mode_overlap_matrix()
            return overlap_ab, overlap_ba
        overlap_ab = deepcopy(self.overlap[pt1][pt2])
        overlap_ba = deepcopy(self.overlap[pt2][pt1])
        return overlap_ab, overlap_ba

    def get_overlaps(self, simul_params):
        """
        Parameters:
            -simul_params: list of tuple where each tuple is parameter point
        Returns:
            overlap_ab: <E_a, H_b>
            overlap_ba: <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        overlap_ab = np.zeros(shape=(len(simul_params)-1, 2*self.mode_numbers, 2*self.mode_numbers), dtype =np.complex64)
        overlap_ba = np.zeros(shape=(len(simul_params)-1, 2*self.mode_numbers, 2*self.mode_numbers), dtype =np.complex64)

        for i in range(len(simul_params)-1):
            if simul_params[i] == simul_params[i+1]:
                overlap_ab[i] = self.same_mode_overlap_matrix()
                overlap_ba[i] = self.same_mode_overlap_matrix()
                continue
            overlap_ab[i] = deepcopy(self.overlap[simul_params[i]][simul_params[i+1]])
            overlap_ba[i] = deepcopy(self.overlap[simul_params[i+1]][simul_params[i]])
        
        return overlap_ab, overlap_ba

    def get_neffs(self, simul_params):
        neffs = np.zeros(shape = (len(simul_params), 2*self.mode_numbers), dtype = np.complex64)
        for i in range(len(simul_params)):
            point = simul_params[i]
            neffs[i] = deepcopy(self.neff[point])
        return neffs

    def get_TE_pols(self, simul_params):
        TE_pols = np.zeros(shape = (len(simul_params), 2*self.mode_numbers), dtype =np.float16)
        for i in range(len(simul_params)):
            point = simul_params[i]
            TE_pols[i] = deepcopy(self.TE_pol[point])
        return TE_pols
    
    def get_overlaps_modified(self, simul_params, additional_param_dict):
        """
        Parameters:
            -simul_params: list of tuple where each tuple is parameter point
            -additional_param_dict: additional tuple parameter points to calcualte multi_adj_point coupling coeff
                keys: point (tuple)
                values: list of points
        Returns:
            overlap_ab (dict): <E_a, H_b>
            overlap_ba (dict): <E_b, H_a> where a is section nearer to the input port and b is section nearer to output port
        """
        overlap_ab = dict()
        overlap_ba = dict()

        for i in range(len(simul_params)-1):
            overlap_ab[simul_params[i]] = dict()
            overlap_ba[simul_params[i+1]] = dict()

        for i in range(len(simul_params)-1):
            if simul_params[i] == simul_params[i+1]:
                overlap_ab[simul_params[i]][simul_params[i+1]] = self.same_mode_overlap_matrix()
                overlap_ba[simul_params[i+1]][simul_params[i]] = self.same_mode_overlap_matrix()
                continue
            try:
                overlap_ab[simul_params[i]][simul_params[i+1]] = deepcopy(self.overlap[simul_params[i]][simul_params[i+1]])
                overlap_ba[simul_params[i+1]][simul_params[i]] = deepcopy(self.overlap[simul_params[i+1]][simul_params[i]])
            except: # multi adj pts (calculated at below for loop)
                pass
        
        for key in additional_param_dict.keys():    # multi adj pts
            for additional_pt in additional_param_dict[key]:
                if not additional_pt in overlap_ba[additional_pt]: overlap_ba[additional_pt] = dict()
                overlap_ab[key][additional_pt] = deepcopy(self.overlap_ab[key][additional_pt])
                overlap_ba[additional_pt][key] = deepcopy(self.overlap_ba[additional_pt][key])
        
        return overlap_ab, overlap_ba
    
    def get_neffs_modified(self, simul_params):
        neffs = dict()
        for point in simul_params:
            neffs[point] = deepcopy(self.neff[point])
        return neffs

    def get_TE_pols_modified(self, simul_params):
        TE_pols = dict()
        for point in simul_params:
            TE_pols[point] = deepcopy(self.TE_pol[point])
        return TE_pols
        
    
    def same_mode_overlap_matrix(self):
        overlap_matrix = np.zeros(shape=(2*self.mode_numbers, 2*self.mode_numbers))
        I = np.eye(self.mode_numbers)
        overlap_matrix[:self.mode_numbers,:self.mode_numbers] = deepcopy(I)
        overlap_matrix[:self.mode_numbers,self.mode_numbers:] = -deepcopy(I)
        overlap_matrix[self.mode_numbers:,:self.mode_numbers] = deepcopy(I)
        overlap_matrix[self.mode_numbers:,self.mode_numbers:] = -deepcopy(I)
        return overlap_matrix

    #endregion get data point from datset (overlap, neff, TE_pol)

    #region load and save pickle file
    def save_dict_to_pickle(self, data, filename):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, 'wb') as file:
                pickle.dump(data, file)
        except Exception as e:
            print(f"Failed to save data to {filename}: {str(e)}")


    def load_dict_from_pickle(self, filename):
        try: # if file exists
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except: # if file does not exists, generate and load file
            with open(filename, 'wb') as file:
                data = dict()
                pickle.dump(data, file)
            with open(filename, 'rb') as file:
                return pickle.load(file)
    #endregion load and save pickle file      

    #region FDE using functions. (set, run, getsweepresult, calc_overlaps, find_mode_fields)
    def set_FDE_sweep(self, parameter_point, adj_points):
        """
        Parameters:
            - parameter_point: tuple, parameter values are ordered by self.parameter_names
            - adjpoints: list of tuple, each tuple's parameter values are ordered by self.parameter_names
        """
        # set sweep name
        self.mode.deletesweep("sweep")
        self.mode.addsweep(0)

        # set sweep parameters
        adj_num_points = len(adj_points)
        self.mode.setsweep("sweep", "type","values")
        self.mode.setsweep("sweep", "number of points", adj_num_points+1)
        
        for i in range(len(self.parameter_names)):
            parameters = dict()
            param_name = self.parameter_names[i]
            parameters["Name"] = param_name
            parameters["Parameter"] = "::model::" + param_name
            parameters["Type"] = self.parameter_types[param_name]

            parameters["value_1"] = parameter_point[i]
            for j in range(adj_num_points):
                parameters["value_" + str(j+2)] = adj_points[j][i]
            
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

    def set_FDE_sweep_modified(self, parameter_point, adj_points):
        """
        Parameters:
            - parameter_point: tuple, parameter values are ordered by self.parameter_names
            - adjpoints: list of tuple, each tuple's parameter values are ordered by self.parameter_names
        """
        # set sweep name
        self.mode.deletesweep("sweep")
        self.mode.addsweep(0)

        # set sweep parameters
        adj_num_points = len(adj_points)
        self.mode.setsweep("sweep", "type","values")
        self.mode.setsweep("sweep", "number of points", adj_num_points+1)
        
        for i in range(len(self.parameter_names)):
            parameters = dict()
            param_name = self.parameter_names[i]
            parameters["Name"] = param_name
            parameters["Parameter"] = "::model::" + param_name
            parameters["Type"] = self.parameter_types[param_name]

            parameters["value_1"] = parameter_point[i]
            for j in range(adj_num_points):
                parameters["value_" + str(j+2)] = adj_points[j][i]
            
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

    def set_FDE_sweep_variable_FDE_size(self, parameter_point, adj_points):
        """
        Parameters:
            - parameter_point: tuple, parameter values are ordered by self.parameter_names
            - adjpoints: list of tuple, each tuple's parameter values are ordered by self.parameter_names
        """
        # set sweep name
        self.mode.deletesweep("sweep")
        self.mode.addsweep(0)

        # set sweep parameters
        adj_num_points = len(adj_points)
        self.mode.setsweep("sweep", "type","values")
        self.mode.setsweep("sweep", "number of points", adj_num_points+1)
        
        for i in range(len(self.parameter_names)):
            parameters = dict()
            param_name = self.parameter_names[i]
            parameters["Name"] = param_name
            parameters["Parameter"] = "::model::" + param_name
            parameters["Type"] = self.parameter_types[param_name]

            parameters["value_1"] = parameter_point[i]

            for j in range(adj_num_points):
                parameters["value_" + str(j+2)] = adj_points[j][i]
            
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


    def run_FDE_sweep(self):
        self.mode.runsweep("sweep")

    def post_process_sweep_data(self, adj_points):
        # extract cross section array
        x = self.mode.getsweepresult("sweep", "x")[self.crosssection_x][:,0]
        y = self.mode.getsweepresult("sweep", "y")[self.crosssection_y][:,0]

        # result ndarray initialization
        neff = np.zeros(shape = (2*self.mode_numbers,), dtype=np.complex64)
        TE_pol = np.zeros(shape = (2*self.mode_numbers,), dtype=np.float16)
        Efield = np.zeros(shape = (len(adj_points) + 1, 2*self.mode_numbers, 3, len(x), len(y)), dtype=np.complex64)
        Hfield = np.zeros(shape = (len(adj_points) + 1, 2*self.mode_numbers, 3, len(x), len(y)), dtype=np.complex64)
        overlaps = dict()

        # extract sweep data
        for i in range(self.mode_numbers):
            # forward propagating mode
            neff[i] = oct.correct_gain(np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_neff")["neff"])[0]) # first element is parameter point value
            TE_pol[i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_TEpol")["TE polarization fraction"])[0]
            Efield[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_E")["E"]).transpose(2,3,0,1)   # (len_x, len_y, num_points, 3) -> (num_points, 3, len_x, len_y)
            Hfield[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_H")["H"]).transpose(2,3,0,1)
        
        ## overlaps
        prop_axis = list({"x", "y", "z"} - {self.crosssection_x, self.crosssection_y})[0]
        prop_axis = 0 if prop_axis == 'x' else 1 if prop_axis == 'y' else 2
        # normalize field data
        Efield, Hfield = oct.normalize_field(Efield, Hfield, x, y, prop_axis)

        for i in range(self.mode_numbers):
            # backward propagating mode
            neff[i+self.mode_numbers] = -deepcopy(neff[i])
            TE_pol[i+self.mode_numbers] = deepcopy(TE_pol[i])
            # Since, overlap concerns the transverse field of the field, the following is valid for reciprocal medium
            Efield[:,i+self.mode_numbers] = np.conjugate(deepcopy(Efield[:,i]))
            Hfield[:,i+self.mode_numbers] = -np.conjugate(deepcopy(Hfield[:,i]))

        # calculate overlaps
        for i in range(len(adj_points)):
            # Efield[0]: parameter point, Efield[1:] : adj points
            overlaps[adj_points[i]] = self.calc_overlap(Efield[0], Hfield[i+1], x, y, prop_axis)
        
        return neff, TE_pol, overlaps
    

    def post_process_sweep_data_modified(self, parameter_point, adj_points):
        # extract cross section array
        x = self.mode.getsweepresult("sweep", "x")[self.crosssection_x][:,0]
        y = self.mode.getsweepresult("sweep", "y")[self.crosssection_y][:,0]

        # result ndarray initialization
        neff = np.zeros(shape = (len(adj_points) + 1, 2*self.mode_numbers,), dtype=np.complex64)
        TE_pol = np.zeros(shape = (len(adj_points) + 1, 2*self.mode_numbers,), dtype=np.float16)
        Efield = np.zeros(shape = (len(adj_points) + 1, 2*self.mode_numbers, 3, len(x), len(y)), dtype=np.complex64)
        Hfield = np.zeros(shape = (len(adj_points) + 1, 2*self.mode_numbers, 3, len(x), len(y)), dtype=np.complex64)
        overlaps = dict()

        # extract sweep data
        for i in range(self.mode_numbers):
            # forward propagating mode
            neff[:,i] = oct.correct_gain_modified(np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_neff")["neff"])) # first element is parameter point value
            TE_pol[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_TEpol")["TE polarization fraction"])
            Efield[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_E")["E"]).transpose(2,3,0,1)   # (len_x, len_y, num_points, 3) -> (num_points, 3, len_x, len_y)
            Hfield[:,i] = np.squeeze(self.mode.getsweepresult("sweep", "mode" + str(i+1) + "_H")["H"]).transpose(2,3,0,1)
        
        ## overlaps
        prop_axis = list({"x", "y", "z"} - {self.crosssection_x, self.crosssection_y})[0]
        prop_axis = 0 if prop_axis == 'x' else 1 if prop_axis == 'y' else 2
        # normalize field data
        Efield, Hfield = oct.normalize_field(Efield, Hfield, x, y, prop_axis)

        for i in range(self.mode_numbers):
            # backward propagating mode
            neff[:,i+self.mode_numbers] = -deepcopy(neff[:,i])
            TE_pol[:,i+self.mode_numbers] = deepcopy(TE_pol[:,i])
            # Since, overlap concerns the transverse field of the field, the following is valid for reciprocal medium
            Efield[:,i+self.mode_numbers] = np.conjugate(deepcopy(Efield[:,i]))
            Hfield[:,i+self.mode_numbers] = -np.conjugate(deepcopy(Hfield[:,i]))

        overlaps[parameter_point] = dict()
        neff_dict = dict()
        TE_pol_dict = dict()
        neff_dict[parameter_point] = neff[0]
        TE_pol_dict[parameter_point] = TE_pol[0]
        # calculate overlaps & neff, TE_pol
        for i in range(len(adj_points)):
            overlaps[adj_points[i]] = dict()
            # Efield[0]: parameter point, Efield[1:] : adj points
            overlaps[parameter_point][adj_points[i]] = self.calc_overlap(Efield[0], Hfield[i+1], x, y, prop_axis)
            overlaps[adj_points[i]][parameter_point] = self.calc_overlap(Efield[i+1], Hfield[0], x, y, prop_axis)
            # neff, TE_pol
            neff_dict[adj_points[i]] = neff[i+1]
            TE_pol_dict[adj_points[i]] = TE_pol[i+1]
        
        return neff_dict, TE_pol_dict, overlaps
    
    def find_mode_fields(self, parameter_point):
        """
        Parameters:
            - parameter_point: tuple, parameter values are ordered by self.parameter_names
        Returns:
            -Efields: ndarray of shape (mode_nums, len_x, len_y, 3)
            -Hfields: ndarray of shape (mode_nums, len_x, len_y, 3)
            -x, y : 1darray
        """
        # set FDE
        mode_number = self.get_mode_numbers()
        self.mode.switchtolayout()
        self.mode.select("::model")
        for i in range(len(self.parameter_names)):
            param_name = self.parameter_names[i]
            self.mode.set(param_name, parameter_point[i])

        # run FDE
        self.mode.findmodes()

        # extract field dtat from FDE
        x = np.squeeze(self.mode.getdata("FDE::data::material", self.crosssection_x))
        y = np.squeeze(self.mode.getdata("FDE::data::material", self.crosssection_y))
        Efields = np.zeros(shape = (mode_number, len(x), len(y), 3), dtype=np.complex64)
        Hfields = np.zeros(shape = (mode_number, len(x), len(y), 3), dtype=np.complex64)
        for i in range(mode_number):
            Efields[i] = np.squeeze(self.mode.getresult("FDE::data::mode" + str(i+1), "E")["E"]) # shape (len_x, len_y, 3)
            Hfields[i] = np.squeeze(self.mode.getresult("FDE::data::mode" + str(i+1), "H")["H"])
        
        return Efields, Hfields, x, y
    
    #endregion FDE using functions. (set, run, getsweepresult, calc_overlaps, find_mode_fields)

    
    #region old vers
    @staticmethod
    def calc_overlap(Efield, Hfield, x, y, prop_axis):
        def compute_differences(arr):
            # Initialize an empty array with the same length as the input array
            output = np.empty_like(arr)
            # Calculate the difference between adjacent elements and store in the output array
            output[:-1] = arr[1:] - arr[:-1]
            # Set the last component as the prior component
            output[-1] = output[-2]
            
            return output
        # Add new axes for broadcasting
        Efield_expanded = Efield[:, np.newaxis, :, :, :]
        Hfield_expanded = Hfield[np.newaxis, :, :, :, :]
        
        # Compute cross products using broadcasting
        cross_product = np.cross(Efield_expanded, Hfield_expanded, axis=2)

        # del Efield_expanded
        del Hfield_expanded
        gc.collect()

        x = compute_differences(x)
        y = compute_differences(y)

        weight_mask = np.outer(x, y)

        # The shape of weight_mask should be the same as func
        weight_mask_expanded = weight_mask[np.newaxis, np.newaxis,np.newaxis,:,:]
        
        overlap = (weight_mask_expanded * cross_product).sum(axis=(3,4))
        overlap = overlap[:,:,prop_axis]/2
        return overlap
    #endregion oldvers

    #endregion calculate overlaps

    #region test functions
    def _extract_field_test(self, mode_number = 0):
        """
        it assumes that the FDE is already ran. It only extract the data for test
        """
        if mode_number == 0: mode_number = self.get_mode_numbers()
        x = np.squeeze(self.mode.getdata("FDE::data::material", self.crosssection_x))
        y = np.squeeze(self.mode.getdata("FDE::data::material", self.crosssection_y))
        Efields = np.zeros(shape = (mode_number, len(x), len(y), 3), dtype=np.complex64)
        Hfields = np.zeros(shape = (mode_number, len(x), len(y), 3), dtype=np.complex64)
        for i in range(mode_number):
            Efields[i] = np.squeeze(self.mode.getresult("FDE::data::mode" + str(i+1), "E")["E"]) # shape (len_x, len_y, 3)
            Hfields[i] = np.squeeze(self.mode.getresult("FDE::data::mode" + str(i+1), "H")["H"])
        
        return Efields, Hfields, x, y
    #endregion testfunctions