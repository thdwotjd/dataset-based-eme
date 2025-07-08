import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt


from .. import matrix_calculation_tool as mct
from ..propagator.multi_propagator.multi_eme import MultiEME
from ..propagator.multi_propagator.multi_propagator import MultiPropagator


class MultiRunner():
    def __init__(self, propagator_instance: MultiPropagator, input_amplitudes:list = None):
        """
        propagator_instance: MultiEME or MultiCLMT
        """
        if not issubclass(type(propagator_instance), MultiPropagator):
            print("propagator_instance should be instance of the subclass of the MultiPropagator")
            return 0
        
        self._propagator_instance = propagator_instance
        propagators = propagator_instance.propagators   # list of propagators


        if not self._propagator_instance._is_tmatrix_calculated:
            self._propagator_instance.calc_Tmatrix()

        if not self._propagator_instance._is_smatrix_calculated:
            self._propagator_instance.calc_Smatrix()

        if not self._propagator_instance._is_lengths_per_matrices_calculated:
            self._propagator_instance._find_lengths_per_matrices()
        
        if not self._propagator_instance._is_smatrix_merged:
            self._propagator_instance._merge_result_data()
        
        ## initialize runners
        runners = []
        from .single_runner import SingleRunner
        for propagator in propagators:
            runner = SingleRunner(propagator)
            runners.append(runner)

        ## multirunner parameter
        self.runners = runners
        self._merged_smatrix = self._propagator_instance._merged_smatrix
        # list of ndarray (each ndarray is expanded dimension radiation mode mask)
        self._radiation_mode_mask = self._propagator_instance._radiation_mode_masks
        self.input_amplitudes = None
        self.sectional_amplitudes_each_geometry = None
        self.total_sectional_amplitudes = None

        # status
        self._is_propagated = False
    
    #region main functions

    def propagate(self, input_amplitudes: list):
        input_amplitudes = np.array(input_amplitudes)
        # validity of input only depends on the first structure
        if not self.runners[0]._is_input_amplitudes_okay(input_amplitudes): return 
        input_amplitudes = self._convert_input(input_amplitudes)
        self.input_amplitudes = input_amplitudes

        mode_count = int(len(input_amplitudes)/2)
        length = self._merged_smatrix.shape[0]

        sectional_amplitudes = np.zeros((length + 1, 2*mode_count), dtype = np.complex64)
        sectional_amplitudes[0] = input_amplitudes

        smatrix = deepcopy(self._merged_smatrix)
        accumulated_smatrix = np.eye(2*mode_count)
        for i in range(length):
            accumulated_smatrix = mct._redheffer_star_product(accumulated_smatrix, smatrix[i])
            sectional_amplitudes[i+1] = accumulated_smatrix @ sectional_amplitudes[0]
        self.total_sectional_amplitudes = sectional_amplitudes

        self._is_propagated = True


    def plot_complexplane(self):
        # if self.sectional_amplitudes_each_geometry == None:
        if not self._is_propagated:
            print("Propagate the simulation first. Use propagate(input_amplitudes) method")
            return
        for runner in self.runners:
            runner.plot_complexplane()
    
    def plot_intensity_along_propagation(self, show_radiation_mode = False):
        # if self.sectional_amplitudes_each_geometry == None:
        if not self._is_propagated:
            print("Propagate the simulation first. Use propagate(input_amplitudes) method")
            return
        
        lengths = deepcopy(self._propagator_instance._lengths_per_matrices)
        num_geometries = len(lengths)
        cum_lengths = []
        cum_lengths_um = []
        for i in range(num_geometries):
            if i == 0: cum_lengths.append(np.insert(np.cumsum(lengths[i]), 0, [0]))
            else: cum_lengths.append(np.insert(np.cumsum(lengths[i]), 0, [0]) + cum_lengths[i-1][-1])
            # else: cum_lengths.append(np.cumsum(lengths[i]) + cum_lengths[i-1][-1])
            cum_lengths_um.append(deepcopy(cum_lengths[i] * 1e6))
        

        input_amplitudes = deepcopy(self.input_amplitudes)
        input_intensities = np.abs(input_amplitudes)**2
        standard_intensity = input_intensities.max()

        total_sectional_amplitudes = deepcopy(self.total_sectional_amplitudes)
        total_sectional_amplitudes /= standard_intensity

        mode_count = int(len(input_intensities)/2)
        concatenated_cum_lengths_um = np.concatenate(cum_lengths_um)
        for i in range(mode_count):
            mode_name = "mode " + str(i)
            plt.plot(concatenated_cum_lengths_um, total_sectional_amplitudes[:,i], label = mode_name)

        plt.xlabel("Propagation Length [$\mu$m]")
        plt.ylabel("Normalized Intensity")
        plt.legend()
        plt.title("Variation of Intensity Inside the Structure")

    
    def propagate_lumped_smatrix(self, input_amplitudes: list):
        print("This function is available only in single runner.")
        return

    #endregion main functions

    #region subtask functions (_convert_input)

    def _convert_input(self, input_amplitudes):
        input_amplitudes = self.runners[0]._convert_input(input_amplitudes)
        # expand the input amplitudes to expanded dimension
        mode_num = int(len(input_amplitudes)/2)
        expanded_mode_num = int(self._merged_smatrix.shape[1]/2)
        
        expanded_input = np.zeros(shape=(2*expanded_mode_num,), dtype=np.complex64)
        expanded_input[:mode_num] = input_amplitudes[:mode_num]
        expanded_input[expanded_mode_num:expanded_mode_num+mode_num] = input_amplitudes[mode_num:]

        return expanded_input
    
    #endregion subtask functions (_convert_input)

