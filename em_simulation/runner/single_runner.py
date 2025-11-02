import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from tqdm import tqdm


from .. import matrix_calculation_tool as mct
from ..propagator.propagator import Propagator
from ..propagator.single_propagator.single_eme import SingleEME



class SingleRunner():
    def __init__(self, propagator_instance: Propagator, input_amplitudes:list = None):
        if not issubclass(type(propagator_instance), Propagator):
            print("propagator_instance should be instance of the subclass of the Propagator")
            return 0
        
        self._radiation_mode_mask = propagator_instance.output_data["radiation_mode_mask"]
        if not propagator_instance._is_smatrix_calculated:
            propagator_instance.calc_Smatrix()
        self._propagator_instance = propagator_instance
        self._smatrix = propagator_instance.smatrix
        self._lumped_smatrix = None
        self._mode_count = int(self._smatrix.shape[1]/2)
        self.sectional_amplitudes = None    # numpy array with shape (length + 1, 2*self._mode_count)

        # sweep parameters
        self._sweep_lengths = None  # 1darray of shape (sweep_points,)
        self._output_intensities_swept = None   # ndarray of shape (sweep_points, 2*self._mode_count)

        # status
        self._is_sectional_amplitudes_calculated = False
        self._is_output_intensities_swept_calculated = False
        self._is_lumped_smatrix_calculated = False

        
    #region main functions
    def propagate(self, input_amplitudes: list):
        """Propagate through each segment and record sectional amplitudes.

        :param input_amplitudes: Complex amplitudes for the launched modes.
        :type input_amplitudes: list
        :returns: Array of forward/backward amplitudes at each section.
        :rtype: np.ndarray
        """
        input_amplitudes = np.array(input_amplitudes)
        if not self._is_input_amplitudes_okay(input_amplitudes): return
        input_amplitudes = self._convert_input(input_amplitudes)
        self.input_amplitudes = input_amplitudes
        
            
        length = self._smatrix.shape[0]
        sectional_amplitudes = np.zeros((length + 1, 2*self._mode_count), dtype = np.complex64)
        sectional_amplitudes[0,:2*self._mode_count] = input_amplitudes

        smatrix = deepcopy(self._smatrix)
        accumulated_smatrix = np.eye(2*self._mode_count)
        for i in range(length):
            accumulated_smatrix = mct._redheffer_star_product(accumulated_smatrix, smatrix[i])
            sectional_amplitudes[i+1] = accumulated_smatrix @ sectional_amplitudes[0]
        
        self.sectional_amplitudes = deepcopy(sectional_amplitudes)
        self._is_sectional_amplitudes_calculated = True
        return sectional_amplitudes
    
    
    def plot_intensity_along_propagation(self, show_radiation_mode = False, mode_nums = None):
        """Plot normalized intensities along the propagation path.

        :param show_radiation_mode: Include radiation modes when ``True``.
        :type show_radiation_mode: bool
        :param mode_nums: Optional subset of mode indices to display.
        :type mode_nums: list or None
        :returns: Dictionary with propagation lengths and intensity traces.
        :rtype: dict
        """
        if not self._is_sectional_amplitudes_calculated:
            print("Propagate the simulation first. Use propagate(input_amplitudes) method")
            return
        lengths = deepcopy(self._propagator_instance._lengths_per_matrix)
        cum_lengths = np.insert(np.cumsum(lengths), 0, [0])

        input_amplitudes = deepcopy(self.input_amplitudes)
        input_intensities = np.abs(input_amplitudes)**2
        standard_intensity = input_intensities.max()

        # only forward modes are considered
        sectional_amplitudes= deepcopy(self.sectional_amplitudes[:,:self._mode_count])
        sectional_intensities = np.abs(sectional_amplitudes)**2
        normalized_sectional_intensities = sectional_intensities/standard_intensity

        radiation_mode_mask = self._radiation_mode_mask[:,:self._mode_count]
        if issubclass(type(self._propagator_instance), SingleEME):
            normalized_sectional_intensities = np.insert(normalized_sectional_intensities[1::2], 0, normalized_sectional_intensities[0], axis=0)
            cum_lengths = np.insert(cum_lengths[1::2], 0, cum_lengths[0])

        if not show_radiation_mode:
            normalized_sectional_intensities = np.where(radiation_mode_mask, np.nan, normalized_sectional_intensities)

        cum_lengths_um = cum_lengths * 1e6

        fig = plt.figure()
        ax = plt.subplot(111)
        return_values = dict()
        
        return_values["prop_lengths"] = cum_lengths_um
        if mode_nums:
            for mode_num in mode_nums:
                mode_name = "mode" + str(mode_num)
                return_values[mode_name] = normalized_sectional_intensities[:,mode_num-1]
                line, = ax.plot(cum_lengths_um, normalized_sectional_intensities[:,mode_num-1], label = mode_name)
        else:
            for i in range(self._mode_count):
                mode_name = "mode" + str(i+1)
                line, = ax.plot(cum_lengths_um, normalized_sectional_intensities[:,i], label = mode_name)
                return_values[mode_name] = normalized_sectional_intensities[:,i]

        box = ax.get_position()
        # Shrink current axis's height by 10% on the bottom
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=5)


        plt.xlabel("Propagation Length [$\mu$m]")
        plt.ylabel("Normalized Intensity")
        plt.title("Variation of Intensity Inside the Structure")
        
        return return_values
    

    def plot_complexplane(self, mode_nums = None):
        if not self._is_sectional_amplitudes_calculated:
            print("Propagate the simulation first. Use propagate(input_amplitudes) method")
            return

        return_values = dict()

        # x-y ticks
        t = np.linspace(0, np.pi*2, 1000)
        circle_x = np.cos(t)
        circle_y = np.sin(t)

        fig, ax = plt.subplots()

        ax.plot(circle_x, circle_y, '--', alpha=0.5, color='k')
        ax.set_aspect("equal")

        # Set axes to cross at (x=0, y=0)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')

        # Hide the other axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Set the direction of axis ticks
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xticks([-1,-0.5, 0.5, 1])
        ax.set_yticks([-1,-0.5, 0, 0.5, 1])
        ax.set_xlabel("Re(amp)")
        ax.set_ylabel("Im(amp)")

        ax.xaxis.set_label_coords(1.05, 0.5)
        ax.yaxis.set_label_coords(0.5, 1.05)

        if mode_nums == None:
            for i in range(self._mode_count):
                mode_name = "mode " + str(i+1)
                ax.plot(self.sectional_amplitudes[:,i].real, self.sectional_amplitudes[:,i].imag, label=mode_name)
                return_values[mode_name] = {"Re": self.sectional_amplitudes[:,i].real, "Imag":self.sectional_amplitudes[:,i].imag}
        else:
            for mode_num in mode_nums:
                mode_name = "mode " + str(mode_num)
                ax.plot(self.sectional_amplitudes[:,mode_num-1].real, self.sectional_amplitudes[:,mode_num-1].imag, label=mode_name)
                return_values[mode_name] = {"Re":self.sectional_amplitudes[:,mode_num-1].real, "Imag":self.sectional_amplitudes[:,mode_num-1].imag}
        ax.legend()
        return return_values
    
    def sweep_geometry_length(self, length_1, length_2, num_points, show_radiation_mode = False):
        """Sweep structure length and evaluate and plot output intensities.

        :param length_1: Starting structure length in meters.
        :type length_1: float
        :param length_2: Ending structure length in meters.
        :type length_2: float
        :param num_points: Number of sweep samples.
        :type num_points: int
        :param show_radiation_mode: Include radiation modes when ``True``.
        :type show_radiation_mode: bool
        :returns: Tuple of sweep lengths and intensities per mode.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        input_amplitudes = deepcopy(self.input_amplitudes)
        output_amplitudes_swept = np.zeros(shape=(num_points, 2*self._mode_count), dtype=np.complex64)

        sweep_lengths = np.linspace(length_1, length_2, num_points)
        Smatrices_swept = np.zeros(shape=(num_points, 2*self._mode_count, 2*self._mode_count), dtype=np.complex64)
        for i in tqdm(range(num_points)):
            smatrix = self._propagator_instance._find_Smatrix_new_length(sweep_lengths[i])
            smatrix_lumped = smatrix[0]
            for j in range(len(smatrix)-1):
                smatrix_lumped = mct._redheffer_star_product(smatrix_lumped, smatrix[j+1])           
            Smatrices_swept[i] = smatrix_lumped

        for i in range(num_points):
            output_amplitudes_swept[i] = Smatrices_swept[i] @ input_amplitudes
        
        output_intensities_swept = np.abs(output_amplitudes_swept)**2

        self._sweep_lengths = sweep_lengths
        self._output_intensities_swept = output_intensities_swept
        self._is_output_intensities_swept_calculated = True

        self.plot_sweep_data(show_radiation_mode = show_radiation_mode)

        return self._sweep_lengths, self._output_intensities_swept


    def plot_sweep_data(self, show_radiation_mode = False):
        """
        It plots swept output intensities
        """
        if not self._is_output_intensities_swept_calculated:
            print("Sweep the structure lengths first. Use 'sweep_geometry_length()' method")
            return

        if show_radiation_mode:
            for i in range(self._mode_count):
                mode_name = "mode" + str(i+1)
                plt.plot(self._sweep_lengths, self._output_intensities_swept[:,i], label = mode_name)
        else:
            radiation_mode_mask = self._radiation_mode_mask[-1]
            for i in range(self._mode_count):
                if radiation_mode_mask[i]: continue
                mode_name = "mode" + str(i+1)
                plt.plot(self._sweep_lengths, self._output_intensities_swept[:,i], label = mode_name)

        
        plt.xlabel("structure length")
        plt.ylabel("Normalized intensitiy")
        plt.legend()
        plt.title("Output Intensities vs. Structure Length")

    def propagate_lumped_smatrix(self, input_amplitudes: list):
        """Apply the lumped scattering matrix to the given input amplitudes.

        :param input_amplitudes: Complex amplitudes for the launched modes.
        :type input_amplitudes: list
        :returns: Output amplitudes after the lumped matrix.
        :rtype: np.ndarray
        """
        if not self._is_lumped_smatrix_calculated:
            smatrix = deepcopy(self._smatrix)
            length = self._smatrix.shape[0]
            accumulated_smatrix = np.eye(2*self._mode_count)
            for i in range(length):
                accumulated_smatrix = mct._redheffer_star_product(accumulated_smatrix, smatrix[i])
            self._lumped_smatrix = accumulated_smatrix
            self._is_lumped_smatrix_calculated = True
        
        input_amplitudes = np.array(input_amplitudes)
        if not self._is_input_amplitudes_okay(input_amplitudes): return
        input_amplitudes = self._convert_input(input_amplitudes)

        result = self._lumped_smatrix @ input_amplitudes
        return result
    
    #endregion main functions
    

    #region subtask functions (_is_input_amplitudes_okay, _convert_input)

    def _is_input_amplitudes_okay(self, input_amplitudes):
        """
        Paramters:
            input_amplitudes: numpy ndarray
        It checks the input amplitudes shape
        """
        initial_mode_mask = self._radiation_mode_mask[0]
        non_rad_count = np.count_nonzero(initial_mode_mask == False)

        if input_amplitudes.shape == (non_rad_count,):
            return 1
        else:
            print("Input amplitude dimension error. Input amplitude should be list with ", str(non_rad_count), " elements")
            return 0
        
    def _convert_input(self, input_amplitudes):
        total_mode_num = self._radiation_mode_mask.shape[1]
        initial_mode_mask = self._radiation_mode_mask[0]
        non_rad_count = np.count_nonzero(initial_mode_mask == False)
        rad_count = total_mode_num - non_rad_count

        zeros = np.zeros(rad_count)
        input_amplitudes = np.concatenate([input_amplitudes, zeros])
        return input_amplitudes
    #endregion subtask functions (_is_input_amplitudes_okay, _convert_input)
