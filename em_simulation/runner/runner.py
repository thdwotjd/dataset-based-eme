from ..propagator.propagator import Propagator
from .single_runner import SingleRunner
from .multi_runner import MultiRunner

class Runner():
    def __init__(self, propagator_instance: Propagator, input_amplitudes:list = None):
        """Controller that wraps single- and multi-section propagators.

        :param propagator_instance: Prepared propagator that provides transfer
            and scattering matrices.
        :type propagator_instance: Propagator
        :param input_amplitudes: Optional launch amplitudes applied at
            instantiation.
        :type input_amplitudes: list, optional
        """
        if not issubclass(type(propagator_instance), Propagator):
            print("propagator_instance should be instance of the subclass of the Propagator")
            return 0
        
        if propagator_instance._is_multipropagator:
            runner = MultiRunner(propagator_instance.propagator, input_amplitudes)
            self.runner = runner
            self._is_multirunner = 1
        else:
            runner = SingleRunner(propagator_instance.propagator, input_amplitudes)
            self.runner = runner
            self._is_multirunner = 0
    
    def propagate(self, input_amplitudes: list):
        """Propagate through each segment and record sectional amplitudes.

        :param input_amplitudes: Complex amplitudes for the launched modes.
        :type input_amplitudes: list
        :returns: Array of forward/backward amplitudes at each section.
        :rtype: np.ndarray
        """
        return self.runner.propagate(input_amplitudes)
    
    def propagate_lumped_smatrix(self, input_amplitudes: list):
        """Apply the lumped scattering matrix to the given input amplitudes.

        :param input_amplitudes: Complex amplitudes for the launched modes.
        :type input_amplitudes: list
        :returns: Output amplitudes after the lumped matrix.
        :rtype: np.ndarray
        """
        return self.runner.propagate_lumped_smatrix(input_amplitudes)

    def plot_complexplane(self, mode_nums = None):
        return self.runner.plot_complexplane(mode_nums)

    def plot_intensity_along_propagation(self, show_radiation_mode = False, mode_nums = None):
        """Plot normalized intensities along the propagation path.

        :param show_radiation_mode: Include radiation modes when ``True``.
        :type show_radiation_mode: bool
        :param mode_nums: Optional subset of mode indices to display.
        :type mode_nums: list or None
        :returns: Dictionary with propagation lengths and intensity traces.
        :rtype: dict
        """
        return self.runner.plot_intensity_along_propagation(show_radiation_mode, mode_nums)

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
        return self.runner.sweep_geometry_length(length_1, length_2, num_points, show_radiation_mode)
