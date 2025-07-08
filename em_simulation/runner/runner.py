from ..propagator.propagator import Propagator
from .single_runner import SingleRunner
from .multi_runner import MultiRunner

class Runner():
    def __init__(self, propagator_instance: Propagator, input_amplitudes:list = None):
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
        return self.runner.propagate(input_amplitudes)
    
    def propagate_lumped_smatrix(self, input_amplitudes: list):
        return self.runner.propagate_lumped_smatrix(input_amplitudes)

    def plot_complexplane(self, mode_nums = None):
        return self.runner.plot_complexplane(mode_nums)

    def plot_intensity_along_propagation(self, show_radiation_mode = False, mode_nums = None):
        return self.runner.plot_intensity_along_propagation(show_radiation_mode, mode_nums)

    def sweep_geometry_length(self, length_1, length_2, num_points, show_radiation_mode = False):
        return self.runner.sweep_geometry_length(length_1, length_2, num_points, show_radiation_mode)