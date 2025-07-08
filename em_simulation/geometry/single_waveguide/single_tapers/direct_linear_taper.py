import numpy as np
from scipy.interpolate import interp1d

from ....data_extractor.data_extractor import DataExtractor
from ..direct_single_waveguide import DirectSingleWaveguide


class DirectLinearTaper(DirectSingleWaveguide):
    def __init__(self, dataset: DataExtractor, input_width, output_width, length, prop_angle = 0, limit_mode_number = 0, resolution = 20, use_existing_data = False):
        DirectSingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number, use_existing_data= use_existing_data)
        self._input_width = input_width
        self._output_width = output_width
        self._total_length = length
        self._prop_angle = prop_angle

        self.resolution = resolution
        self._prop_lengths = np.linspace(0, length, resolution)
        self._top_widths = np.linspace(input_width, output_width, resolution)
        self._prop_angles = np.ones(shape=(resolution,), dtype= float) * prop_angle
        self._curvatures = np.zeros(shape = (resolution,))

    def _parametrized_function(self):
        curvature_function = interp1d(self._prop_lengths, self._curvatures)
        prop_angle_function = interp1d(self._prop_lengths, self._prop_angles)
        top_width_function = interp1d(self._prop_lengths, self._top_widths)

        return top_width_function, curvature_function, prop_angle_function
    
    def calc_total_length(self):
        return self._total_length