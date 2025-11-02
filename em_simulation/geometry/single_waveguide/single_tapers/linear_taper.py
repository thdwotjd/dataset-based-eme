import numpy as np
from scipy.interpolate import interp1d

from ....data_updater.data_updater import DataUpdater
from ..single_waveguide import SingleWaveguide


class LinearTaper(SingleWaveguide):
    def __init__(self, dataset: DataUpdater, input_width, output_width, length, prop_angle = 0, limit_mode_number = 0):
        """Create a straight linear taper between two waveguide widths.

        :param dataset: Modal dataset providing overlap and neff information.
        :type dataset: DataUpdater
        :param input_width: Width at the taper entrance in meters.
        :type input_width: float
        :param output_width: Width at the taper exit in meters.
        :type output_width: float
        :param length: Physical taper length in meters.
        :type length: float
        :param prop_angle: Propagation angle referenced to the optic axis.
        :type prop_angle: float
        :param limit_mode_number: Maximum number of modes to retrieve from the
            dataset (``0`` uses all available modes).
        :type limit_mode_number: int
        """
        self._input_width = input_width
        self._output_width = output_width
        self._total_length = length
        self._prop_angle = prop_angle
        SingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)

        resolution = 10
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

    def _calc_xy(self):
        x = np.linspace(0, self._total_length, self._resolution)
        y = np.zeros(self._resolution)
        return x, y
