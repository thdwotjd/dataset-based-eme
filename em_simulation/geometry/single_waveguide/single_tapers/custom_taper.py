import numpy as np
from scipy.interpolate import interp1d

from ....data_updater.data_updater import DataUpdater
from ..single_waveguide import SingleWaveguide

class CustomTaper(SingleWaveguide):
    def __init__(self, dataset: DataUpdater, length, num_sections, width_list, prop_angle = 0, resolution = 3000, limit_mode_number = 0, verbose = True):
        """Create a piecewise-linear taper with user-defined section widths. Widths in width_list are corresponding to equally spaced sections along the taper length.

        :param dataset: Modal dataset supplying overlap and effective index
            information.
        :type dataset: DataUpdater
        :param length: Total taper length in meters.
        :type length: float
        :param num_sections: Number of discrete sections describing the taper.
        :type num_sections: int
        :param width_list: Sequence of waveguide widths applied to each section.
        :type width_list: array-like
        :param prop_angle: Propagation angle relative to the reference axis in
            degrees.
        :type prop_angle: float
        :param resolution: Number of samples used by the parent waveguide class.
        :type resolution: int
        :param limit_mode_number: Maximum number of modes retrieved from the
            dataset (``0`` uses all available modes).
        :type limit_mode_number: int
        :param verbose: Emit diagnostic messages when ``True``.
        :type verbose: bool
        """
        SingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number, resolution=resolution, verbose=verbose)
        self._total_length = length
        self._prop_angle = prop_angle
        if num_sections != len(width_list):
            print("The width_list should have the same number of elements as num_sections.")
            return
        self._width_list = np.array(width_list)

        resolution = num_sections
        self._prop_lengths = np.linspace(0, length, resolution)
        self._top_widths = self._width_list
        self._prop_angles = np.ones(shape=(resolution,), dtype= float) * prop_angle
        self._curvatures = np.zeros(shape = (resolution,))

        pass

    def _parametrized_function(self):
        curvature_function = interp1d(self._prop_lengths, self._curvatures)
        prop_angle_function = interp1d(self._prop_lengths, self._prop_angles)
        top_width_function = interp1d(self._prop_lengths, self._top_widths)

        return top_width_function, curvature_function, prop_angle_function
    
    #region abstractmethods
    def calc_total_length(self):
        return self._total_length
    
    def _calc_xy(self):
        x = np.linspace(0, self._total_length, self._resolution)
        y = np.zeros(self._resolution)
        return x, y
    #endregion abstractmethods
    
