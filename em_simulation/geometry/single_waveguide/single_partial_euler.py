import numpy as np
from scipy.interpolate import interp1d

from ...data_updater.data_updater import DataUpdater
from ..bend_shapes.partial_euler import PartialEulerCurve
from ..bend_shapes.partial_euler import Euler
from ..bend_shapes.partial_euler import PartialEulerBend

from .single_waveguide import SingleWaveguide


class SinglePartialEuler(SingleWaveguide, PartialEulerCurve):
    def __init__(self, dataset: DataUpdater, top_width, input_angle, total_length, bend_angle, p, resolution=1000, limit_mode_number=0):
        SingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)
        PartialEulerCurve.__init__(self, input_angle, total_length, bend_angle, p, resolution)
        self._top_width = top_width

        self._prop_length, self._curvature, self._prop_angle, self._ds = PartialEulerCurve._calc_parameters()

    def _parametrized_function(self):
        top_widths = self._top_width * np.ones(self._num_points)
        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(self._prop_length, top_widths)

        return top_width_function, curvature_function, prop_angle_function

# class SingleEuler(SingleWaveguide, Euler):
class SingleEuler(SingleWaveguide, PartialEulerBend):
    def __init__(self, dataset: DataUpdater, top_width, input_angle_deg, effective_radius, bend_angle_deg, p, resolution=1000, limit_mode_number=0):
        SingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)
        PartialEulerBend.__init__(self,p,input_angle_deg, bend_angle_deg, effective_radius, resolution)
        self._top_width = top_width

        self._prop_length, self._curvature, self._prop_angle = PartialEulerBend._calc_parameters(self)

    def _parametrized_function(self):
        top_widths = self._top_width * np.ones(len(self._prop_length))
        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(self._prop_length, top_widths)

        return top_width_function, curvature_function, prop_angle_function
    
    #region plot
    def plot_2D_structure(self):
        return SingleWaveguide.plot_2D_structure(self)
    
    def plot_structure_parameters(self):
        return SingleWaveguide.plot_structure_parameters(self)

    #endregion plot

    def _calc_xy(self):
        return PartialEulerBend._calc_xy(self)
    
    def calc_total_length(self):
        return PartialEulerBend.calc_total_length(self)
