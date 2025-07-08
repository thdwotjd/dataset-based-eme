import numpy as np
from scipy.interpolate import interp1d

from ...data_extractor.data_extractor import DataExtractor
from ..bend_shapes.partial_euler import PartialEulerCurve
from ..bend_shapes.partial_euler import Euler
from .direct_single_waveguide import DirectSingleWaveguide


class DirectSinglePartialEuler(DirectSingleWaveguide, PartialEulerCurve):
    def __init__(self, dataset: DataExtractor, top_width, input_angle, total_length, bend_angle, p, resolution=40, limit_mode_number=0):
        self.function_resolution = 1000
        self.resolution = 40
        DirectSingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)
        PartialEulerCurve.__init__(self, input_angle, total_length, bend_angle, p, self.function_resolution)
        self._top_width = top_width

        self._prop_length, self._curvature, self._prop_angle, self._ds = PartialEulerCurve._calc_parameters()

    def _parametrized_function(self):
        top_widths = self._top_width * np.ones(self.function_resolution)
        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(self._prop_length, top_widths)

        return top_width_function, curvature_function, prop_angle_function
    
    def calc_total_length(self):
        return self._prop_length[-1]
    
class DirectSingleEuler(DirectSingleWaveguide, Euler):
    def __init__(self, dataset: DataExtractor, top_width, input_angle_deg, effective_radius, bend_angle_deg, p, resolution=40, limit_mode_number=0):
        self.function_resolution = 1000
        self.resolution = 40
        DirectSingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)
        Euler.__init__(self,p,input_angle_deg, bend_angle_deg, effective_radius, self.function_resolution)
        self._top_width = top_width

        self._prop_length, self._curvature, self._prop_angle, self._ds = Euler._calc_parameters(self)

    def _parametrized_function(self):
        top_widths = self._top_width * np.ones(len(self._prop_length))
        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(self._prop_length, top_widths)

        return top_width_function, curvature_function, prop_angle_function
    
    def calc_total_length(self):
        return self._prop_length[-1]
    
    #region plot
    def plot_2D_structure(self):
        return Euler.plot_2D_structure(self)
    
    def plot_structure_parameters(self):
        return DirectSingleWaveguide.plot_structure_parameters(self)

    #endregion plot