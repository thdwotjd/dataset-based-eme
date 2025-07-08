import numpy as np
from scipy.interpolate import interp1d

from ...bend_shapes.partial_euler import Euler
from ...bend_shapes.partial_euler import PartialEulerBend
from ....data_updater.data_updater import DataUpdater
from ..single_waveguide import SingleWaveguide



# class SingleLinearTaperEulerBend(SingleWaveguide, Euler):
class SingleLinearTaperEulerBend(SingleWaveguide, PartialEulerBend):

    # def __init__(self, dataset: DataUpdater, top_width_input, top_width_output, input_angle_deg, effective_radius, bend_angle_deg, p, resolution=1000, limit_mode_number=0):
    def __init__(self, dataset: DataUpdater, **params):
        """
        Parameters:
            - dataset
            - top_width_input
            - top_width_output
            - input_angle_deg
            - bend_angle_deg
            - effective_radius
            - p
            - resolution
            - limit_mode_number
        """
        # Access parameters using dictionary keys
        top_width_input = params.get('top_width_input')
        top_width_output = params.get('top_width_output')
        input_angle_deg = params.get('input_angle_deg')
        bend_angle_deg = params.get('bend_angle_deg')
        effective_radius = params.get('effective_radius')
        p = params.get('p')
        resolution = params.get('resolution')
        limit_mode_number = params.get('limit_mode_number')
        
        is_input_okay = not(top_width_input is None or top_width_output is None or input_angle_deg is None or bend_angle_deg is None\
                         or effective_radius is None or p is None)
        if not is_input_okay:
            raise ValueError("Missing required parameters")
        if resolution is None: resolution = 1000
        if limit_mode_number is None: limit_mode_number = False
        
        SingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)
        PartialEulerBend.__init__(self,p,input_angle_deg, bend_angle_deg, effective_radius, resolution)
        self._top_width_input = top_width_input
        self._top_width_output = top_width_output
        self._resolution = resolution

        self._prop_length, self._curvature, self._prop_angle = PartialEulerBend._calc_parameters(self)

    def _parametrized_function(self):
        top_widths = np.linspace(self._top_width_input, self._top_width_output, self._resolution)
        top_widths_prop_length = np.linspace(0, self._prop_length[-1], self._resolution)

        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(top_widths_prop_length, top_widths, kind='linear')

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

