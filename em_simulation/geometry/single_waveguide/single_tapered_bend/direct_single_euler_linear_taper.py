import numpy as np
from scipy.interpolate import interp1d

from ...bend_shapes.partial_euler import Euler
from ....data_extractor.data_extractor import DataExtractor
from ..direct_single_waveguide import DirectSingleWaveguide



class DirectSingleLinearTaperEulerBend(DirectSingleWaveguide, Euler):
    # def __init__(self, dataset: DataUpdater, top_width_input, top_width_output, input_angle_deg, effective_radius, bend_angle_deg, p, resolution=1000, limit_mode_number=0):
    def __init__(self, dataset: DataExtractor, **params):
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
        if resolution is None: resolution = 50
        self.function_resolution = 1000
        if limit_mode_number is None: limit_mode_number = False
        
        DirectSingleWaveguide.__init__(self, dataset, limit_mode_number= limit_mode_number)
        Euler.__init__(self,p,input_angle_deg, bend_angle_deg, effective_radius, self.function_resolution)
        self._top_width_input = top_width_input
        self._top_width_output = top_width_output
        self.resolution = resolution

        self._prop_length, self._curvature, self._prop_angle, self._ds = Euler._calc_parameters(self)

    def _parametrized_function(self):
        top_widths = np.linspace(self._top_width_input, self._top_width_output, self.function_resolution)
        top_widths_pro_length = np.linspace(0, self._prop_length[-1], self.function_resolution)

        curvature_function = interp1d(self._prop_length, self._curvature, kind='linear')
        prop_angle_function = interp1d(self._prop_length, self._prop_angle, kind='linear')
        top_width_function = interp1d(top_widths_pro_length, top_widths, kind='linear')

        return top_width_function, curvature_function, prop_angle_function
    
    def calc_total_length(self):
        return self._prop_length[-1]
    
    #region plot
    def plot_2D_structure(self):
        return Euler.plot_2D_structure(self)
    
    def plot_structure_parameters(self):
        return DirectSingleWaveguide.plot_structure_parameters(self)
